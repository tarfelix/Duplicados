#!/usr/bin/env python3
"""
Script STANDALONE para extrair 50 casos reais de duplicacao.
Nao depende do repositorio - roda em qualquer pasta.

INSTRUCOES:
  1. Instale dependencias (se nao tiver):
       pip install mysql-connector-python pandas rapidfuzz unidecode
  2. Rode:
       python extrair_standalone.py
  3. Cole o conteudo do arquivo 'casos_reais.json' na conversa com o Claude.
"""

import re
import json
import sys
from datetime import date, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# ============================================================
# CONFIGURACAO - Ajuste se necessario
# ============================================================
MYSQL_CONFIG = {
    "host": "40.88.40.110",
    "user": "tarcisio",
    "password": "123qwe",
    "database": "zion_flow",
    "connect_timeout": 15,
}
DIAS_HISTORICO = 30
MIN_SIM = 90        # 0-100
MIN_CONTAINMENT = 55  # 0-100
MIN_TOKENS = 3
USE_CNJ = True
# ============================================================

print("=" * 60)
print("EXTRATOR DE CASOS DE DUPLICACAO (standalone)")
print("=" * 60)

# ------------------------------------------------------------------
# Dependencias
# ------------------------------------------------------------------
try:
    import mysql.connector
except ImportError:
    print("ERRO: mysql-connector-python nao instalado.")
    print("Rode: pip install mysql-connector-python")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERRO: pandas nao instalado. Rode: pip install pandas")
    sys.exit(1)

try:
    from rapidfuzz import fuzz
except ImportError:
    print("ERRO: rapidfuzz nao instalado. Rode: pip install rapidfuzz")
    sys.exit(1)

try:
    from unidecode import unidecode
except ImportError:
    print("ERRO: unidecode nao instalado. Rode: pip install unidecode")
    sys.exit(1)


# ------------------------------------------------------------------
# Matcher (copiado do backend para ser standalone)
# ------------------------------------------------------------------
CNJ_RE = re.compile(r"(?:\b|^)(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})(?:\b|$)")
URL_RE = re.compile(r"https?://\S+")
DATENUM_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
NUM_RE = re.compile(r"\b\d+\b")

STOPWORDS_BASE = set("""
    de da do das dos e em a o os as na no para por com que ao aos às à um uma umas uns
    tipo titulo inteiro teor publicado publicacao disponibilizacao orgao vara tribunal
    processo recurso intimacao notificacao justica nacional diario djen poder judiciario trabalho
""".split())


def extract_meta(text: str) -> Dict[str, str]:
    t = text or ""
    meta = {}
    cnj_match = CNJ_RE.search(t)
    cnj = cnj_match.group(1) if cnj_match else None
    if not cnj:
        proc_match = re.search(r"PROCESSO:\s*([0-9\-.]+)", t, re.IGNORECASE)
        if proc_match:
            cnj = proc_match.group(1)
    meta["processo"] = cnj or ""
    patterns = {
        "orgao": r"\bORGAO:\s*([^-\n\r]+)",
        "vara": r"\bVARA\s+DO\s+TRABALHO\s+[^-\n\r]+",
        "tipo_doc": r"\bTIPO\s+DE\s+DOCUMENTO:\s*([^-]+)",
        "tipo_com": r"\bTIPO\s+DE\s+COMUNICACAO:\s*([^-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, t, re.IGNORECASE)
        if match:
            meta[key] = match.group(1).strip() if key != "vara" else match.group(0).strip()
    return meta


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub(" url ", text)
    t = CNJ_RE.sub(" numproc ", t)
    t = DATENUM_RE.sub(" data ", t)
    t = NUM_RE.sub(" # ", t)
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    tokens = [w for w in t.split() if w not in STOPWORDS_BASE]
    return " ".join(tokens)


def combined_score(a_norm, b_norm, meta_a, meta_b):
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm)
    sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    tokens_a, tokens_b = a_norm.split(), b_norm.split()
    if not tokens_a or not tokens_b:
        contain = 0.0
    else:
        small, big = (tokens_a, set(tokens_b)) if len(tokens_a) <= len(tokens_b) else (tokens_b, set(tokens_a))
        contain = 100.0 * (sum(1 for t in small if t in big) / len(small))

    len_a, len_b = len(a_norm), len(b_norm)
    lp = 1.0
    if len_a == 0 or len_b == 0:
        lp = 0.7
    elif len_a > 0 and len_b > 0:
        diff_ratio = abs(len_a - len_b) / max(len_a, len_b)
        lp = max(0.7, 1.0 - diff_ratio * 0.4)
        if (meta_a.get("processo") and meta_a["processo"] == meta_b.get("processo")) or \
           (meta_a.get("orgao") and meta_a["orgao"] == meta_b.get("orgao")):
            lp = max(lp, 0.85)

    base_score = 0.6 * set_ratio + 0.2 * sort_ratio + 0.2 * contain
    base_after_lp = base_score * lp
    bonus = 0
    if base_after_lp >= 80:
        if meta_a.get("processo") and meta_a["processo"] == meta_b.get("processo"):
            bonus += 6
        if meta_a.get("orgao") and meta_a["orgao"] == meta_b.get("orgao"):
            bonus += 3
        if meta_a.get("tipo_doc") and meta_a["tipo_doc"] == meta_b.get("tipo_doc"):
            bonus += 3
        if meta_a.get("tipo_com") and meta_a["tipo_com"] == meta_b.get("tipo_com"):
            bonus += 2
        bonus = min(bonus, 5)
    final_score = max(0.0, min(100.0, base_after_lp + bonus))
    return final_score, {"set": set_ratio, "sort": sort_ratio, "contain": contain, "len_pen": lp, "bonus": bonus}


def create_groups(df, min_sim, min_containment, use_cnj, min_tokens):
    if df.empty:
        return []
    work = df.copy()
    work["_meta"] = work["Texto"].apply(extract_meta)
    work["_norm"] = work["Texto"].apply(normalize_text)

    buckets = defaultdict(list)
    for i, row in work.iterrows():
        key = f"folder::{row.get('activity_folder', 'SEM')}"
        if use_cnj:
            key += f"::cnj::{row['_meta']['processo'] or 'SEM'}"
        buckets[key].append(i)

    groups = []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        n = len(idxs)
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            cluster = [idxs[i]]
            queue = deque([i])
            visited[i] = True
            while queue:
                curr = queue.popleft()
                for j in range(n):
                    if not visited[j]:
                        ri, rj = work.loc[idxs[curr]], work.loc[idxs[j]]
                        ni, nj = ri["_norm"], rj["_norm"]
                        ti = ni.split() if isinstance(ni, str) else []
                        tj = nj.split() if isinstance(nj, str) else []
                        if len(ti) < min_tokens or len(tj) < min_tokens:
                            continue
                        score, details = combined_score(ni, nj, ri["_meta"], rj["_meta"])
                        if score >= min_sim and details["contain"] >= min_containment:
                            visited[j] = True
                            cluster.append(idxs[j])
                            queue.append(j)
            if len(cluster) > 1:
                rows = work.loc[cluster].sort_values("activity_date", ascending=False)
                groups.append(rows.to_dict("records"))
    return groups


# ------------------------------------------------------------------
# Funcoes auxiliares de analise
# ------------------------------------------------------------------
def detect_source(text):
    t = (text or "").upper()
    if "AASP" in t:
        return "AASP"
    if "ADVISE" in t:
        return "ADVISE"
    if "DJEN" in t:
        return "DJEN"
    if "DJE" in t or "DIARIO ELETRONICO" in t:
        return "DJE"
    return "OTHER"


def has_retificacao(text):
    return bool(re.search(
        r"\b(RETIFICA|REPUBLICA|ERRATA|ONDE SE L[EÊ])\b",
        text or "", re.IGNORECASE
    ))


def classify_dup_type(items):
    sources = set(i["source"] for i in items)
    dates = set(i["date"][:10] if i["date"] else "?" for i in items)
    if len(sources) > 1 and "OTHER" not in sources:
        return "CROSS_SOURCE"
    if len(dates) == 1:
        return "SAME_DAY"
    return "DIFF_DAY"


# ------------------------------------------------------------------
# 1. Conectar e carregar dados
# ------------------------------------------------------------------
print(f"\nConectando ao MySQL ({MYSQL_CONFIG['host']})...")
try:
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor(dictionary=True)
    print("Conectado!")
except Exception as e:
    print(f"ERRO de conexao: {e}")
    sys.exit(1)

limite = date.today() - timedelta(days=DIAS_HISTORICO)
query = """
    SELECT activity_id, activity_folder, user_profile_name,
           activity_date, activity_status, Texto
    FROM ViewGrdAtividadesTarcisio
    WHERE activity_type='Verificar'
      AND (activity_status='Aberta' OR DATE(activity_date) >= %s)
"""
print(f"Carregando atividades (ultimos {DIAS_HISTORICO} dias)...")
cursor.execute(query, (limite,))
rows = cursor.fetchall()
cursor.close()
conn.close()

if not rows:
    print("Nenhum dado retornado. Verifique a conexao e os filtros.")
    sys.exit(1)

df = pd.DataFrame(rows)
df["activity_id"] = df["activity_id"].astype(str)
df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
df["Texto"] = df["Texto"].fillna("").astype(str)
df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
df = df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])

print(f"Total atividades: {len(df)}")
print(f"Pastas: {df['activity_folder'].nunique()}")
print(f"Status: {dict(df['activity_status'].value_counts())}")

# ------------------------------------------------------------------
# 2. Rodar matcher
# ------------------------------------------------------------------
print(f"\nRodando matcher (min_sim={MIN_SIM}, min_containment={MIN_CONTAINMENT})...")
groups = create_groups(df, MIN_SIM, MIN_CONTAINMENT, USE_CNJ, MIN_TOKENS)
print(f"Grupos encontrados: {len(groups)}")
print(f"Atividades em grupos: {sum(len(g) for g in groups)}")

if not groups:
    print("Nenhum grupo encontrado. Tente reduzir MIN_SIM.")
    sys.exit(1)

# ------------------------------------------------------------------
# 3. Analisar 50 grupos
# ------------------------------------------------------------------
print("\nAnalisando 50 grupos em detalhe...")
results = []

for idx, group in enumerate(groups[:50]):
    items = []
    norms = []
    metas = []

    for row in group:
        texto = row.get("Texto", "")
        meta = extract_meta(texto)
        norm = normalize_text(texto)
        source = detect_source(texto)
        norms.append(norm)
        metas.append(meta)

        items.append({
            "activity_id": str(row.get("activity_id", "")),
            "folder": row.get("activity_folder", ""),
            "status": row.get("activity_status", ""),
            "date": str(row.get("activity_date", ""))[:19],
            "user": row.get("user_profile_name", ""),
            "source": source,
            "processo": meta.get("processo", ""),
            "orgao": meta.get("orgao", ""),
            "tipo_doc": meta.get("tipo_doc", ""),
            "tipo_com": meta.get("tipo_com", ""),
            "text_length": len(texto),
            "text_preview": texto[:500],
            "text_tail": texto[-300:] if len(texto) > 500 else "",
            "norm_tokens_count": len(norm.split()) if norm else 0,
            "has_retificacao": has_retificacao(texto),
            "cnj_count": len(CNJ_RE.findall(texto)),
        })

    pair_scores = []
    for a_idx in range(len(group)):
        for b_idx in range(a_idx + 1, len(group)):
            score, details = combined_score(norms[a_idx], norms[b_idx], metas[a_idx], metas[b_idx])
            pair_scores.append({
                "a_id": items[a_idx]["activity_id"],
                "b_id": items[b_idx]["activity_id"],
                "score": round(score, 2),
                "set_ratio": round(details["set"], 2),
                "sort_ratio": round(details["sort"], 2),
                "containment": round(details["contain"], 2),
                "len_penalty": round(details["len_pen"], 4),
                "bonus": details["bonus"],
            })

    dup_type = classify_dup_type(items)
    avg_score = round(sum(p["score"] for p in pair_scores) / len(pair_scores), 2) if pair_scores else 0
    min_score = min((p["score"] for p in pair_scores), default=0)

    results.append({
        "group_num": idx + 1,
        "size": len(group),
        "dup_type": dup_type,
        "folder": items[0]["folder"],
        "avg_score": avg_score,
        "min_score": min_score,
        "unique_sources": sorted(set(i["source"] for i in items)),
        "unique_processos": sorted(set(i["processo"] for i in items if i["processo"])),
        "any_retificacao": any(i["has_retificacao"] for i in items),
        "items": items,
        "pair_scores": pair_scores,
    })

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
summary = {
    "total_activities": len(df),
    "total_groups": len(groups),
    "total_in_groups": sum(len(g) for g in groups),
    "analyzed_groups": len(results),
    "params": {"min_sim": MIN_SIM, "min_containment": MIN_CONTAINMENT, "min_tokens": MIN_TOKENS, "use_cnj": USE_CNJ},
    "dup_type_distribution": {},
    "source_distribution": {},
    "score_ranges": {"90-100": 0, "80-89": 0, "70-79": 0, "<70": 0},
    "group_size_distribution": {},
    "folders": sorted(df["activity_folder"].dropna().unique().tolist()),
    "status_counts": {k: int(v) for k, v in df["activity_status"].value_counts().items()},
}

for r in results:
    dt = r["dup_type"]
    summary["dup_type_distribution"][dt] = summary["dup_type_distribution"].get(dt, 0) + 1
    for s in r["unique_sources"]:
        summary["source_distribution"][s] = summary["source_distribution"].get(s, 0) + 1
    for ps in r["pair_scores"]:
        sc = ps["score"]
        if sc >= 90:
            summary["score_ranges"]["90-100"] += 1
        elif sc >= 80:
            summary["score_ranges"]["80-89"] += 1
        elif sc >= 70:
            summary["score_ranges"]["70-79"] += 1
        else:
            summary["score_ranges"]["<70"] += 1
    sz = str(r["size"])
    summary["group_size_distribution"][sz] = summary["group_size_distribution"].get(sz, 0) + 1

output = {"summary": summary, "groups": results}

# ------------------------------------------------------------------
# 5. Salvar
# ------------------------------------------------------------------
output_path = "casos_reais.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

import os
file_size = os.path.getsize(output_path) / 1024

print(f"\nSalvo em: {output_path} ({file_size:.1f} KB)")
print()
print("=" * 60)
print("RESUMO:")
print(f"  Atividades: {summary['total_activities']}")
print(f"  Grupos total: {summary['total_groups']}")
print(f"  Analisados: {summary['analyzed_groups']}")
print(f"  Tipos: {summary['dup_type_distribution']}")
print(f"  Sources: {summary['source_distribution']}")
print(f"  Scores: {summary['score_ranges']}")
print(f"  Tamanhos: {summary['group_size_distribution']}")
print(f"  Pastas: {summary['folders']}")
print("=" * 60)

if file_size > 200:
    print(f"\nARQUIVO GRANDE ({file_size:.0f} KB)!")
    print("Gerando versao compacta (10 grupos)...")
    compact = {"summary": summary, "groups": results[:10]}
    with open("casos_reais_compact.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2, default=str)
    compact_size = os.path.getsize("casos_reais_compact.json") / 1024
    print(f"Versao compacta: casos_reais_compact.json ({compact_size:.1f} KB)")
    print("\nCole o conteudo de casos_reais_compact.json na conversa.")
else:
    print("\nCole o conteudo de casos_reais.json na conversa.")
