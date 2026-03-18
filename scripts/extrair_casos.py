#!/usr/bin/env python3
"""
Script para extrair 50 casos reais de duplicacao e salvar em JSON.

Rode na sua maquina local (que tem acesso ao MySQL):
  cd Duplicados
  python scripts/extrair_casos.py

O resultado sera salvo em: scripts/casos_reais.json
Cole o conteudo desse arquivo na conversa com o Claude.
"""

import sys
import os
import json
import re
from pathlib import Path

# Adicionar o diretorio raiz ao path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.mysql_client import carregar_dados_mysql
from backend.services.matcher import (
    create_groups, extract_meta, normalize_text, combined_score, CNJ_RE
)
from backend.config import get_settings

settings = get_settings()

# ------------------------------------------------------------------
# 1. Carregar dados do MySQL
# ------------------------------------------------------------------
print("Conectando ao MySQL e carregando atividades (ultimos 30 dias)...")
df = carregar_dados_mysql(dias_historico=30)

if df.empty:
    print("ERRO: Nenhum dado retornado. Verifique a conexao MySQL e o .env")
    sys.exit(1)

print(f"Total de atividades carregadas: {len(df)}")
print(f"Pastas unicas: {df['activity_folder'].nunique()}")
print(f"Status: {dict(df['activity_status'].value_counts())}")

# ------------------------------------------------------------------
# 2. Rodar o matcher
# ------------------------------------------------------------------
params = {
    'min_sim': settings.similarity_min_sim_global,
    'min_containment': settings.similarity_min_containment,
    'min_tokens': settings.similarity_min_tokens_to_match,
    'use_cnj': True,
    'stopwords_extra': settings.stopwords_extra_list,
    'cutoffs_map': settings.cutoffs_map,
}
print(f"Params: min_sim={params['min_sim']}, min_containment={params['min_containment']}")
print("Rodando matcher (pode demorar alguns segundos)...")
groups = create_groups(df, params)
print(f"Total de grupos encontrados: {len(groups)}")
print(f"Total de atividades em grupos: {sum(len(g) for g in groups)}")

if not groups:
    print("Nenhum grupo encontrado. Tente aumentar dias_historico ou reduzir min_sim.")
    sys.exit(1)


# ------------------------------------------------------------------
# 3. Funcoes auxiliares
# ------------------------------------------------------------------
def detect_source(text: str) -> str:
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


def classify_dup_type(group_data):
    """Classifica o tipo de duplicacao do grupo."""
    sources = set(item["source"] for item in group_data)
    dates = set(item["date"][:10] if item["date"] else "?" for item in group_data)

    if len(sources) > 1 and "OTHER" not in sources:
        return "CROSS_SOURCE"
    if len(dates) == 1:
        return "SAME_DAY"
    return "DIFF_DAY"


def has_retificacao(text: str) -> bool:
    return bool(re.search(
        r"\b(RETIFICA[CÇ][AÃ]O|REPUBLICA[CÇ][AÃ]O|ERRATA|ONDE SE L[EÊ])\b",
        text or "", re.IGNORECASE
    ))


def count_cnjs(text: str) -> int:
    return len(CNJ_RE.findall(text or ""))


# ------------------------------------------------------------------
# 4. Analisar os primeiros 50 grupos
# ------------------------------------------------------------------
print("\nAnalisando 50 grupos em detalhe...")
stopwords = settings.stopwords_extra_list
results = []

for idx, group in enumerate(groups[:50]):
    items = []
    norms = []
    metas = []

    for row in group:
        texto = row.get("Texto", "")
        meta = extract_meta(texto)
        norm = normalize_text(texto, stopwords)
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
            "text_preview": texto[:300],
            "text_tail": texto[-200:] if len(texto) > 300 else "",
            "norm_tokens_count": len(norm.split()) if norm else 0,
            "has_retificacao": has_retificacao(texto),
            "cnj_count": count_cnjs(texto),
        })

    # Calcular scores entre todos os pares
    pair_scores = []
    for a_idx in range(len(group)):
        for b_idx in range(a_idx + 1, len(group)):
            score, details = combined_score(
                norms[a_idx], norms[b_idx], metas[a_idx], metas[b_idx]
            )
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
    avg_score = (
        round(sum(p["score"] for p in pair_scores) / len(pair_scores), 2)
        if pair_scores else 0
    )
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
# 5. Gerar estatisticas resumidas
# ------------------------------------------------------------------
summary = {
    "total_activities": len(df),
    "total_groups": len(groups),
    "total_in_groups": sum(len(g) for g in groups),
    "analyzed_groups": len(results),
    "params": {
        "min_sim": params["min_sim"],
        "min_containment": params["min_containment"],
        "min_tokens": params["min_tokens"],
        "use_cnj": params["use_cnj"],
    },
    "dup_type_distribution": {},
    "source_distribution": {},
    "score_ranges": {"90-100": 0, "80-89": 0, "70-79": 0, "<70": 0},
    "group_size_distribution": {},
    "folders": sorted(df["activity_folder"].unique().tolist()),
    "status_counts": dict(df["activity_status"].value_counts()),
}

for r in results:
    # dup type
    dt = r["dup_type"]
    summary["dup_type_distribution"][dt] = summary["dup_type_distribution"].get(dt, 0) + 1

    # sources
    for s in r["unique_sources"]:
        summary["source_distribution"][s] = summary["source_distribution"].get(s, 0) + 1

    # scores
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

    # sizes
    sz = str(r["size"])
    summary["group_size_distribution"][sz] = summary["group_size_distribution"].get(sz, 0) + 1

output = {
    "summary": summary,
    "groups": results,
}

# ------------------------------------------------------------------
# 6. Salvar
# ------------------------------------------------------------------
output_path = ROOT / "scripts" / "casos_reais.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\nResultado salvo em: {output_path}")
print(f"Tamanho do arquivo: {output_path.stat().st_size / 1024:.1f} KB")
print()
print("=" * 60)
print("RESUMO RAPIDO:")
print(f"  Grupos analisados: {len(results)}")
print(f"  Tipos: {summary['dup_type_distribution']}")
print(f"  Sources: {summary['source_distribution']}")
print(f"  Scores: {summary['score_ranges']}")
print(f"  Tamanhos: {summary['group_size_distribution']}")
print("=" * 60)
print()
print("Agora copie o conteudo de scripts/casos_reais.json")
print("e cole na conversa com o Claude para analise detalhada.")
print()
print("Se o arquivo for muito grande (>100KB), rode com --compact:")
print("  python scripts/extrair_casos.py --compact")

# Modo compacto: so summary + 10 grupos
if "--compact" in sys.argv:
    compact_output = {
        "summary": summary,
        "groups": results[:10],
    }
    compact_path = ROOT / "scripts" / "casos_reais_compact.json"
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact_output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nVersao compacta (10 grupos) salva em: {compact_path}")
    print(f"Tamanho: {compact_path.stat().st_size / 1024:.1f} KB")
