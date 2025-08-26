
# app_optimized.py — Streamlit app para detecção e revisão de publicações duplicadas
# com foco em performance e utilidade.
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine

from rapidfuzz import fuzz
from rapidfuzz.fuzz import QRatio

# ======== Configurações ========

st.set_page_config(page_title="Revisão de Duplicidades — Soares Picon", layout="wide")

# ======== Helpers de estado =========

class SK:
    GROUP_STATES = "group_states"              # dict[group_key] -> {principal_id, cancelados:set, open_compare: str|None}
    ANALYSIS_RESULTS = "analysis_results"      # {"sig": signature, "groups": [ {"ids":[...], "bucket": "..."} ]}
    PARAMS = "params"
    DRY_RUN = "dry_run"

for k, v in [
    (SK.GROUP_STATES, {}),
    (SK.ANALYSIS_RESULTS, {}),
    (SK.PARAMS, {}),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ======== Conexões e clientes ========

@st.cache_resource(show_spinner=False)
def engine() -> Engine:
    # Ajuste a chave conforme seu secrets.toml
    url = st.secrets["mysql"]["url"]
    return create_engine(url, pool_pre_ping=True)

@st.cache_resource(show_spinner=False)
def api_client():
    from api_functions_retry_opt import HttpClientRetry, RetryConfig  # local import para facilitar deploy
    cfg = RetryConfig(
        max_retries=5,
        backoff_factor=0.6,
        calls_per_second=float(st.secrets.get("api_client", {}).get("calls_per_second", 4.0)),
        timeout=float(st.secrets.get("api_client", {}).get("timeout", 30.0)),
        dry_run=bool(st.session_state.get(SK.DRY_RUN, False)),
    )
    token = st.secrets.get("api_client", {}).get("token", "")
    base_url = st.secrets.get("api_client", {}).get("base_url", "http://localhost:8080")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return HttpClientRetry(base_url=base_url, headers=headers, retry=cfg)

# ======== Carregamento de dados (leve) ========

@st.cache_data(ttl=1800, show_spinner=False)
def carregar_lista_atividades(_eng: Engine, dias_historico: int) -> pd.DataFrame:
    """
    Carrega apenas colunas mínimas necessárias para montar a lista e paginação.
    """
    q = text(
        """
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar'
          AND (activity_status='Aberta' OR activity_date >= DATE_SUB(CURDATE(), INTERVAL :dias DAY))
        ORDER BY activity_date DESC
        """
    )
    with _eng.connect() as conn:
        df = pd.read_sql(q, conn, params={"dias": int(dias_historico)})
    df["activity_id"] = df["activity_id"].astype(str)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_textos_por_id(_eng: Engine, ids: Tuple[str, ...]) -> Dict[str, str]:
    if not ids:
        return {}
    ids_sorted = tuple(sorted(set(ids)))
    stmt = text(
        """
        SELECT activity_id, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_id IN :ids
        """
    ).bindparams(bindparam("ids", expanding=True))
    with _eng.connect() as conn:
        df = pd.read_sql(stmt, conn, params={"ids": list(ids_sorted)})
    df["activity_id"] = df["activity_id"].astype(str)
    return pd.Series(df["Texto"].values, index=df["activity_id"]).to_dict()

# ======== Normalização, meta e score ========

def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_meta(s: str) -> Dict[str, Any]:
    # Exemplos simples; ajuste para sua realidade
    m = re.search(r"\b(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})\b", s)  # padrão CNJ
    cnj = m.group(1) if m else None
    tipo = "despacho" if "despacho" in s else ("intimação" if "intim" in s else None)
    return {"cnj": cnj, "tipo_doc": tipo}

def token_set_containment(tokens_a: List[str], set_b: Set[str]) -> float:
    if not tokens_a:
        return 0.0
    inter = sum(1 for t in tokens_a if t in set_b)
    return 100.0 * (inter / max(1, len(tokens_a)))

def combined_score_fast(a_row: Dict[str, Any], b_row: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    a_norm, b_norm = a_row["_norm"], b_row["_norm"]
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm)
    sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    # containment assimétrico, usa menor lista como base
    if len(a_row["_tokens"]) <= len(b_row["_tokens"]):
        contain = token_set_containment(a_row["_tokens"], b_row["_set"])
    else:
        contain = token_set_containment(b_row["_tokens"], a_row["_set"])
    score = round(0.6 * set_ratio + 0.4 * sort_ratio, 2)
    return score, {"set": set_ratio, "sort": sort_ratio, "contain": contain}

# ======== Agrupamento (ids apenas no cache) ========

@dataclass
class GroupIds:
    ids: List[str]
    bucket: str
    principal_id: Optional[str] = None  # opcional: “best” estimado

class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def group_key_from_ids(ids: Iterable[str]) -> str:
    return hashlib.md5(json.dumps(sorted(set(ids))).encode()).hexdigest()

def block_key(row: Dict[str, Any]) -> str:
    # Pasta + CNJ + primeiras 3 palavras do texto normalizado + tipo_doc
    prefix = " ".join(row["_tokens"][:3]) if row["_tokens"] else ""
    cnj = row["_meta"].get("cnj") or "SEM_CNJ"
    tipo = row["_meta"].get("tipo_doc") or "NA"
    return f"{row['activity_folder']}||{cnj}||{tipo}||{prefix}"

@st.cache_data(ttl=3600, show_spinner=False)
def criar_grupos_ids(
    df_min: pd.DataFrame,
    textos_map: Dict[str, str],
    min_sim_pct: float,
    min_containment_pct: float,
    pre_delta: float,
    use_cnj: bool = True,
    candidate_cap: int = 200,
    max_tokens_per_doc: int = 300,
) -> List[GroupIds]:
    """
    Retorna apenas IDs por grupo para manter cache leve.
    """
    if df_min.empty:
        return []

    # Monta data de trabalho com normalização e meta
    work = df_min.copy()
    work["Texto"] = work["activity_id"].map(textos_map).fillna("")
    work["_norm"] = work["Texto"].map(normalize_text)
    work["_tokens"] = work["_norm"].map(lambda s: s.split())
    work["_tokens"] = work["_tokens"].map(lambda toks: toks[:max_tokens_per_doc])  # limita índice
    work["_set"] = work["_tokens"].map(set)
    work["_meta"] = work["_norm"].map(extract_meta)

    # Bucketing
    buckets: Dict[str, List[int]] = {}
    for idx, row in work.iterrows():
        if use_cnj:
            bkey = block_key(row)
        else:
            # sem cnj no bucket
            prefix = " ".join(row["_tokens"][:3]) if row["_tokens"] else ""
            tipo = row["_meta"].get("tipo_doc") or "NA"
            bkey = f"{row['activity_folder']}||{tipo}||{prefix}"
        buckets.setdefault(bkey, []).append(idx)

    groups: List[GroupIds] = []
    # Progresso por bucket
    progress = st.progress(0.0, text="Agrupando duplicidades...")
    bkeys = list(buckets.keys())

    for bi, bkey in enumerate(bkeys, 1):
        idxs = buckets[bkey]
        if len(idxs) <= 1:
            # grupos unitários não interessam para duplicidade
            progress.progress(bi / len(bkeys))
            continue

        # Índice invertido simples por token
        token_index: Dict[str, List[int]] = {}
        for i in idxs:
            for t in set(work.at[i, "_tokens"]):
                token_index.setdefault(t, []).append(i)

        # Grafo de similaridade: DSU
        pos_map = {i: k for k, i in enumerate(idxs)}
        dsu = DSU(len(idxs))

        # Thresholds
        threshold = float(min_sim_pct)
        pre_cutoff = max(0.0, threshold - float(pre_delta))

        for i in idxs:
            seen: Set[int] = set()
            # Candidatos por tokens compartilhados (cap por doc)
            for tok in set(work.at[i, "_tokens"]):
                for j in token_index.get(tok, []):
                    if j == i or j in seen:
                        continue
                    seen.add(j)
                    # Cap por doc
                    if len(seen) > candidate_cap:
                        break

                    # Pré-filtro muito barato
                    if QRatio(work.at[i, "_norm"], work.at[j, "_norm"]) < pre_cutoff - 5:
                        continue

                    score, details = combined_score_fast(work.loc[i], work.loc[j])
                    if score >= threshold and details["contain"] >= min_containment_pct:
                        dsu.union(pos_map[i], pos_map[j])

            # (não precisa checar retorno do break cap; cap atua por doc)

        # Extrai componentes
        comp_map: Dict[int, List[int]] = {}
        for idx_local, i in enumerate(idxs):
            root = dsu.find(idx_local)
            comp_map.setdefault(root, []).append(i)

        for comp in comp_map.values():
            if len(comp) <= 1:
                continue
            ids = [str(work.at[i, "activity_id"]) for i in comp]
            groups.append(GroupIds(ids=ids, bucket=bkey))

        progress.progress(bi / len(bkeys))

    return groups

# ======== UI ========

st.title("🔁 Revisão de Duplicidades")

with st.sidebar:
    st.header("Parâmetros")
    dias_hist = st.slider("Dias de histórico (banco)", 1, 90, 15, 1)
    min_sim = st.slider("Similaridade mínima (%)", 70, 100, 88, 1)
    min_containment = st.slider("Containment mínimo (%)", 40, 100, 70, 1)
    pre_delta = st.slider("Pré-corte (Δ pontos)", 0, 15, 3, 1, help="Relaxamento no pré-filtro rápido (QRatio)")
    use_cnj = st.checkbox("Usar CNJ no bloqueio", value=True)
    candidate_cap = st.slider("Máx. candidatos por doc", 50, 500, 200, 50)
    dry_run = st.toggle("Simular (dry-run)", value=False, help="Não chama a API real; útil para testes rápidos")
    st.session_state[SK.DRY_RUN] = dry_run

    st.divider()
    if st.button("🔍 Analisar", use_container_width=True):
        st.session_state[SK.PARAMS] = dict(
            dias_hist=dias_hist, min_sim=min_sim, min_containment=min_containment,
            pre_delta=pre_delta, use_cnj=use_cnj, candidate_cap=candidate_cap
        )
        st.session_state["submitted"] = True
    else:
        st.session_state.setdefault("submitted", False)

# Carrega lista leve
df_min = carregar_lista_atividades(engine(), st.session_state[SK.PARAMS].get("dias_hist", dias_hist))

# Escolha simples de pasta/status para reduzir volume na interface (sem recomputar grupos desnecessariamente)
colf1, colf2, colf3 = st.columns([1,1,1])
with colf1:
    pastas = ["(Todas)"] + sorted(df_min["activity_folder"].dropna().unique().tolist())
    pasta_sel = st.selectbox("Pasta", pastas, index=0)
with colf2:
    status_opts = ["(Todos)", "Aberta", "Fechada"]
    status_sel = st.selectbox("Status", status_opts, index=0)
with colf3:
    itens_por_pagina = st.slider("Itens por página (grupos)", 3, 20, 7, 1)

mask = pd.Series(True, index=df_min.index)
if pasta_sel != "(Todas)":
    mask &= df_min["activity_folder"].fillna("") == pasta_sel
if status_sel != "(Todos)":
    mask &= df_min["activity_status"].fillna("") == status_sel
df_filtered = df_min[mask].copy()

# ======== Rodar análise (cache por assinatura dos parâmetros + ids) ========

def core_signature(df_ids: Iterable[str], params: Dict[str, Any]) -> str:
    pay = {
        "ids": tuple(sorted(map(str, df_ids))),
        "min_sim": params["min_sim"],
        "min_cont": params["min_containment"],
        "pre_delta": params["pre_delta"],
        "use_cnj": params["use_cnj"],
        "candidate_cap": params["candidate_cap"],
    }
    return hashlib.md5(json.dumps(pay, sort_keys=True).encode()).hexdigest()

params = st.session_state[SK.PARAMS] or dict(
    dias_hist=dias_hist, min_sim=min_sim, min_containment=min_containment,
    pre_delta=pre_delta, use_cnj=use_cnj, candidate_cap=candidate_cap,
)

if st.session_state.get("submitted", False):
    # Busca textos apenas para os ids em df_filtered (melhor: sob demanda — mas aqui faremos 1x para análise)
    textos_map = carregar_textos_por_id(engine(), tuple(df_filtered["activity_id"].astype(str)))
    sig = core_signature(df_filtered["activity_id"], params)
    if st.session_state[SK.ANALYSIS_RESULTS].get("sig") != sig:
        with st.spinner("Gerando grupos..."):
            groups = criar_grupos_ids(
                df_filtered,
                textos_map,
                min_sim_pct=float(params["min_sim"]),
                min_containment_pct=float(params["min_containment"]),
                pre_delta=float(params["pre_delta"]),
                use_cnj=bool(params["use_cnj"]),
                candidate_cap=int(params["candidate_cap"]),
            )
        st.session_state[SK.ANALYSIS_RESULTS] = {"sig": sig, "groups": groups}
        st.session_state[SK.GROUP_STATES] = {}  # limpa estados antigos (grupos mudaram)

groups: List[GroupIds] = st.session_state[SK.ANALYSIS_RESULTS].get("groups", [])

st.subheader(f"Grupos encontrados: {len(groups)}")

# Ordena por potencial de cancelamento (maiores primeiro)
def potential_cancel_count(g: GroupIds) -> int:
    return max(0, len(g.ids) - 1)
groups_sorted = sorted(groups, key=potential_cancel_count, reverse=True)

# Paginação
total_pages = max(1, (len(groups_sorted) + itens_por_pagina - 1) // itens_por_pagina)
page = st.number_input("Página", 1, total_pages, 1, 1)
start = (page - 1) * itens_por_pagina
end = start + itens_por_pagina
page_groups = groups_sorted[start:end]

# Carrega textos on-demand somente para grupos exibidos
ids_needed: Set[str] = set()
for g in page_groups:
    ids_needed.update(g.ids)
textos_page = carregar_textos_por_id(engine(), tuple(ids_needed))

# Prepara dataframe auxiliar com norm/meta/tokens para comparação e diff
work_rows: Dict[str, Dict[str, Any]] = {}
for aid in ids_needed:
    txt = textos_page.get(aid, "")
    norm = normalize_text(txt)
    toks = norm.split()
    work_rows[aid] = {
        "activity_id": aid,
        "_norm": norm,
        "_tokens": toks[:300],
        "_set": set(toks[:300]),
        "_meta": extract_meta(norm),
        "Texto": txt,
        # campos resolvidos do df_min (folder, status, date, user)
        "activity_folder": df_min.loc[df_min["activity_id"] == aid, "activity_folder"].values[0]
            if (df_min["activity_id"] == aid).any() else "",
        "activity_status": df_min.loc[df_min["activity_id"] == aid, "activity_status"].values[0]
            if (df_min["activity_id"] == aid).any() else "",
        "activity_date": df_min.loc[df_min["activity_id"] == aid, "activity_date"].values[0]
            if (df_min["activity_id"] == aid).any() else "",
        "user_profile_name": df_min.loc[df_min["activity_id"] == aid, "user_profile_name"].values[0]
            if (df_min["activity_id"] == aid).any() else "",
    }

def render_diff(a_text: str, b_text: str, limit_sentences: int = 80) -> Tuple[str, str]:
    # Diff simplificado: destaca diferenças com marcadores; aqui usamos abordagem simples
    # Para produção, você pode integrar uma lib de diff com HTML seguro.
    def first_n_sentences(t: str, n: int) -> str:
        parts = re.split(r"([.!?\n]+)", t)
        return "".join(parts[: 2 * n])
    a = first_n_sentences(a_text, limit_sentences)
    b = first_n_sentences(b_text, limit_sentences)
    # Destaque básico (não HTML pesado) — mantemos performance
    return a, b

# ======== Renderização de grupos ========

def ensure_group_state(ids: List[str]) -> Dict[str, Any]:
    gkey = group_key_from_ids(ids)
    state = st.session_state[SK.GROUP_STATES].setdefault(
        gkey, {"principal_id": ids[0], "cancelados": set(), "open_compare": None}
    )
    # garante consistência
    state["cancelados"] = {i for i in state["cancelados"] if i in ids}
    if state.get("principal_id") not in ids:
        state["principal_id"] = ids[0]
    return state

client = api_client()

for gi, g in enumerate(page_groups, start=start + 1):
    st.markdown(f"### Grupo {gi} — {len(g.ids)} itens  \n*Bucket*: `{g.bucket}`")
    state = ensure_group_state(g.ids)

    # Cabeçalho com ações rápidas
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.write("Selecione o **principal** e marque duplicatas para cancelar.")
    with c2:
        if st.button("✅ Tornar principal e cancelar demais", key=f"one_shot_{gi}"):
            state["cancelados"] = set(i for i in g.ids if i != state["principal_id"])
            st.rerun()
    with c3:
        if st.button("🗑️ Marcar todos (forte)", key=f"mark_all_{gi}"):
            principal = work_rows[state["principal_id"]]
            limiar = float(params["min_sim"]) + 5.0
            for rid in g.ids:
                if rid == state["principal_id"]:
                    continue
                score, det = combined_score_fast(principal, work_rows[rid])
                if score >= limiar and det["contain"] >= float(params["min_containment"]):
                    state["cancelados"].add(rid)
            st.rerun()
    with c4:
        if st.button("❌ Limpar marcações", key=f"clear_{gi}"):
            state["cancelados"].clear()
            st.rerun()

    st.divider()

    # Tabela simples (cartões compactos)
    for rid in g.ids:
        row = work_rows[rid]
        cols = st.columns([0.5, 3, 3, 1.2, 1.2])
        with cols[0]:
            make_principal = st.radio(
                "Principal", options=[False, True],
                index=1 if rid == state["principal_id"] else 0,
                key=f"principal_{gi}_{rid}", label_visibility="collapsed"
            )
            if make_principal and rid != state["principal_id"]:
                state["principal_id"] = rid
        with cols[1]:
            st.caption(f"**ID:** {rid} | **Pasta:** {row['activity_folder']} | **Usuário:** {row['user_profile_name']}")
            st.write(row["Texto"][:500] + ("..." if len(row["Texto"]) > 500 else ""))
        with cols[2]:
            # Comparação com o principal (lazy)
            if rid != state["principal_id"]:
                sc, details = combined_score_fast(work_rows[state["principal_id"]], row)
                st.metric("Similaridade", f"{sc:.1f}%")
                st.caption(f"contain: {details['contain']:.1f}% | set: {details['set']:.1f}% | sort: {details['sort']:.1f}%")
                if st.button("🔎 Comparar", key=f"cmp_{gi}_{rid}"):
                    state["open_compare"] = rid
                if state.get("open_compare") == rid:
                    a, b = render_diff(work_rows[state["principal_id"]]["Texto"], row["Texto"], 80)
                    st.text_area("Principal (trecho)", a, height=180)
                    st.text_area("Comparado (trecho)", b, height=180)
                    if st.button("Mostrar diff completo", key=f"cmp_full_{gi}_{rid}"):
                        a, b = render_diff(work_rows[state["principal_id"]]["Texto"], row["Texto"], 10**9)
                        st.text_area("Principal (completo)", a, height=240)
                        st.text_area("Comparado (completo)", b, height=240)
            else:
                st.info("👑 Publicação principal")
        with cols[3]:
            mark_cancel = st.checkbox("Cancelar", value=(rid in state["cancelados"]), key=f"cancel_{gi}_{rid}")
            if mark_cancel:
                state["cancelados"].add(rid)
            else:
                state["cancelados"].discard(rid)
        with cols[4]:
            st.caption(f"Status: {row['activity_status']}")
            st.caption(f"Data: {row['activity_date']}")

    st.markdown("---")

    # Ações para o grupo
    gcol1, gcol2, gcol3 = st.columns([1, 1, 2])
    with gcol1:
        if st.button("💾 Definir Principal", key=f"setp_{gi}"):
            res = client.set_principal(state["principal_id"], group_key_from_ids(g.ids), user=st.secrets.get("user", "ui"))
            if res.get("ok"):
                st.success("Principal definido com sucesso.")
            else:
                st.error(f"Falha ao definir principal: {res.get('error') or res}")
    with gcol2:
        to_cancel = [(rid, "Duplicata confirmada") for rid in state["cancelados"]]
        if st.button(f"🚮 Cancelar selecionadas ({len(to_cancel)})", key=f"bulk_{gi}") and to_cancel:
            prog = st.progress(0.0, text="Cancelando...")
            total_local = len(to_cancel)
            def upd(i, tot):
                prog.progress(i / tot, text=f"Cancelando... ({i}/{tot})")
            cps = float(st.secrets.get("api_client", {}).get("calls_per_second", 4.0))
            max_workers = max(2, int(cps * 2))
            results = client.process_cancellations_concurrently(
                to_cancel, requested_by=st.secrets.get("user", "ui"),
                update_progress=upd, max_workers=max_workers
            )
            ok = sum(1 for r in results if r.get("ok"))
            st.success(f"Concluído: {ok}/{len(results)} canceladas.")
            # opcional: marcar no UI como removidas
            # state["cancelados"].clear()

# ======== Exportação leve ========

def export_groups_csv(groups: List[GroupIds]) -> bytes:
    rows = []
    for i, g in enumerate(groups, 1):
        for aid in g.ids:
            rows.append({
                "group_index": i,
                "group_size": len(g.ids),
                "activity_id": aid,
                "bucket": g.bucket,
            })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Exportar grupos (CSV leve)",
    data=export_groups_csv(groups_sorted),
    file_name="grupos_duplicidade.csv",
    mime="text/csv",
    use_container_width=True
)

st.caption("Dica: para produção, ajuste os endpoints do cliente API em `api_functions_retry_opt.py`.")
