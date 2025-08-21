# -*- coding: utf-8 -*-
"""
Verificador de Duplicidade — Versão Otimizada e Corrigida
=========================================================

Este aplicativo combina as funcionalidades avançadas da versão 'appdelta'
com as otimizações de performance e a lógica de negócio corrigida para
a seleção do item principal, evitando que atividades canceladas sejam
sugeridas como principais.

Principais Correções e Melhorias:
- Lógica do Principal Corrigida: A função `get_best_principal_id` agora
  ignora atividades com status 'Cancelado' ao eleger o principal.
- Otimização do Cache: O cálculo pesado de similaridade é executado
  apenas quando os dados ou os parâmetros principais mudam.
- Gerenciamento de Estado: O 'session_state' que armazena o estado dos grupos
  é limpo quando os dados são recalculados, evitando acúmulo de memória.
- Reintrodução de Funcionalidades: Os links para ZFlow v1 e v2 foram adicionados
  de volta à interface de cada atividade.
- Melhoria de UX: Um spinner é exibido durante o cálculo de similaridade.
"""
from __future__ import annotations

import os
import re
import html
import logging
import time
import math
from datetime import datetime, timedelta, date
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from zoneinfo import ZoneInfo
from unidecode import unidecode
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher

# Importa o cliente de API.
try:
    from api_functions_retry import HttpClientRetry
except ImportError:
    st.error("Erro: O arquivo 'api_functions_retry.py' não foi encontrado.")
    HttpClientRetry = None

# Opcional para o gráfico de calibração
try:
    import altair as alt
except ImportError:
    alt = None

# Importações do Firebase para auditoria
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    st.warning("Aviso: A biblioteca 'firebase-admin' não foi encontrada. O log de auditoria será desativado.")
    firebase_admin = None


# =============================================================================
# CONFIGURAÇÃO GERAL E CONSTANTES
# =============================================================================
APP_TITLE = "Verificador de Duplicidade Avançado"
TZ_SP = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# Chaves para o session_state do Streamlit
SUFFIX = "_v6_final_corrigido"
class SK:
    USERNAME = f"username_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"
    CFG = f"cfg_{SUFFIX}"
    SHOW_CANCEL_CONFIRM = f"show_cancel_confirm_{SUFFIX}"
    IGNORED_GROUPS = f"ignored_groups_{SUFFIX}"

DEFAULTS = {
    "itens_por_pagina": 10,
    "dias_filtro_inicio": 7,
    "dias_filtro_fim": 14,
    "min_sim_global": 90,
    "min_containment": 55,
    "pre_cutoff_delta": 10,
    "diff_hard_limit": 12000,
}

# Configuração da página e estilos CSS
st.set_page_config(layout="wide", page_title=APP_TITLE)
st.markdown("""
<style>
    pre.highlighted-text {
        white-space: pre-wrap; word-wrap: break-word; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; height: 360px; overflow-y: auto;
    }
    .diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
    .diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }
    .card { border-left: 5px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #fff; }
    .card-cancelado { background-color: #f5f5f5; border-left: 5px solid #e0e0e0; }
    .card-principal { border-left: 5px solid #4CAF50; }
    .similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 600; display: inline-block; margin-bottom: 6px; }
    .badge-green { background:#C8E6C9; } .badge-yellow { background:#FFF9C4; } .badge-red { background:#FFCDD2; }
    .meta-chip { background:#E0F7FA; padding:2px 6px; margin-right:6px; border-radius:8px; display:inline-block; font-size:0.85em; }
    .small-muted { color:#777; font-size:0.85em; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INICIALIZAÇÃO DE SERVIÇOS
# =============================================================================

@st.cache_resource
def db_engine_mysql() -> Optional[Engine]:
    cfg = st.secrets.get("database", {})
    db_params = {k: cfg.get(k) for k in ["host", "user", "password", "name"]}
    if not all(db_params.values()):
        st.error("Credenciais do banco de dados (MySQL) ausentes.")
        st.stop()
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['name']}",
            pool_pre_ping=True, pool_recycle=3600
        )
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error(f"Erro ao conectar no banco (MySQL): {e}"); st.stop()

@st.cache_resource
def api_client() -> Optional[HttpClientRetry]:
    if HttpClientRetry is None: return None
    api_cfg = st.secrets.get("api", {})
    client_cfg = st.secrets.get("api_client", {})
    api_params = {k: api_cfg.get(k) for k in ["url_api", "entity_id", "token"]}
    if not all(api_params.values()):
        st.warning("Configuração da API ausente. Cancelamento desativado.")
        return None
    return HttpClientRetry(
        base_url=api_params["url_api"], entity_id=api_params["entity_id"], token=api_params["token"],
        calls_per_second=float(client_cfg.get("calls_per_second", 3.0)), max_attempts=int(client_cfg.get("max_attempts", 3)),
        timeout=int(client_cfg.get("timeout", 15)), dry_run=bool(client_cfg.get("dry_run", False))
    )

@st.cache_resource
def init_firebase():
    if not firebase_admin: return None
    try:
        if not firebase_admin._apps:
            creds_config = st.secrets.get("firebase_credentials")
            if not creds_config: st.warning("Credenciais do Firebase não encontradas."); return None
            creds_dict = dict(creds_config)
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        st.sidebar.success("Auditoria (Firebase) conectada. ✅")
        return db
    except Exception as e:
        st.sidebar.error(f"Falha ao conectar no Firebase: {e}."); return None

def log_action_to_firestore(db, user: str, action: str, details: Dict):
    if db is None: return
    try:
        doc_ref = db.collection("duplicidade_actions").document()
        log_entry = {"ts": firestore.SERVER_TIMESTAMP, "user": user, "action": action, "details": details}
        doc_ref.set(log_entry)
    except Exception as e:
        logging.error(f"Erro ao registrar ação no Firestore: {e}")
        st.toast(f"⚠️ Erro ao salvar log de auditoria: {e}", icon="🔥")


# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data(ttl=1800, hash_funcs={Engine: lambda _: None})
def carregar_dados_mysql(_eng: Engine, dias_historico: int) -> pd.DataFrame:
    limite = date.today() - timedelta(days=dias_historico)
    query = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND (activity_status='Aberta' OR DATE(activity_date) >= :limite)
    """)
    try:
        with _eng.connect() as conn:
            df = pd.read_sql(query, conn, params={"limite": limite})
        if df.empty: return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        return df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error(f"Erro ao carregar dados do banco: {e}"); return pd.DataFrame()

# =============================================================================
# LÓGICA DE SIMILARIDADE E FUNÇÕES AUXILIARES
# =============================================================================
CNJ_RE = re.compile(r"(?:\b|^)(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})(?:\b|$)")
URL_RE = re.compile(r"https?://\S+")
DATENUM_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
NUM_RE = re.compile(r"\b\d+\b")
STOPWORDS_BASE = set("de da do das dos e em a o os as na no para por com que ao aos às à um uma umas uns tipo titulo inteiro teor publicado publicacao disponibilizacao orgao vara tribunal processo recurso intimacao notificacao justica nacional diario djen poder judiciario trabalho".split())

def get_zflow_links(activity_id: str | int) -> dict:
    """Gera os links para as plataformas ZFlow v1 e v2."""
    return {
        "v1": f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}",
        "v2": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    }

def extract_meta(text: str) -> Dict[str, str]:
    t = text or ""; meta = {}
    cnj_match = CNJ_RE.search(t)
    cnj = cnj_match.group(1) if cnj_match else None
    if not cnj:
        proc_match = re.search(r"PROCESSO:\s*([0-9\-.]+)", t, re.IGNORECASE)
        if proc_match: cnj = proc_match.group(1)
    meta["processo"] = cnj or ""
    patterns = {
        "orgao": r"\bORGAO:\s*([^-\n\r]+)", "vara": r"\bVARA\s+DO\s+TRABALHO\s+[^-\n\r]+",
        "tipo_doc": r"\bTIPO\s+DE\s+DOCUMENTO:\s*([^-]+)", "tipo_com": r"\bTIPO\s+DE\s+COMUNICACAO:\s*([^-]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, t, re.IGNORECASE)
        if match: meta[key] = match.group(1).strip() if key != "vara" else match.group(0).strip()
    return meta

def normalize_for_match(text: str, stopwords_extra: List[str]) -> str:
    if not isinstance(text, str): return ""
    t = URL_RE.sub(" url ", text); t = CNJ_RE.sub(" numproc ", t); t = DATENUM_RE.sub(" data ", t)
    t = NUM_RE.sub(" # ", t); t = unidecode(t.lower()); t = re.sub(r"[^\w\s]", " ", t)
    all_stopwords = STOPWORDS_BASE.union(stopwords_extra)
    return " ".join([w for w in t.split() if w not in all_stopwords])

def token_containment(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    small, big = (a_tokens, set(b_tokens)) if len(a_tokens) <= len(b_tokens) else (b_tokens, set(a_tokens))
    return 100.0 * (sum(1 for token in small if token in big) / len(small))

def length_penalty(len_a: int, len_b: int) -> float:
    if len_a == 0 or len_b == 0: return 0.9
    return max(0.9, 1.0 - (abs(len_a - len_b) / max(len_a, len_b)) * 0.1)

def fields_bonus(meta_a: Dict[str,str], meta_b: Dict[str,str]) -> int:
    bonus = 0
    if meta_a.get("processo") and meta_a.get("processo") == meta_b.get("processo"): bonus += 6
    if meta_a.get("orgao") and meta_a.get("orgao") == meta_b.get("orgao"): bonus += 3
    if meta_a.get("tipo_doc") and meta_a.get("tipo_doc") == meta_b.get("tipo_doc"): bonus += 3
    if meta_a.get("tipo_com") and meta_a.get("tipo_com") == meta_b.get("tipo_com"): bonus += 2
    return bonus

def combined_score(a_norm: str, b_norm: str, meta_a: Dict[str,str], meta_b: Dict[str,str]) -> Tuple[float, Dict[str,float]]:
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm); sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    contain = token_containment(a_norm.split(), b_norm.split()); lp = length_penalty(len(a_norm), len(b_norm))
    bonus = fields_bonus(meta_a, meta_b)
    base_score = 0.6 * set_ratio + 0.2 * sort_ratio + 0.2 * contain
    final_score = max(0.0, min(100.0, base_score * lp + bonus))
    details = {"set": set_ratio, "sort": sort_ratio, "contain": contain, "len_pen": lp, "bonus": bonus, "base": base_score}
    return final_score, details

# =============================================================================
# LÓGICA DE AGRUPAMENTO
# =============================================================================

def build_buckets(df: pd.DataFrame, use_cnj: bool) -> Dict[str, List[int]]:
    buckets = defaultdict(list)
    for i, row in df.iterrows():
        folder = str(row.get("activity_folder") or "SEM_PASTA"); cnj = row.get("_meta", {}).get("processo", "")
        key = f"folder::{folder}"
        if use_cnj: key = f"{key}::cnj::{cnj or 'SEM_CNJ'}"
        buckets[key].append(i)
    return buckets

@st.cache_data(ttl=3600)
def criar_grupos_de_duplicatas(_df: pd.DataFrame, params: Dict) -> List[List[Dict]]:
    if _df.empty: return []

    work_df = _df.copy()
    stopwords_extra = st.secrets.get("similarity", {}).get("stopwords_extra", [])
    work_df["_meta"] = work_df["Texto"].apply(extract_meta)
    work_df["_norm"] = work_df["Texto"].apply(lambda t: normalize_for_match(t, stopwords_extra))

    buckets = build_buckets(work_df, params['use_cnj'])
    cutoffs_map = st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {})

    groups = []; memo_score: Dict[Tuple[int, int], Tuple[float, Dict]] = {}
    
    for bkey, idxs in buckets.items():
        if len(idxs) < 2: continue
        bucket_df = work_df.loc[idxs].reset_index().rename(columns={"index": "orig_idx"})
        texts = bucket_df["_norm"].tolist()
        folder_name = bkey.split("::")[1] if bkey.startswith("folder::") else None
        min_sim_bucket = float(cutoffs_map.get(folder_name, params['min_sim']))
        pre_cutoff = max(0, int(min_sim_bucket * 100) - params['pre_delta'])
        
        prelim_matrix = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=pre_cutoff)
        
        n = len(bucket_df); visited = set()
        
        def are_connected(i, j) -> bool:
            key = tuple(sorted((i, j)))
            if key in memo_score: score, details = memo_score[key]
            else:
                score, details = combined_score(bucket_df.loc[i, "_norm"], bucket_df.loc[j, "_norm"],
                                                bucket_df.loc[i, "_meta"], bucket_df.loc[j, "_meta"])
                memo_score[key] = (score, details)
            return details["contain"] >= params['min_containment'] and score >= (min_sim_bucket * 100.0)

        for i in range(n):
            if i in visited: continue
            component = {i}; queue = deque([i]); visited.add(i)
            while queue:
                current_node = queue.popleft()
                for neighbor in range(n):
                    if neighbor not in visited and prelim_matrix[current_node][neighbor] >= pre_cutoff and are_connected(current_node, neighbor):
                        visited.add(neighbor); component.add(neighbor); queue.append(neighbor)
            if len(component) > 1:
                sorted_idxs = sorted(list(component), key=lambda ix: bucket_df.loc[ix, "activity_date"], reverse=True)
                group_data = [work_df.loc[bucket_df.loc[ix, "orig_idx"]].to_dict() for ix in sorted_idxs]
                groups.append(group_data)

    return groups

# =============================================================================
# COMPONENTES DE UI E RENDERIZAÇÃO
# =============================================================================

def highlight_diffs_safe(text1: str, text2: str, hard_limit: int) -> Tuple[str,str]:
    t1, t2 = (text1 or ""), (text2 or "")
    if (len(t1) + len(t2)) > hard_limit:
        s1 = " ".join(re.split(r'([.!?\n]+)', t1)[:100]); s2 = " ".join(re.split(r'([.!?\n]+)', t2)[:100])
        h1, h2 = highlight_diffs(s1, s2)
        note = "<div class='small-muted'>⚠️ Diff parcial. Comparando apenas o início.</div>"
        return (note + h1, note + h2)
    return highlight_diffs(t1, t2)

def highlight_diffs(a: str, b: str) -> Tuple[str,str]:
    tokens1 = [tok for tok in re.split(r'(\W+)', a or "") if tok]; tokens2 = [tok for tok in re.split(r'(\W+)', b or "") if tok]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False); out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        s1, s2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal': out1.append(s1); out2.append(s2)
        elif tag == 'replace': out1.append(f"<span class='diff-del'>{s1}</span>"); out2.append(f"<span class='diff-ins'>{s2}</span>")
        elif tag == 'delete': out1.append(f"<span class='diff-del'>{s1}</span>")
        elif tag == 'insert': out2.append(f"<span class='diff-ins'>{s2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

def sidebar_controls(df_full: pd.DataFrame) -> Dict:
    st.sidebar.header("👤 Sessão"); username = st.session_state.get(SK.USERNAME, "Não logado")
    st.sidebar.success(f"Logado como: **{username}**")
    if st.sidebar.button("🔄 Forçar Atualização dos Dados"):
        carregar_dados_mysql.clear(); criar_grupos_de_duplicatas.clear(); st.rerun()

    st.sidebar.header("⚙️ Parâmetros de Similaridade")
    sim_cfg = st.secrets.get("similarity", {})
    min_sim = st.sidebar.slider("Similaridade Mínima Global (%)", 0, 100, int(sim_cfg.get("min_sim_global", DEFAULTS["min_sim_global"])), 1) / 100.0
    min_containment = st.sidebar.slider("Containment Mínimo (%)", 0, 100, int(sim_cfg.get("min_containment", DEFAULTS["min_containment"])), 1)
    pre_delta = st.sidebar.slider("Delta do Pré-corte", 0, 30, int(sim_cfg.get("pre_cutoff_delta", DEFAULTS["pre_cutoff_delta"])), 1)
    diff_limit = st.sidebar.number_input("Limite de Caracteres do Diff", min_value=5000, value=int(sim_cfg.get("diff_hard_limit", DEFAULTS["diff_hard_limit"])), step=1000)

    st.sidebar.header("👁️ Filtros de Exibição")
    dias_hist = st.sidebar.number_input("Dias de Histórico para Análise", min_value=7, max_value=365, value=10, step=1)
    
    pastas_opts = sorted(df_full["activity_folder"].dropna().unique()) if not df_full.empty else []
    status_opts = sorted(df_full["activity_status"].dropna().unique()) if not df_full.empty else []
    default_statuses = [s for s in status_opts if "Cancelad" not in s]
    pastas_sel = st.sidebar.multiselect("Filtrar por Pastas", pastas_opts)
    status_sel = st.sidebar.multiselect("Filtrar por Status", status_opts, default=default_statuses)
    
    only_groups_with_open = st.sidebar.toggle("Apenas grupos com atividades abertas", value=True)
    strict_only = st.sidebar.toggle("Modo Estrito", value=True)

    st.sidebar.header("🚀 Otimizações (Pré-índice)")
    use_cnj = st.sidebar.toggle("Restringir por Nº do Processo (CNJ)", value=True)
    
    st.sidebar.header("📡 API de Cancelamento")
    dry_run = st.sidebar.toggle("Modo Teste (Dry-run)", value=bool(st.secrets.get("api_client", {}).get("dry_run", False)))
    st.session_state[SK.CFG] = {"dry_run": dry_run}
    
    with st.sidebar.expander("Regras de Similaridade por Pasta"):
        st.json(st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {}))

    return dict(
        min_sim=min_sim, min_containment=min_containment, pre_delta=pre_delta,
        diff_limit=diff_limit, dias_hist=dias_hist,
        pastas=pastas_sel, status=status_sel, use_cnj=use_cnj,
        strict_only=strict_only, only_groups_with_open=only_groups_with_open
    )

def get_best_principal_id(group_rows: List[Dict], min_sim_pct: float, min_containment_pct: float) -> str:
    """
    Calcula qual item do grupo é o 'melhor principal' (medoid).
    LÓGICA ATUALIZADA: Prioriza atividades com status 'Fechada' ou 'Concluída',
    ignora 'Cancelada' e deixa 'Aberta' como última opção.
    """
    if not group_rows:
        return ""

    # Filtramos explicitamente as atividades canceladas.
    # Elas só serão consideradas se não houver nenhuma outra opção.
    active_candidates = [r for r in group_rows if "Cancelad" not in r.get("activity_status", "")]

    # Se todas as atividades no grupo já estiverem canceladas, retorne a mais recente.
    if not active_candidates:
        return group_rows[0]['activity_id']

    # Dentro dos candidatos ativos, separamos por prioridade.
    # Prioridade 1: Atividades já finalizadas (Fechada, Concluída, etc.)
    closed_candidates = [r for r in active_candidates if r.get("activity_status") != "Aberta"]
    # Prioridade 2: Atividades abertas (que queremos cancelar)
    open_candidates = [r for r in active_candidates if r.get("activity_status") == "Aberta"]

    # A lista de candidatos para o cálculo agora prioriza os fechados e ignora os cancelados.
    candidates = closed_candidates + open_candidates
    if not candidates: # Fallback caso algo dê errado
        return group_rows[0]['activity_id']

    best_id, max_avg_score = None, -1.0
    
    # O restante da lógica de cálculo de score permanece a mesma
    cache = {r['activity_id']: (normalize_for_match(r.get('Texto', ''), []), extract_meta(r.get('Texto', ''))) for r in group_rows}

    for candidate in candidates:
        candidate_id = candidate['activity_id']
        c_norm, c_meta = cache[candidate_id]
        scores = []
        for other in group_rows:
            if other['activity_id'] == candidate_id: continue
            o_norm, o_meta = cache[other['activity_id']]
            
            score, details = combined_score(c_norm, o_norm, c_meta, o_meta)
            if score >= min_sim_pct and details['contain'] >= min_containment_pct:
                scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if best_id is None or avg_score > max_avg_score:
            max_avg_score, best_id = avg_score, candidate_id
        
        # Otimização: se já temos um principal 'fechado', não precisamos continuar
        # avaliando os 'abertos', pois eles têm prioridade menor.
        if candidate in open_candidates and best_id in [c['activity_id'] for c in closed_candidates]:
            break
            
    return best_id or group_rows[0]['activity_id']

def render_group(group_rows: List[Dict], params: Dict, db_firestore):
    group_id = group_rows[0]["activity_id"]; user = st.session_state.get(SK.USERNAME, "desconhecido")
    state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {"principal_id": None, "open_compare": None, "cancelados": set()})

    if state["principal_id"] is None or not any(r["activity_id"] == state["principal_id"] for r in group_rows):
        state["principal_id"] = get_best_principal_id(group_rows, params['min_sim'] * 100, params['min_containment'])

    principal = next((r for r in group_rows if r["activity_id"] == state["principal_id"]), group_rows[0])
    p_norm = normalize_for_match(principal.get("Texto", ""), []); p_meta = extract_meta(principal.get("Texto", ""))

    visible_rows = [principal] if params['strict_only'] else group_rows
    if params['strict_only']:
        for row in group_rows:
            if row["activity_id"] == principal["activity_id"]: continue
            r_norm = normalize_for_match(row.get("Texto", ""), []); r_meta = extract_meta(row.get("Texto", ""))
            score, details = combined_score(p_norm, r_norm, p_meta, r_meta)
            if score >= (params['min_sim'] * 100) and details['contain'] >= params['min_containment']: visible_rows.append(row)

    open_count = sum(1 for r in group_rows if r.get('activity_status') == 'Aberta')
    expander_title = (f"Grupo: {len(group_rows)} itens ({open_count} Abertas) | Pasta: {group_rows[0].get('activity_folder', '')} | Principal Sugerido: #{state['principal_id']}")
    
    with st.expander(expander_title):
        cols = st.columns([1/3, 1/3, 1/3])
        if cols[0].button("⭐ Recalcular Principal", key=f"recalc_princ_{group_id}", use_container_width=True):
            best_id = get_best_principal_id(group_rows, params['min_sim'] * 100, params['min_containment'])
            log_action_to_firestore(db_firestore, user, "set_principal", {"group_id": group_id, "previous_principal_id": state["principal_id"], "new_principal_id": best_id, "method": "automatic_recalc"}); state["principal_id"] = best_id; state["open_compare"] = None; st.rerun()
        if cols[1].button("🗑️ Marcar Todos p/ Cancelar", key=f"cancel_all_{group_id}", use_container_width=True):
            ids_to_cancel = {r['activity_id'] for r in visible_rows if r['activity_id'] != state['principal_id']}
            state['cancelados'].update(ids_to_cancel); log_action_to_firestore(db_firestore, user, "mark_all_cancel", {"group_id": group_id, "principal_id": state["principal_id"], "cancelled_ids": list(ids_to_cancel)}); st.rerun()
        if cols[2].button("👍 Não é Duplicado", key=f"not_dup_{group_id}", use_container_width=True):
            st.session_state[SK.IGNORED_GROUPS].add(group_id); log_action_to_firestore(db_firestore, user, "mark_not_duplicate", {"group_id": group_id, "member_ids": [r['activity_id'] for r in group_rows]}); st.rerun()
        st.markdown("---")

        for row in visible_rows:
            rid = row["activity_id"]; is_principal = (rid == state["principal_id"]); is_comparing = (rid == state["open_compare"]); is_marked_for_cancel = (rid in state["cancelados"])
            card_class = "card card-principal" if is_principal else "card card-cancelado" if is_marked_for_cancel else "card"
            
            with st.container():
                st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
                c1, c2 = st.columns([0.7, 0.3])
                with c1:
                    dt = pd.to_datetime(row.get("activity_date")).tz_localize(TZ_UTC).tz_convert(TZ_SP) if pd.notna(row.get("activity_date")) else None
                    st.markdown(f"**ID:** `{rid}` {'⭐ **Principal**' if is_principal else ''} {'🗑️ **Marcado p/ Cancelar**' if is_marked_for_cancel else ''}")
                    st.caption(f"**Data:** {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} | **Status:** {row.get('activity_status','')} | **Usuário:** {row.get('user_profile_name','')}")
                    if not is_principal:
                        r_norm = normalize_for_match(row.get("Texto", ""), []); r_meta = extract_meta(row.get("Texto", ""))
                        score, details = combined_score(p_norm, r_norm, p_meta, r_meta); score_pct = params['min_sim'] * 100
                        badge_color = "badge-green" if score >= score_pct + 5 else "badge-yellow" if score >= score_pct else "badge-red"
                        tooltip = f"Set: {details['set']:.0f}% | Sort: {details['sort']:.0f}% | Contain: {details['contain']:.0f}% | Bônus: {details['bonus']}"
                        st.markdown(f"<span class='similarity-badge {badge_color}' title='{tooltip}'>Similaridade: {score:.0f}%</span>", unsafe_allow_html=True)
                    st.text_area("Texto", row.get("Texto", ""), height=100, disabled=True, key=f"txt_{rid}")
                    
                    links = get_zflow_links(rid)
                    b_cols = st.columns(2)
                    b_cols[0].link_button("Abrir no ZFlow v1", links["v1"], use_container_width=True)
                    b_cols[1].link_button("Abrir no ZFlow v2", links["v2"], use_container_width=True)

                with c2:
                    if not is_principal:
                        if st.button("⭐ Tornar Principal", key=f"mkp_{rid}", use_container_width=True):
                            log_action_to_firestore(db_firestore, user, "set_principal", {"group_id": group_id, "previous_principal_id": state["principal_id"], "new_principal_id": rid, "method": "manual"}); state["principal_id"] = rid; state["open_compare"] = None; st.rerun()
                        if st.button("⚖️ Comparar com Principal", key=f"cmp_{rid}", use_container_width=True):
                            state["open_compare"] = rid if not is_comparing else None; st.rerun()
                    if not is_principal and is_comparing:
                        st.markdown("---")
                        cancel_checked = st.checkbox("🗑️ Marcar para Cancelar", value=is_marked_for_cancel, key=f"cancel_{rid}")
                        if cancel_checked != is_marked_for_cancel:
                            action = "mark_cancel" if cancel_checked else "unmark_cancel"; log_action_to_firestore(db_firestore, user, action, {"group_id": group_id, "principal_id": state["principal_id"], "target_activity_id": rid})
                            if cancel_checked: state["cancelados"].add(rid)
                            else: state["cancelados"].discard(rid)
                            st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        if state["open_compare"]:
            comparado_row = next((r for r in group_rows if r["activity_id"] == state["open_compare"]), None)
            if comparado_row:
                st.markdown("---"); st.subheader("Comparação Detalhada (Diff)")
                st.markdown("""<div style='margin-bottom: 10px;'><strong>Legenda:</strong> <span style='background-color: #c8e6c9;'>Adicionado</span> <span style='background-color: #ffcdd2; margin-left: 10px;'>Removido</span></div>""", unsafe_allow_html=True)
                c1, c2 = st.columns(2); c1.markdown(f"**Principal: ID `{principal['activity_id']}`**"); c2.markdown(f"**Comparado: ID `{comparado_row['activity_id']}`**")
                hA, hB = highlight_diffs_safe(principal.get("Texto", ""), comparado_row.get("Texto", ""), params['diff_limit'])
                c1.markdown(hA, unsafe_allow_html=True); c2.markdown(hB, unsafe_allow_html=True)

# =============================================================================
# AÇÕES, CALIBRAÇÃO E HISTÓRICO
# =============================================================================
def export_groups_csv(groups: List[List[Dict]]) -> bytes:
    rows = [];
    for i, g in enumerate(groups):
        for r in g:
            rows.append({"group_index": i + 1, "group_size": len(g), "activity_id": r.get("activity_id"), "activity_folder": r.get("activity_folder"), "activity_date": r.get("activity_date"), "activity_status": r.get("activity_status"), "user_profile_name": r.get("user_profile_name"), "Texto": r.get("Texto","")})
    if not rows: return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def process_cancellations(to_cancel_with_context: List[Dict], user: str, db_firestore):
    client = api_client();
    if not client: st.error("Cliente de API não configurado."); return
    client.dry_run = st.session_state[SK.CFG].get("dry_run", True)
    st.info(f"Iniciando o cancelamento de {len(to_cancel_with_context)} atividades...")
    progress = st.progress(0); results = {"ok": 0, "err": 0}
    for i, item in enumerate(to_cancel_with_context):
        act_id = item["ID a Cancelar"]; principal_id = item["Duplicata do Principal"]
        try:
            response = client.activity_canceled(activity_id=act_id, user_name=user, principal_id=principal_id)
            if response and (response.get("ok") or response.get("success") or response.get("code") == '200'):
                results["ok"] += 1; log_action_to_firestore(db_firestore, user, "process_cancellation_success", item)
            else:
                results["err"] += 1; item["api_response"] = response; log_action_to_firestore(db_firestore, user, "process_cancellation_failure", item); st.warning(f"Falha ao cancelar {act_id}. Resposta: {response}")
        except Exception as e:
            results["err"] += 1; item["exception"] = str(e); log_action_to_firestore(db_firestore, user, "process_cancellation_exception", item); st.error(f"Erro de exceção ao cancelar {act_id}: {e}")
        progress.progress((i + 1) / len(to_cancel_with_context))
    st.success(f"Processamento concluído! Sucessos: {results['ok']}, Falhas: {results['err']}.")
    if client.dry_run: st.warning("Atenção: O modo Teste (Dry-run) está ativo.")
    for g_state in st.session_state[SK.GROUP_STATES].values(): g_state["cancelados"].clear()
    carregar_dados_mysql.clear(); criar_grupos_de_duplicatas.clear(); st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()

@st.dialog("Confirmação de Cancelamento")
def confirm_cancellation_dialog(groups: List[List[Dict]], user: str, db_firestore, params: Dict):
    to_cancel_with_context = []; score_cache = {}
    for g in groups:
        gid = g[0]["activity_id"]; state = st.session_state[SK.GROUP_STATES].get(gid, {}); principal_id = state.get("principal_id")
        if principal_id:
            principal_row = next((r for r in g if r['activity_id'] == principal_id), None)
            if not principal_row: continue
            p_norm = normalize_for_match(principal_row.get("Texto", ""), []); p_meta = extract_meta(principal_row.get("Texto", ""))
            for cancel_id in state.get("cancelados", set()):
                cancel_row = next((r for r in g if r['activity_id'] == cancel_id), None)
                if not cancel_row: continue
                if (principal_id, cancel_id) not in score_cache:
                    c_norm = normalize_for_match(cancel_row.get("Texto", ""), []); c_meta = extract_meta(cancel_row.get("Texto", ""))
                    score, _ = combined_score(p_norm, c_norm, p_meta, c_meta); score_cache[(principal_id, cancel_id)] = score
                to_cancel_with_context.append({"ID a Cancelar": cancel_id, "Duplicata do Principal": principal_id, "Pasta": cancel_row.get("activity_folder", "N/A"), "Similaridade (%)": f"{score_cache[(principal_id, cancel_id)]:.0f}"})
    if not to_cancel_with_context:
        st.info("Nenhuma atividade foi marcada para cancelamento.");
        if st.button("Fechar"): st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()
        return
    st.warning("Atenção: A ação a seguir é irreversível."); st.write(f"Você está prestes a cancelar **{len(to_cancel_with_context)}** atividades."); st.dataframe(to_cancel_with_context, use_container_width=True)
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirmar e Cancelar", type="primary", use_container_width=True): process_cancellations(to_cancel_with_context, user, db_firestore)
    if col2.button("Voltar", use_container_width=True): st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()

def render_calibration_tab(df: pd.DataFrame):
    st.subheader("📊 Calibração de Similaridade por Pasta")
    st.info("Esta ferramenta ajuda a encontrar o limiar de similaridade ideal para cada pasta.")
    if df.empty: st.warning("Não há dados para calibrar."); return
    pasta = st.selectbox("Selecione uma pasta:", sorted(df["activity_folder"].dropna().unique()))
    col1, col2 = st.columns(2); num_samples = col1.slider("Nº de Pares Aleatórios", 50, 2000, 500, 50); min_containment_filter = col2.slider("Filtro de Containment Mínimo (%)", 0, 100, 0, 1)
    if st.button("Analisar Pasta"):
        sample_df = df[df["activity_folder"] == pasta].copy()
        if len(sample_df) < 2: st.warning("A pasta tem menos de 2 atividades."); return
        stopwords_extra = st.secrets.get("similarity", {}).get("stopwords_extra", [])
        sample_df["_meta"] = sample_df["Texto"].apply(extract_meta); sample_df["_norm"] = sample_df["Texto"].apply(lambda t: normalize_for_match(t, stopwords_extra)); sample_df = sample_df.reset_index()
        n = len(sample_df); indices = np.arange(n); pairs = set(); rng = np.random.default_rng(seed=42)
        while len(pairs) < min(num_samples, (n * (n - 1)) // 2): pairs.add(tuple(sorted(rng.choice(indices, size=2, replace=False))))
        scores = []; progress = st.progress(0, text="Calculando scores...")
        for i, (idx1, idx2) in enumerate(pairs):
            row1, row2 = sample_df.iloc[idx1], sample_df.iloc[idx2]
            score, details = combined_score(row1["_norm"], row2["_norm"], row1["_meta"], row2["_meta"])
            if details["contain"] >= min_containment_filter: scores.append({"score": score, "containment": details["contain"]})
            progress.progress((i + 1) / len(pairs))
        progress.empty()
        if not scores: st.info("Nenhum par encontrado após filtro de containment."); return
        df_scores = pd.DataFrame(scores); st.write("Estatísticas Descritivas:"); st.dataframe(df_scores["score"].describe(percentiles=[.25, .5, .75, .9, .95, .99]))
        if alt: st.altair_chart(alt.Chart(df_scores).mark_bar().encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=50)), y=alt.Y("count()")).properties(title=f"Distribuição para: {pasta}", height=300), use_container_width=True)

@st.cache_data(ttl=600)
def get_firestore_history(_db, limit=100):
    if _db is None: return []
    try: return [doc.to_dict() for doc in _db.collection("duplicidade_actions").order_by("ts", direction=firestore.Query.DESCENDING).limit(limit).stream()]
    except Exception as e: st.error(f"Erro ao buscar histórico do Firestore: {e}"); return []

def render_history_tab(db_firestore):
    st.subheader("📜 Histórico de Ações (Auditoria)");
    if db_firestore is None: st.warning("A conexão com o Firebase (auditoria) não está ativa."); return
    if st.button("Atualizar Histórico"): get_firestore_history.clear()
    history = get_firestore_history(db_firestore)
    if not history: st.info("Nenhum registro de auditoria encontrado."); return
    for log in history:
        ts = log.get("ts"); ts_local = ts.astimezone(TZ_SP) if isinstance(ts, datetime) else None
        timestamp_str = ts_local.strftime('%d/%m/%Y %H:%M:%S') if ts_local else "N/A"
        user = log.get("user", "N/A"); action = log.get("action", "N/A").replace("_", " ").title()
        with st.expander(f"**{action}** por **{user}** em {timestamp_str}"): st.json(log.get("details", {}))

# =============================================================================
# FLUXO PRINCIPAL DO APLICATIVO
# =============================================================================
def main():
    st.title(APP_TITLE)
    for key in [SK.USERNAME, SK.GROUP_STATES, SK.CFG, SK.SHOW_CANCEL_CONFIRM, SK.IGNORED_GROUPS]:
        if key not in st.session_state:
            st.session_state[key] = set() if key == SK.IGNORED_GROUPS else {} if key == SK.GROUP_STATES else False

    if not st.session_state.get(SK.USERNAME):
        with st.sidebar.form("login_form"):
            username = st.text_input("Nome de Usuário"); password = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                if username and password and st.secrets.credentials.usernames.get(username) == password:
                    st.session_state[SK.USERNAME] = username; st.rerun()
                else: st.sidebar.error("Usuário ou senha inválidos.")
        st.info("👋 Bem-vindo! Por favor, faça o login na barra lateral."); st.stop()

    engine = db_engine_mysql(); db_firestore = init_firebase()
    df_full = carregar_dados_mysql(engine, 365)
    params = sidebar_controls(df_full)
    df_analysis = carregar_dados_mysql(engine, params["dias_hist"])
    
    if df_analysis.empty: st.warning("Nenhuma atividade encontrada para o período de análise."); st.stop()

    core_params = {k: params[k] for k in ['min_sim', 'min_containment', 'pre_delta', 'use_cnj']}
    
    with st.spinner("Analisando duplicatas... Este processo pode levar um momento."):
        all_groups = criar_grupos_de_duplicatas(df_analysis, core_params)

    current_first_group_id = all_groups[0][0]['activity_id'] if all_groups else None
    if 'last_group_id' not in st.session_state or st.session_state.last_group_id != current_first_group_id:
        st.session_state[SK.GROUP_STATES] = {}
        st.session_state.last_group_id = current_first_group_id

    filtered_groups = all_groups
    if params["pastas"]:
        filtered_groups = [g for g in filtered_groups if g[0].get("activity_folder") in params["pastas"]]
    if params["status"]:
        filtered_groups = [g for g in filtered_groups if any(r.get("activity_status") in params["status"] for r in g)]
    if params["only_groups_with_open"]:
        filtered_groups = [g for g in filtered_groups if any(r.get("activity_status") == "Aberta" for r in g)]
    
    filtered_groups = [g for g in filtered_groups if g[0]['activity_id'] not in st.session_state[SK.IGNORED_GROUPS]]

    tab1, tab2, tab3 = st.tabs(["🔎 Análise de Duplicidades", "📊 Calibração", "📜 Histórico de Ações"])

    with tab1:
        st.metric("Grupos de Duplicatas Encontrados (após filtros)", len(filtered_groups))
        page_size = st.number_input("Grupos por página", min_value=5, value=DEFAULTS["itens_por_pagina"], step=5)
        total_pages = max(1, math.ceil(len(filtered_groups) / page_size))
        page_num = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page_num - 1) * page_size; end_idx = start_idx + page_size
        st.caption(f"Exibindo grupos {start_idx + 1}–{min(end_idx, len(filtered_groups))} de {len(filtered_groups)}")

        for group in filtered_groups[start_idx:end_idx]:
            render_group(group, params, db_firestore)

        st.markdown("---"); st.header("⚡ Ações em Massa")
        col_a, col_b = st.columns(2)
        col_a.download_button("⬇️ Exportar Grupos para CSV", data=export_groups_csv(filtered_groups), file_name="relatorio_duplicatas.csv", mime="text/csv", use_container_width=True)
        if col_b.button("🚀 Processar Cancelamentos Marcados", type="primary", use_container_width=True):
            st.session_state[SK.SHOW_CANCEL_CONFIRM] = True
        
        if st.session_state.get(SK.SHOW_CANCEL_CONFIRM):
            confirm_cancellation_dialog(filtered_groups, st.session_state.get(SK.USERNAME), db_firestore, params)

    with tab2: render_calibration_tab(df_full)
    with tab3: render_history_tab(db_firestore)

if __name__ == "__main__":
    main()
