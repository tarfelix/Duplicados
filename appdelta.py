# -*- coding: utf-8 -*-
"""
Verificador de Duplicidade — Versão Híbrida Final
==================================================

Esta versão combina a precisão do algoritmo de agrupamento original com as
otimizações de performance e usabilidade desenvolvidas.

- **Precisão Restaurada:** O algoritmo de agrupamento principal foi revertido
  para a versão original (baseada em `process.cdist`) para garantir que
  nenhum duplicado seja perdido.
- **Performance Mantida:** A carga de dados "lazy" (sob demanda), a UI
  controlada por um botão de análise e a estabilidade da sessão foram mantidas.
- **Bugs Corrigidos:** O `ValueError` na renderização, o erro de SQL na
  carga de textos e a falha no processo de cancelamento foram corrigidos.
- **Funcionalidades Restauradas:** Os links para o ZFlow, a aba de Calibração,
  a aba de Histórico completa (com exportações) e a similaridade na
  confirmação de cancelamento foram reintroduzidos.
- **NOVO:**
    - **Lógica de Datas:** O app agora extrai e compara datas de publicação/
      disponibilização, penalizando scores de itens com datas muito
      diferentes e exibindo um alerta visual.
    - **Ocultação de Grupos Resolvidos:** Grupos são ocultados apenas se
      restar uma (ou nenhuma) atividade não cancelada.
"""
from __future__ import annotations

import os
import re
import html
import logging
import time
import math
import json
import hashlib
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

# Importa o cliente de API otimizado.
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
APP_TITLE = "Verificador de Duplicidade (Versão Híbrida Final)"
TZ_SP = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# Chaves para o session_state do Streamlit
SUFFIX = "_v20_resolved_logic_fix"
class SK:
    USERNAME = f"username_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"
    CFG = f"cfg_{SUFFIX}"
    SHOW_CANCEL_CONFIRM = f"show_cancel_confirm_{SUFFIX}"
    IGNORED_GROUPS = f"ignored_groups_{SUFFIX}"
    ANALYSIS_RESULTS = f"analysis_results_{SUFFIX}"

DEFAULTS = {
    "itens_por_pagina": 10,
    "min_sim_global": 95,
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
        font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; max-height: 360px; overflow-y: auto;
    }
    .diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
    .diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }
    .card { border-left: 5px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .card-cancelado { background-color: #FFEBEE; border-left: 5px solid #F44336; }
    .card-principal { background-color: #E8F5E9; border-left: 5px solid #4CAF50; }
    .similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 600; display: inline-block; margin-bottom: 6px; }
    .badge-green { background:#C8E6C9; } .badge-yellow { background:#FFF9C4; } .badge-red { background:#FFCDD2; }
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
        calls_per_second=float(client_cfg.get("calls_per_second", 4.0)), max_attempts=int(client_cfg.get("max_attempts", 3)),
        timeout=int(client_cfg.get("timeout", 20)), dry_run=bool(client_cfg.get("dry_run", False))
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
# CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS (OTIMIZADO)
# =============================================================================

@st.cache_data(ttl=1800)
def carregar_dados_minimos(_eng: Engine, dias_historico: int) -> pd.DataFrame:
    limite = datetime.now() - timedelta(days=dias_historico)
    query = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND (activity_status='Aberta' OR activity_date >= :limite)
    """)
    try:
        with _eng.connect() as conn:
            df = pd.read_sql(query, conn, params={"limite": limite})
        if df.empty: return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce").dt.tz_localize(TZ_UTC)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        return df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error(f"Erro ao carregar dados do banco: {e}"); return pd.DataFrame()

@st.cache_data(ttl=3600)
def carregar_textos_por_id(_eng: Engine, ids: Tuple[str, ...]) -> Dict[str, str]:
    if not ids: return {}
    params = {f"id_{i}": id_val for i, id_val in enumerate(ids)}
    param_names = [f":{key}" for key in params.keys()]
    query_string = f"SELECT activity_id, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_id IN ({', '.join(param_names)})"
    query = text(query_string)
    try:
        with _eng.connect() as conn:
            df_textos = pd.read_sql(query, conn, params=params)
        return pd.Series(df_textos.Texto.values, index=df_textos.activity_id.astype(str)).to_dict()
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error(f"Erro ao buscar textos: {e}"); return {}

@st.cache_data(ttl=30)
def verificar_status_atividades(_eng: Engine, ids: List[str]) -> Dict[str, str]:
    if not ids:
        return {}
    params = {f"id_{i}": id_val for i, id_val in enumerate(ids)}
    param_names = [f":{key}" for key in params.keys()]
    query_string = f"SELECT activity_id, activity_status FROM ViewGrdAtividadesTarcisio WHERE activity_id IN ({', '.join(param_names)})"
    query = text(query_string)
    try:
        with _eng.connect() as conn:
            df_status = pd.read_sql(query, conn, params=params)
        return pd.Series(df_status.activity_status.values, index=df_status.activity_id.astype(str)).to_dict()
    except exc.SQLAlchemyError as e:
        st.error(f"Erro ao verificar status no banco de dados: {e}")
        return {}

# =============================================================================
# LÓGICA DE SIMILARIDADE E FUNÇÕES AUXILIARES
# =============================================================================
CNJ_RE = re.compile(r"(?:\b|^)(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})(?:\b|$)")
STOPWORDS_BASE = set("de da do das dos e em a o os as na no para por com que ao aos às à um uma umas uns tipo titulo inteiro teor publicado publicacao disponibilizacao orgao vara tribunal processo recurso intimacao notificacao justica nacional diario djen poder judiciario trabalho".split())

DATE_PATTERNS = {
    "disponibilizacao": re.compile(r"Disponibiliza(?:ç|c)(?:ã|a)o:\s*.*?,\s*(\d{1,2})\s*DE\s*([A-ZÇÃ-ú]+)\s*DE\s*(\d{4})", re.IGNORECASE),
    "publicacao": re.compile(r"Publica(?:ç|c)(?:ã|a)o:\s*.*?,\s*(\d{1,2})\s*DE\s*([A-ZÇÃ-ú]+)\s*DE\s*(\d{4})", re.IGNORECASE)
}
MONTHS = {"JANEIRO": 1, "FEVEREIRO": 2, "MARCO": 3, "ABRIL": 4, "MAIO": 5, "JUNHO": 6, "JULHO": 7, "AGOSTO": 8, "SETEMBRO": 9, "OUTUBRO": 10, "NOVEMBRO": 11, "DEZEMBRO": 12}

def parse_date_from_text(text: str, pattern: re.Pattern) -> Optional[date]:
    match = pattern.search(text)
    if match:
        day, month_str, year = match.groups()
        month = MONTHS.get(unidecode(month_str.upper()))
        if month:
            try:
                return date(int(year), month, int(day))
            except ValueError:
                return None
    return None

def get_zflow_links(activity_id: str | int) -> dict:
    return {
        "v1": f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}",
        "v2": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    }

def extract_meta(text: str) -> Dict[str, any]:
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
    
    meta["data_disponibilizacao"] = parse_date_from_text(t, DATE_PATTERNS["disponibilizacao"])
    meta["data_publicacao"] = parse_date_from_text(t, DATE_PATTERNS["publicacao"])
    return meta

def normalize_for_match(text: str, stopwords_extra: List[str]) -> str:
    if not isinstance(text, str): return ""
    t = re.sub(r"https?://\S+", " url ", text)
    t = CNJ_RE.sub(" numproc ", t)
    t = re.sub(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", " data ", t)
    t = re.sub(r"\b\d+\b", " # ", t)
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    all_stopwords = STOPWORDS_BASE.union(stopwords_extra)
    return " ".join([w for w in t.split() if w not in all_stopwords])

def date_penalty(meta_a: Dict, meta_b: Dict) -> Tuple[float, bool]:
    date_a = meta_a.get("data_publicacao") or meta_a.get("data_disponibilizacao")
    date_b = meta_b.get("data_publicacao") or meta_b.get("data_disponibilizacao")
    
    if not date_a or not date_b:
        return 1.0, False # Sem penalidade se não encontrar as datas
    
    delta = abs((date_a - date_b).days)
    
    # Se a diferença for de 1 dia ou 3 (caso de fim de semana), é provável que seja disp/pub.
    if delta == 1 or (delta == 3 and date_a.weekday() == 4 and date_b.weekday() == 0) or (delta == 3 and date_b.weekday() == 4 and date_a.weekday() == 0):
        return 1.0, False
    
    # Se as datas são idênticas
    if delta == 0:
        return 1.0, False
        
    # Se a diferença for grande, aplica uma penalidade severa.
    return 0.7, True # 30% de penalidade e alerta de data suspeita

def combined_score(a_norm: str, b_norm: str, meta_a: Dict, meta_b: Dict) -> Tuple[float, Dict]:
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm)
    sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    a_tokens, b_tokens = a_norm.split(), b_norm.split()
    if not a_tokens or not b_tokens: return 0.0, {}
    small, big = (a_tokens, set(b_tokens)) if len(a_tokens) <= len(b_tokens) else (b_tokens, set(a_tokens))
    contain = 100.0 * (sum(1 for token in small if token in big) / len(small))
    len_a, len_b = len(a_norm), len(b_norm)
    lp = max(0.9, 1.0 - (abs(len_a - len_b) / max(len_a, len_b)) * 0.1) if len_a > 0 and len_b > 0 else 0.9
    bonus = 0
    if meta_a.get("processo") and meta_a.get("processo") == meta_b.get("processo"): bonus += 6
    if meta_a.get("orgao") and meta_a.get("orgao") == meta_b.get("orgao"): bonus += 3
    if meta_a.get("tipo_doc") and meta_a.get("tipo_doc") == meta_b.get("tipo_doc"): bonus += 3
    if meta_a.get("tipo_com") and meta_a.get("tipo_com") == meta_b.get("tipo_com"): bonus += 2
    
    date_pen, date_alert = date_penalty(meta_a, meta_b)

    base_score = 0.6 * set_ratio + 0.2 * sort_ratio + 0.2 * contain
    final_score = max(0.0, min(100.0, (base_score * lp + bonus) * date_pen))
    details = {"set": set_ratio, "sort": sort_ratio, "contain": contain, "len_pen": lp, "bonus": bonus, "base": base_score, "date_penalty": date_pen, "date_alert": date_alert}
    return final_score, details

# =============================================================================
# LÓGICA DE AGRUPAMENTO (ORIGINAL RESTAURADA + OTIMIZAÇÕES)
# =============================================================================

@st.cache_data(ttl=3600, max_entries=10)
def criar_grupos_de_duplicatas_hibrido(_df_min: pd.DataFrame, params: Dict, _textos: Dict[str, str]) -> List[List[Dict]]:
    if _df_min.empty: return []
    work_df = _df_min.copy()
    work_df["Texto"] = work_df["activity_id"].map(_textos)
    work_df = work_df.dropna(subset=["Texto"])
    stopwords_extra = st.secrets.get("similarity", {}).get("stopwords_extra", [])
    work_df["_meta"] = work_df["Texto"].apply(extract_meta)
    work_df["_norm"] = work_df["Texto"].apply(lambda t: normalize_for_match(t, stopwords_extra))
    buckets = defaultdict(list)
    for i, row in work_df.iterrows():
        folder = str(row.get("activity_folder") or "SEM_PASTA")
        cnj = row.get("_meta", {}).get("processo", "")
        key = f"folder::{folder}"
        if params['use_cnj']: key = f"{key}::cnj::{cnj or 'SEM_CNJ'}"
        buckets[key].append(i)
    cutoffs_map = st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {})
    groups = []
    memo_score: Dict[Tuple[int, int], Tuple[float, Dict]] = {}
    progress_bar = st.progress(0, text="Analisando buckets...")
    total_buckets = len(buckets)
    for i, (bkey, idxs) in enumerate(buckets.items()):
        progress_bar.progress((i + 1) / total_buckets, text=f"Analisando bucket {i+1}/{total_buckets}")
        if len(idxs) < 2: continue
        bucket_df = work_df.loc[idxs].reset_index().rename(columns={"index": "orig_idx"})
        texts = bucket_df["_norm"].tolist()
        folder_name = bkey.split("::")[1] if bkey.startswith("folder::") else None
        min_sim_bucket = float(cutoffs_map.get(folder_name, params['min_sim']))
        pre_cutoff = max(0, int(min_sim_bucket * 100) - params['pre_delta'])
        prelim_matrix = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=pre_cutoff)
        n = len(bucket_df)
        visited = set()
        def are_connected(i, j) -> bool:
            key = tuple(sorted((i, j)))
            if key in memo_score: score, details = memo_score[key]
            else:
                score, details = combined_score(bucket_df.loc[i, "_norm"], bucket_df.loc[j, "_norm"], bucket_df.loc[i, "_meta"], bucket_df.loc[j, "_meta"])
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
    progress_bar.empty()
    return groups

# =============================================================================
# COMPONENTES DE UI E RENDERIZAÇÃO
# =============================================================================

def generate_group_key(group_rows: List[Dict]) -> str:
    if not group_rows: return ""
    ids = sorted([r['activity_id'] for r in group_rows])
    return hashlib.md5(json.dumps(ids).encode()).hexdigest()

def highlight_diffs(a: str, b: str, hard_limit: int) -> Tuple[str,str]:
    t1, t2 = (a or ""), (b or "")
    if (len(t1) + len(t2)) > hard_limit:
        note = f"<div class='small-muted'>⚠️ Diff truncado em {hard_limit // 2} caracteres por texto.</div>"
        t1, t2 = t1[:hard_limit//2], t2[:hard_limit//2]
    else:
        note = ""
    tokens1 = [tok for tok in re.split(r'(\W+)', t1) if tok]; tokens2 = [tok for tok in re.split(r'(\W+)', t2) if tok]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False); out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        s1, s2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal': out1.append(s1); out2.append(s2)
        elif tag == 'replace': out1.append(f"<span class='diff-del'>{s1}</span>"); out2.append(f"<span class='diff-ins'>{s2}</span>")
        elif tag == 'delete': out1.append(f"<span class='diff-del'>{s1}</span>")
        elif tag == 'insert': out2.append(f"<span class='diff-ins'>{s2}</span>")
    return (f"{note}<pre class='highlighted-text'>{''.join(out1)}</pre>", f"{note}<pre class='highlighted-text'>{''.join(out2)}</pre>")

def sidebar_controls(df_full: pd.DataFrame) -> Dict:
    st.sidebar.header("👤 Sessão"); username = st.session_state.get(SK.USERNAME, "Não logado")
    st.sidebar.success(f"Logado como: **{username}**")
    if st.sidebar.button("🔄 Forçar Atualização dos Dados"):
        st.cache_data.clear(); st.rerun()
    params = {}
    with st.sidebar.form(key='params_form'):
        st.header("⚙️ Parâmetros de Análise")
        min_sim = st.slider("Similaridade Mínima Global (%)", 0, 100, DEFAULTS["min_sim_global"], 1) / 100.0
        min_containment = st.slider("Containment Mínimo (%)", 0, 100, DEFAULTS["min_containment"], 1)
        pre_delta = st.slider("Delta do Pré-corte", 0, 30, DEFAULTS["pre_cutoff_delta"], 1)
        use_cnj = st.toggle("Restringir por Nº do Processo (CNJ)", value=True)
        analysis_submitted = st.form_submit_button("🚀 Aplicar e Analisar", type="primary")
    params.update({"min_sim": min_sim, "min_containment": min_containment, "pre_delta": pre_delta, "use_cnj": use_cnj, "analysis_submitted": analysis_submitted})
    st.sidebar.header("👁️ Filtros de Exibição")
    dias_hist = st.sidebar.number_input("Dias de Histórico para Análise", min_value=7, max_value=365, value=14, step=1)
    pastas_opts = sorted(df_full["activity_folder"].dropna().unique()) if not df_full.empty else []
    status_opts = sorted(df_full["activity_status"].dropna().unique()) if not df_full.empty else []
    default_statuses = [s for s in status_opts if "Cancelad" not in s]
    pastas_sel = st.sidebar.multiselect("Filtrar por Pastas", pastas_opts)
    status_sel = st.sidebar.multiselect("Filtrar por Status", status_opts, default=default_statuses)
    only_groups_with_open = st.sidebar.toggle("Apenas grupos com atividades abertas", value=True)
    strict_only = st.sidebar.toggle("Modo Estrito", value=True)
    params.update({"dias_hist": dias_hist, "pastas": pastas_sel, "status": status_sel, "strict_only": strict_only, "only_groups_with_open": only_groups_with_open})
    st.sidebar.header("📡 Configurações Adicionais")
    diff_limit = st.sidebar.number_input("Limite de Caracteres do Diff", min_value=5000, value=DEFAULTS["diff_hard_limit"], step=1000)
    dry_run = st.sidebar.toggle("Modo Teste (Dry-run)", value=bool(st.secrets.get("api_client", {}).get("dry_run", False)))
    st.session_state[SK.CFG] = {"dry_run": dry_run}
    params["diff_limit"] = diff_limit
    return params

def get_best_principal_id(group_rows: List[Dict], min_sim_pct: float, min_containment_pct: float) -> str:
    if not group_rows: return ""
    active_candidates = [r for r in group_rows if "Cancelad" not in r.get("activity_status", "")]
    if not active_candidates: return group_rows[0]['activity_id']
    closed_candidates = [r for r in active_candidates if r.get("activity_status") != "Aberta"]
    open_candidates = [r for r in active_candidates if r.get("activity_status") == "Aberta"]
    candidates = closed_candidates + open_candidates
    if not candidates: return group_rows[0]['activity_id']
    best_id, max_avg_score = None, -1.0
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
        if candidate in open_candidates and best_id in [c['activity_id'] for c in closed_candidates]:
            break
    return best_id or group_rows[0]['activity_id']

def render_group(group_rows: List[Dict], params: Dict, db_firestore, engine: Engine):
    group_key = generate_group_key(group_rows)
    user = st.session_state.get(SK.USERNAME, "desconhecido")
    state = st.session_state[SK.GROUP_STATES].setdefault(group_key, {"principal_id": None, "open_compare": None, "cancelados": set()})
    if state["principal_id"] is None or not any(r["activity_id"] == state["principal_id"] for r in group_rows):
        state["principal_id"] = get_best_principal_id(group_rows, params['min_sim'] * 100, params['min_containment'])
    principal_row = next((r for r in group_rows if r["activity_id"] == state["principal_id"]), None)
    if principal_row is None:
        st.error(f"Erro: Atividade principal ID {state['principal_id']} não encontrada. Resetando para o mais recente.")
        state["principal_id"] = group_rows[0]['activity_id']
        principal_row = group_rows[0]
    if params['strict_only']:
        p_norm = normalize_for_match(principal_row.get("Texto", ""), [])
        p_meta = extract_meta(principal_row.get("Texto", ""))
        visible_rows = [principal_row]
        for row in group_rows:
            if row["activity_id"] == principal_row["activity_id"]: continue
            r_norm = normalize_for_match(row.get("Texto", ""), [])
            r_meta = extract_meta(row.get("Texto", ""))
            score, details = combined_score(p_norm, r_norm, p_meta, r_meta)
            if score >= (params['min_sim'] * 100) and details['contain'] >= params['min_containment']:
                visible_rows.append(row)
    else:
        visible_rows = group_rows
    display_rows = sorted(visible_rows, key=lambda r: r["activity_id"] != principal_row["activity_id"])
    open_count = sum(1 for r in display_rows if r.get('activity_status') == 'Aberta')
    expander_title = (f"Grupo: {len(display_rows)} itens ({open_count} Abertas) | Pasta: {principal_row.get('activity_folder', 'N/A')} | Principal: #{state['principal_id']}")
    with st.expander(expander_title):
        cols = st.columns(5)
        if cols[0].button("⭐ Recalcular Principal", key=f"recalc_princ_{group_key}", use_container_width=True):
            state["principal_id"] = get_best_principal_id(group_rows, params['min_sim'] * 100, params['min_containment']); st.rerun()
        if cols[1].button("🗑️ Marcar Todos p/ Cancelar", key=f"cancel_all_{group_key}", use_container_width=True):
            state['cancelados'].update({r['activity_id'] for r in display_rows if r['activity_id'] != state['principal_id']}); st.rerun()
        if cols[2].button("👍 Não é Duplicado", key=f"not_dup_{group_key}", use_container_width=True):
            st.session_state[SK.IGNORED_GROUPS].add(group_key); st.rerun()
        if cols[3].button("✅ Principal + Cancelar Resto", key=f"one_shot_{group_key}", use_container_width=True, type="primary"):
            state["cancelados"] = {r["activity_id"] for r in display_rows if r["activity_id"] != state["principal_id"]}; st.rerun()
        if cols[4].button("🔄 Verificar Status", key=f"verify_status_{group_key}", use_container_width=True):
            with st.spinner("Verificando status no banco de dados..."):
                ids_to_check = [r['activity_id'] for r in group_rows]
                status_map = verificar_status_atividades(engine, ids_to_check)
                if status_map:
                    status_df = pd.DataFrame.from_dict(status_map, orient='index', columns=['Status Atual'])
                    st.dataframe(status_df)
                else:
                    st.warning("Não foi possível obter o status das atividades.")
        st.markdown("---")
        for row in display_rows:
            rid = row["activity_id"]; is_principal = (rid == state["principal_id"]); is_comparing = (rid == state["open_compare"]); is_marked_for_cancel = (rid in state["cancelados"])
            card_class = "card card-principal" if is_principal else "card card-cancelado" if is_marked_for_cancel else "card"
            with st.container():
                st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
                c1, c2 = st.columns([0.7, 0.3])
                with c1:
                    dt = row["activity_date"].tz_convert(TZ_SP) if pd.notna(row["activity_date"]) else None
                    st.markdown(f"**ID:** `{rid}` {'⭐ **Principal**' if is_principal else ''}")
                    st.caption(f"**Data:** {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} | **Status:** {row.get('activity_status','')} | **Usuário:** {row.get('user_profile_name','')}")
                    if not is_principal:
                        score, details = combined_score(normalize_for_match(principal_row.get("Texto", ""), []), normalize_for_match(row.get("Texto", ""), []), extract_meta(principal_row.get("Texto", "")), extract_meta(row.get("Texto", "")))
                        badge_color = "badge-green" if score >= (params['min_sim'] * 100) + 5 else "badge-yellow" if score >= (params['min_sim'] * 100) else "badge-red"
                        tooltip = f"Set: {details['set']:.0f}% | Sort: {details['sort']:.0f}% | Contain: {details['contain']:.0f}% | Bônus: {details['bonus']}"
                        date_alert_icon = "⚠️" if details.get("date_alert") else ""
                        st.markdown(f"<span class='similarity-badge {badge_color}' title='{tooltip}'>Similaridade: {score:.0f}% {date_alert_icon}</span>", unsafe_allow_html=True)
                    st.text_area("Texto", row.get("Texto", ""), height=100, disabled=True, key=f"txt_{rid}")
                    links = get_zflow_links(rid)
                    b_cols = st.columns(2)
                    b_cols[0].link_button("Abrir no ZFlow v1", links["v1"], use_container_width=True)
                    b_cols[1].link_button("Abrir no ZFlow v2", links["v2"], use_container_width=True)
                with c2:
                    if not is_principal:
                        if st.button("⭐ Tornar Principal", key=f"mkp_{rid}", use_container_width=True):
                            state["principal_id"] = rid; state["open_compare"] = None; st.rerun()
                        if st.button("⚖️ Comparar", key=f"cmp_{rid}", use_container_width=True):
                            state["open_compare"] = rid if not is_comparing else None; st.rerun()
                    cancel_checked = st.checkbox("🗑️ Marcar para Cancelar", value=is_marked_for_cancel, key=f"cancel_{rid}")
                    if cancel_checked != is_marked_for_cancel:
                        if cancel_checked: state["cancelados"].add(rid)
                        else: state["cancelados"].discard(rid)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        if state["open_compare"]:
            comparado_row = next((r for r in group_rows if r["activity_id"] == state["open_compare"]), None)
            if comparado_row:
                st.markdown("---"); st.subheader("Comparação Detalhada (Diff)")
                c1, c2 = st.columns(2); c1.markdown(f"**Principal: ID `{principal_row['activity_id']}`**"); c2.markdown(f"**Comparado: ID `{comparado_row['activity_id']}`**")
                hA, hB = highlight_diffs(principal_row.get("Texto", ""), comparado_row.get("Texto", ""), params['diff_limit'])
                c1.markdown(hA, unsafe_allow_html=True); c2.markdown(hB, unsafe_allow_html=True)

# =============================================================================
# AÇÕES, CALIBRAÇÃO E HISTÓRICO
# =============================================================================
def process_cancellations(to_cancel_with_context: List[Dict], user: str, db_firestore, engine: Engine):
    client = api_client()
    if not client: st.error("Cliente de API não configurado."); return
    client.dry_run = st.session_state[SK.CFG].get("dry_run", True)
    st.info(f"Iniciando o cancelamento de {len(to_cancel_with_context)} atividades...")
    progress = st.progress(0); results = {"ok": 0, "err": 0}
    ids_to_check = []
    for i, item in enumerate(to_cancel_with_context):
        act_id = item["ID a Cancelar"]; principal_id = item["Duplicata do Principal"]
        ids_to_check.append(act_id)
        try:
            response = client.activity_canceled(activity_id=act_id, user_name=user, principal_id=principal_id)
            if response and (response.get("ok") or response.get("success")):
                results["ok"] += 1; log_action_to_firestore(db_firestore, user, "process_cancellation_success", item)
            else:
                results["err"] += 1; item["api_response"] = response; log_action_to_firestore(db_firestore, user, "process_cancellation_failure", item); st.warning(f"Falha ao cancelar {act_id}. Resposta: {response}")
        except Exception as e:
            results["err"] += 1; item["exception"] = str(e); log_action_to_firestore(db_firestore, user, "process_cancellation_exception", item); st.error(f"Erro de exceção ao cancelar {act_id}: {e}")
        progress.progress((i + 1) / len(to_cancel_with_context))
    st.success(f"Processamento via API concluído! Sucessos: {results['ok']}, Falhas: {results['err']}.")
    if client.dry_run: st.warning("Atenção: O modo Teste (Dry-run) está ativo.")
    
    st.info("Iniciando verificação de status no banco de dados...")
    if ids_to_check:
        status_atual = verificar_status_atividades(engine, ids_to_check)
        confirmados = 0
        discrepancias = []
        for act_id in ids_to_check:
            status = status_atual.get(act_id)
            if status and "Cancelad" in status:
                confirmados += 1
            else:
                discrepancias.append({"ID da Atividade": act_id, "Status Encontrado": status or "Não encontrado"})
        st.success(f"Verificação no banco de dados concluída: {confirmados} de {len(ids_to_check)} atividades foram confirmadas como 'Cancelada'.")
        if discrepancias:
            st.warning("As seguintes atividades não puderam ser confirmadas como canceladas no banco de dados:")
            st.dataframe(discrepancias)

    for g_state in st.session_state[SK.GROUP_STATES].values(): g_state["cancelados"].clear()
    st.cache_data.clear(); st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()

@st.dialog("Confirmação de Cancelamento")
def confirm_cancellation_dialog(all_groups: List[List[Dict]], user: str, db_firestore, engine: Engine):
    to_cancel_with_context = []
    score_cache = {}
    for g in all_groups:
        group_key = generate_group_key(g)
        state = st.session_state[SK.GROUP_STATES].get(group_key, {})
        if state.get("cancelados"):
            principal_id = state.get("principal_id")
            principal_row = next((r for r in g if r['activity_id'] == principal_id), None)
            if not principal_row: continue
            p_norm = normalize_for_match(principal_row.get("Texto", ""), [])
            p_meta = extract_meta(principal_row.get("Texto", ""))
            for cancel_id in state.get("cancelados", set()):
                cancel_row = next((r for r in g if r['activity_id'] == cancel_id), None)
                if not cancel_row: continue
                if (principal_id, cancel_id) not in score_cache:
                    c_norm = normalize_for_match(cancel_row.get("Texto", ""), [])
                    c_meta = extract_meta(cancel_row.get("Texto", ""))
                    score, _ = combined_score(p_norm, c_norm, p_meta, c_meta)
                    score_cache[(principal_id, cancel_id)] = score
                to_cancel_with_context.append({"ID a Cancelar": cancel_id, "Duplicata do Principal": principal_id, "Pasta": cancel_row.get("activity_folder", "N/A"), "Similaridade (%)": f"{score_cache[(principal_id, cancel_id)]:.0f}"})
    if not to_cancel_with_context:
        st.info("Nenhuma atividade foi marcada para cancelamento.");
        if st.button("Fechar"): st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()
        return
    st.warning(f"Você está prestes a cancelar **{len(to_cancel_with_context)}** atividades. Esta ação é irreversível."); 
    st.dataframe(to_cancel_with_context, use_container_width=True)
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirmar e Cancelar", type="primary", use_container_width=True): process_cancellations(to_cancel_with_context, user, db_firestore, engine)
    if col2.button("Voltar", use_container_width=True): st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()

def render_calibration_tab(df: pd.DataFrame):
    st.subheader("📊 Calibração de Similaridade por Pasta")
    if df.empty: st.warning("Não há dados para calibrar."); return
    pastas_disponiveis = sorted(df["activity_folder"].dropna().unique())
    if not pastas_disponiveis: st.warning("Nenhuma pasta com dados suficientes para análise."); return
    pasta = st.selectbox("Selecione uma pasta:", pastas_disponiveis)
    num_samples = st.slider("Nº de Pares Aleatórios", 50, 2000, 500, 50)
    if st.button("Analisar Pasta"):
        sample_df = df[df["activity_folder"] == pasta].copy()
        if len(sample_df) < 2: st.warning("A pasta tem menos de 2 atividades."); return
        textos = carregar_textos_por_id(db_engine_mysql(), tuple(sample_df['activity_id'].unique()))
        sample_df['Texto'] = sample_df['activity_id'].map(textos)
        sample_df.dropna(subset=['Texto'], inplace=True)
        stopwords_extra = st.secrets.get("similarity", {}).get("stopwords_extra", [])
        sample_df["_meta"] = sample_df["Texto"].apply(extract_meta); sample_df["_norm"] = sample_df["Texto"].apply(lambda t: normalize_for_match(t, stopwords_extra)); sample_df = sample_df.reset_index()
        n = len(sample_df); indices = np.arange(n); pairs = set(); rng = np.random.default_rng(seed=42)
        while len(pairs) < min(num_samples, (n * (n - 1)) // 2): pairs.add(tuple(sorted(rng.choice(indices, size=2, replace=False))))
        scores = []; progress = st.progress(0, text="Calculando scores...")
        for i, (idx1, idx2) in enumerate(pairs):
            row1, row2 = sample_df.iloc[idx1], sample_df.iloc[idx2]
            score, _ = combined_score(row1["_norm"], row2["_norm"], row1["_meta"], row2["_meta"])
            scores.append({"score": score})
            progress.progress((i + 1) / len(pairs))
        progress.empty()
        df_scores = pd.DataFrame(scores); st.write("Estatísticas Descritivas:"); st.dataframe(df_scores["score"].describe(percentiles=[.25, .5, .75, .9, .95, .99]))
        if alt: st.altair_chart(alt.Chart(df_scores).mark_bar().encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=50)), y=alt.Y("count()")).properties(title=f"Distribuição para: {pasta}", height=300), use_container_width=True)

@st.cache_data(ttl=600)
def get_firestore_history(_db, limit=100):
    if _db is None: return []
    try: return [doc.to_dict() for doc in _db.collection("duplicidade_actions").order_by("ts", direction=firestore.Query.DESCENDING).limit(limit).stream()]
    except Exception as e: st.error(f"Erro ao buscar histórico do Firestore: {e}"); return []

def format_history_for_display(history: List[Dict]) -> pd.DataFrame:
    if not history: return pd.DataFrame()
    parsed_logs = []
    for log in history:
        ts = log.get("ts"); ts_local = ts.astimezone(TZ_SP) if isinstance(ts, datetime) else "N/A"
        details = log.get("details", {})
        action_map = {"set_principal": "Definição de Principal", "mark_all_cancel": "Marcar Todos para Cancelar", "mark_not_duplicate": "Marcar Grupo como 'Não Duplicado'", "mark_cancel": "Marcar para Cancelar", "unmark_cancel": "Desmarcar para Cancelar", "process_cancellation_success": "Cancelamento via API (Sucesso)", "process_cancellation_failure": "Cancelamento via API (Falha)", "process_cancellation_exception": "Cancelamento via API (Erro)"}
        description = ""
        if log.get("action") == "set_principal": description = f"ID {details.get('new_principal_id')} definido como principal, substituindo {details.get('previous_principal_id')}."
        elif log.get("action") == "mark_cancel": description = f"ID {details.get('target_activity_id')} marcado para ser cancelado (principal: {details.get('principal_id')})."
        elif log.get("action") == "unmark_cancel": description = f"ID {details.get('target_activity_id')} desmarcado (principal: {details.get('principal_id')})."
        elif log.get("action") == "process_cancellation_success": description = f"ID {details.get('ID a Cancelar')} cancelado com sucesso (duplicata de {details.get('Duplicata do Principal')})."
        elif log.get("action") == "mark_not_duplicate": description = f"Grupo iniciado por {details.get('group_id')} foi marcado como 'Não é Duplicado'."
        else: description = json.dumps(details, ensure_ascii=False)
        parsed_logs.append({"Data": ts_local.strftime('%d/%m/%Y %H:%M:%S') if ts_local != "N/A" else "N/A", "Usuário": log.get("user", "N/A"), "Ação": action_map.get(log.get("action"), log.get("action", "N/A")), "Pasta": details.get("pasta", details.get("Pasta", "N/A")), "Detalhes": description})
    return pd.DataFrame(parsed_logs)

def render_history_tab(db_firestore):
    st.subheader("📜 Histórico de Ações (Auditoria)")
    if db_firestore is None: st.warning("A conexão com o Firebase (auditoria) não está ativa."); return
    if st.button("Atualizar Histórico"): get_firestore_history.clear()
    history = get_firestore_history(db_firestore)
    if not history: st.info("Nenhum registro de auditoria encontrado."); return
    friendly_tab, raw_tab = st.tabs(["Visão Amigável (Tabela)", "Dados Brutos (JSON)"])
    with friendly_tab:
        st.write("Uma visão simplificada das ações realizadas, ideal para auditoria rápida.")
        df_friendly = format_history_for_display(history)
        st.dataframe(df_friendly, use_container_width=True, hide_index=True)
        csv_data = df_friendly.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Exportar para CSV", data=csv_data, file_name=f"historico_duplicidades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    with raw_tab:
        st.write("Os dados completos como estão armazenados no banco de dados. Útil para depuração.")
        def json_converter(o):
            if isinstance(o, datetime): return o.isoformat()
        json_data = json.dumps(history, default=json_converter, indent=2, ensure_ascii=False)
        st.download_button(label="⬇️ Exportar para JSON", data=json_data, file_name=f"historico_duplicidades_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
        st.json(history)

# =============================================================================
# FLUXO PRINCIPAL DO APLICATIVO
# =============================================================================
def main():
    st.title(APP_TITLE)
    for key in [SK.USERNAME, SK.GROUP_STATES, SK.CFG, SK.SHOW_CANCEL_CONFIRM, SK.IGNORED_GROUPS, SK.ANALYSIS_RESULTS]:
        if key not in st.session_state:
            st.session_state[key] = set() if key == SK.IGNORED_GROUPS else {} if key in [SK.GROUP_STATES, SK.ANALYSIS_RESULTS] else False

    if not st.session_state.get(SK.USERNAME):
        with st.sidebar.form("login_form"):
            username = st.text_input("Nome de Usuário"); password = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                creds = st.secrets.get("credentials", {})
                user_pass = creds.get("usernames", {}).get(username)
                if user_pass and user_pass == password:
                    st.session_state[SK.USERNAME] = username; st.rerun()
                else: st.sidebar.error("Usuário ou senha inválidos.")
        st.info("👋 Bem-vindo! Por favor, faça o login na barra lateral."); st.stop()

    engine = db_engine_mysql(); db_firestore = init_firebase()

    with st.spinner("Carregando dados mestre..."):
        df_full = carregar_dados_minimos(engine, 365)
    
    params = sidebar_controls(df_full)
    
    data_limite_analise = datetime.now(TZ_UTC) - timedelta(days=params["dias_hist"])
    df_analysis_min = df_full[df_full['activity_date'] >= data_limite_analise].copy()

    if params["analysis_submitted"]:
        if df_analysis_min.empty:
            st.warning("Nenhuma atividade encontrada para o período de análise.")
            st.session_state[SK.ANALYSIS_RESULTS] = []
        else:
            with st.spinner("Buscando textos para análise..."):
                all_ids_to_fetch = tuple(df_analysis_min['activity_id'].unique())
                textos = carregar_textos_por_id(engine, all_ids_to_fetch)
            
            core_params = {k: params[k] for k in ['min_sim', 'min_containment', 'pre_delta', 'use_cnj']}
            st.session_state[SK.ANALYSIS_RESULTS] = criar_grupos_de_duplicatas_hibrido(df_analysis_min, core_params, textos)
    
    all_groups = st.session_state.get(SK.ANALYSIS_RESULTS, [])
    if not all_groups:
        st.info("Clique em 'Aplicar e Analisar' na barra lateral para iniciar a busca por duplicatas."); st.stop()
    
    final_filtered_groups = []
    for group in all_groups:
        group_key = generate_group_key(group)
        if group_key in st.session_state[SK.IGNORED_GROUPS]: continue
        
        # LÓGICA CORRIGIDA: Oculta o grupo apenas se restar 1 ou 0 atividades não canceladas.
        non_canceled_count = sum(1 for r in group if "Cancelad" not in r.get("activity_status", ""))
        if non_canceled_count <= 1:
            continue

        if params["pastas"] and group[0].get("activity_folder") not in params["pastas"]: continue
        if params["only_groups_with_open"] and not any(r.get("activity_status") == "Aberta" for r in group): continue
        if params["status"] and not any(r.get("activity_status") in params["status"] for r in group): continue
        final_filtered_groups.append(group)

    tab1, tab2, tab3 = st.tabs(["🔎 Análise de Duplicidades", "📊 Calibração", "📜 Histórico de Ações"])

    with tab1:
        st.metric("Grupos de Duplicatas Encontrados (após filtros)", len(final_filtered_groups))
        page_size = st.number_input("Grupos por página", min_value=1, value=DEFAULTS["itens_por_pagina"], step=1)
        total_pages = max(1, math.ceil(len(final_filtered_groups) / page_size))
        page_num = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page_num - 1) * page_size; end_idx = start_idx + page_size
        st.caption(f"Exibindo grupos {start_idx + 1}–{min(end_idx, len(final_filtered_groups))} de {len(final_filtered_groups)}")

        for group_rows in final_filtered_groups[start_idx:end_idx]:
            render_group(group_rows, params, db_firestore, engine)

        st.markdown("---"); st.header("⚡ Ações em Massa")
        if st.button("🚀 Processar Cancelamentos Marcados", type="primary", use_container_width=True):
            st.session_state[SK.SHOW_CANCEL_CONFIRM] = True
        
        if st.session_state.get(SK.SHOW_CANCEL_CONFIRM):
            confirm_cancellation_dialog(final_filtered_groups, st.session_state.get(SK.USERNAME), db_firestore, engine)

    with tab2: render_calibration_tab(df_full)
    with tab3: render_history_tab(db_firestore)

if __name__ == "__main__":
    main()
