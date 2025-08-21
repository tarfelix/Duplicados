# -*- coding: utf-8 -*-
"""
Verificador de Duplicidade — Versão Otimizada e Corrigida
=========================================================

Este aplicativo combina as funcionalidades avançadas da versão 'appdelta'
com as otimizações de performance e a lógica de negócio corrigida para
a seleção do item principal e exibição de grupos.

Principais Correções e Melhorias:
- Lógica de Filtro Corrigida: O filtro "Apenas grupos com atividades abertas"
  agora funciona corretamente e tem prioridade sobre a seleção de status.
- Ordem de Exibição Corrigida: O item "Principal" agora é sempre o primeiro
  a ser exibido dentro de um grupo.
- Estilo Visual Aprimorado: As cores para os status "Principal" e "Cancelado"
  estão mais vivas e destacadas.
- Lógica do Modo Estrito Aprimorada: Garante que apenas itens que atendam
  diretamente ao critério de similaridade com o principal sejam exibidos.
- Similaridade Padrão Ajustada: O valor padrão da similaridade global foi
  reforçado para 95% na interface.
- Histórico Aprimorado: A aba de histórico agora exibe a Pasta da atividade
  e mantém as opções de exportação para CSV e JSON.
"""
from __future__ import annotations

import os
import re
import html
import logging
import time
import math
import json
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
SUFFIX = "_v10_final_filters"
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
        font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; height: 360px; overflow-y: auto;
    }
    .diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
    .diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }
    .card { border-left: 5px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #fff; }
    .card-cancelado { background-color: #FFEBEE; border-left: 5px solid #F44336; }
    .card-principal { background-color: #E8F5E9; border-left: 5px solid #4CAF50; }
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
    """
    Carrega dados do banco.
    Nota sobre performance: A lentidão inicial pode vir da complexidade da
    ViewGrdAtividadesTarcisio no banco. Garantir que as colunas `activity_type`
    e `activity_date` estejam indexadas na tabela original pode acelerar a consulta.
    """
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
    min_sim_default = int(sim_cfg.get("min_sim_global", DEFAULTS["min_sim_global"]))
    min_sim = st.sidebar.slider("Similaridade Mínima Global (%)", 0, 100, min_sim_default, 1) / 
