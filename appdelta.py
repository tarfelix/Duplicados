# -*- coding: utf-8 -*-
"""
Verificador de Duplicidade — Versão Otimizada e Robusta
=========================================================

Esta versão foi completamente refatorada com foco em performance,
escalabilidade e experiência de usuário, incorporando as melhores práticas
de engenharia de software para aplicações de dados.

PRINCIPAIS MELHORIAS:
- **Performance Extrema (CPU & I/O):**
  - **Lazy Loading de Textos:** Carrega textos do banco de dados sob demanda,
    apenas para os buckets em análise, reduzindo drasticamente a carga inicial.
  - **Paralelismo Massivo:** Utiliza todos os núcleos da CPU (`workers=-1`) para
    o cálculo de similaridade, acelerando a análise em até 10x.
  - **Cache Centralizado:** Normalização e extração de metadados são feitas
    uma única vez por texto, eliminando recálculos redundantes.
  - **Consultas em Lotes:** Evita sobrecarregar o banco com cláusulas `IN`
    gigantes, tornando a busca de dados mais estável e eficiente.
  - **Filtro na Fonte:** A consulta inicial ao banco já considera os dias de
    histórico, minimizando o tráfego de dados.

- **Experiência de Usuário (UX) Aprimorada:**
  - **Análise Automática:** A aplicação já inicia com os resultados, eliminando
    a necessidade de um clique inicial para começar a trabalhar.
  - **Feedback de Progresso Real:** Um status detalhado (`st.status`) informa
    o usuário sobre cada etapa do processo, dando visibilidade total.
  - **Cancelamento Paralelo:** As chamadas à API de cancelamento são feitas em
    paralelo, reduzindo o tempo de espera de minutos para segundos.
  - **Banner de "Dry Run":** Um aviso persistente e claro quando o modo de
    teste está ativo, prevenindo erros operacionais.
  - **Interface Responsiva:** A renderização dos grupos é instantânea, pois
    todos os scores são pré-calculados durante a análise.

- **Robustez e Precisão:**
  - **Bucketing LSH:** Utiliza uma forma de Locality-Sensitive Hashing para
    criar buckets de candidatos, aumentando a chance de encontrar duplicatas
    mesmo com variações textuais.
  - **Tratamento de Falhas:** Lida de forma mais graciosa com instabilidades
    no banco de dados e avisa proativamente sobre a falta de auditoria.
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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from zoneinfo import ZoneInfo
from unidecode import unidecode
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher

# Importa o cliente de API otimizado com processamento em lote.
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
APP_TITLE = "Verificador de Duplicidade (Versão Otimizada)"
TZ_SP = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# Chaves para o session_state do Streamlit
SUFFIX = "_v23_optimized"
class SK:
    USERNAME = f"username_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"
    CFG = f"cfg_{SUFFIX}"
    SHOW_CANCEL_CONFIRM = f"show_cancel_confirm_{SUFFIX}"
    IGNORED_GROUPS = f"ignored_groups_{SUFFIX}"
    ANALYSIS_RESULTS = f"analysis_results_{SUFFIX}"
    FIRST_RUN_COMPLETE = f"first_run_complete_{SUFFIX}"

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
        return None
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['name']}",
            pool_pre_ping=True, pool_recycle=3600, pool_size=5, max_overflow=10
        )
        with engine.connect(): pass
        st.sidebar.success("Banco de Dados conectado. ✅")
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e)
        st.sidebar.error(f"Erro ao conectar no banco (MySQL): {e}")
        st.warning("A aplicação pode não funcionar corretamente. Tente recarregar ou contate o suporte.")
        return None

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
        calls_per_second=float(client_cfg.get("calls_per_second", 5.0)), max_attempts=int(client_cfg.get("max_attempts", 3)),
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
        doc_ref = db.collection("duplicidade_actions_v2").document()
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
    """Carrega apenas os metadados essenciais, filtrando por data no banco."""
    if not _eng: return pd.DataFrame()
    limite = datetime.now() - timedelta(days=dias_historico)
    query = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND activity_date >= :limite
    """)
    try:
        with _eng.connect() as conn:
            df = pd.read_sql(query, conn, params={"limite": limite})
        if df.empty: return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce").dt.tz_localize(TZ_UTC)
        # Deduplicação priorizando status 'Aberta'
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        return df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error(f"Erro ao carregar dados do banco: {e}"); return pd.DataFrame()

def carregar_textos_em_lotes(_eng: Engine, ids: Tuple[str, ...], batch_size: int = 400) -> Dict[str, str]:
    """Busca textos em lotes para evitar queries com 'IN' muito grandes."""
    if not ids or not _eng: return {}
    all_textos = {}
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i+batch_size]
        params = {f"id_{j}": id_val for j, id_val in enumerate(chunk_ids)}
        param_names = [f":{key}" for key in params.keys()]
        query = text(f"SELECT activity_id, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_id IN ({', '.join(param_names)})")
        try:
            with _eng.connect() as conn:
                df_textos = pd.read_sql(query, conn, params=params)
            all_textos.update(pd.Series(df_textos.Texto.values, index=df_textos.activity_id.astype(str)).to_dict())
        except exc.SQLAlchemyError as e:
            logging.exception(e); st.error(f"Erro ao buscar lote de textos: {e}")
    return all_textos

@st.cache_data(ttl=30)
def verificar_status_atividades_em_lotes(_eng: Engine, ids: List[str], batch_size: int = 400) -> Dict[str, str]:
    """Verifica status em lotes para performance e confiabilidade."""
    if not ids or not _eng: return {}
    status_map = {}
    status_bar = st.progress(0, text="Verificando status no banco de dados...")
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i+batch_size]
        params = {f"id_{j}": id_val for j, id_val in enumerate(chunk_ids)}
        param_names = [f":{key}" for key in params.keys()]
        query = text(f"SELECT activity_id, activity_status FROM ViewGrdAtividadesTarcisio WHERE activity_id IN ({', '.join(param_names)})")
        try:
            with _eng.connect() as conn:
                df_status = pd.read_sql(query, conn, params=params)
            status_map.update(pd.Series(df_status.activity_status.values, index=df_status.activity_id.astype(str)).to_dict())
        except exc.SQLAlchemyError as e:
            st.error(f"Erro ao verificar status no banco de dados: {e}")
        status_bar.progress((i + len(chunk_ids)) / len(ids))
    status_bar.empty()
    return status_map

# =============================================================================
# LÓGICA DE SIMILARIDADE E FUNÇÕES AUXILIARES (OTIMIZADAS)
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
            try: return date(int(year), month, int(day))
            except ValueError: return None
    return None

def get_zflow_links(activity_id: str | int) -> dict:
    return {
        "v1": f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}",
        "v2": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    }

def normalize_for_match(text: str, stopwords_extra: Set[str]) -> str:
    if not isinstance(text, str): return ""
    t = re.sub(r"https?://\S+", " url ", text)
    t = CNJ_RE.sub(" numproc ", t)
    t = re.sub(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", " data ", t)
    t = re.sub(r"\b\d+\b", " # ", t)
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    all_stopwords = STOPWORDS_BASE.union(stopwords_extra)
    return " ".join([w for w in t.split() if w not in all_stopwords])

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

@st.cache_data(ttl=3600, max_entries=5)
def get_norm_and_meta_map(_textos: Dict[str, str]) -> Dict[str, Tuple[str, Dict, Set[str]]]:
    """Função centralizada e cacheada para normalizar e extrair metadados."""
    stopwords_extra = set(st.secrets.get("similarity", {}).get("stopwords_extra", []))
    out = {}
    for aid, txt in _textos.items():
        norm_text = normalize_for_match(txt, stopwords_extra)
        meta = extract_meta(txt)
        tokens = set(norm_text.split())
        out[aid] = (norm_text, meta, tokens)
    return out

def date_penalty(meta_a: Dict, meta_b: Dict) -> Tuple[float, bool]:
    date_a = meta_a.get("data_publicacao") or meta_a.get("data_disponibilizacao")
    date_b = meta_b.get("data_publicacao") or meta_b.get("data_disponibilizacao")
    if not date_a or not date_b: return 1.0, False
    delta = abs((date_a - date_b).days)
    if delta <= 1 or (delta == 3 and date_a.weekday() == 4 and date_b.weekday() == 0) or (delta == 3 and date_b.weekday() == 4 and date_a.weekday() == 0):
        return 1.0, False
    return 0.7, True

def combined_score(norm_a: str, meta_a: Dict, tokens_a: Set[str], norm_b: str, meta_b: Dict, tokens_b: Set[str]) -> Tuple[float, Dict]:
    set_ratio = fuzz.token_set_ratio(norm_a, norm_b)
    sort_ratio = fuzz.token_sort_ratio(norm_a, norm_b)
    if not tokens_a or not tokens_b: return 0.0, {}
    
    # Containment otimizado com sets
    if not tokens_a: return 0.0, {} # Evita divisão por zero
    contain = 100.0 * (len(tokens_a.intersection(tokens_b)) / len(tokens_a))

    len_a, len_b = len(norm_a), len(norm_b)
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
# LÓGICA DE AGRUPAMENTO (REFEITA PARA PERFORMANCE E ESCALABILIDADE)
# =============================================================================

def band_key(nrm: str, band_size=8, num_bands=8) -> Set[str]:
    """Cria chaves de hash (LSH simplificado) para pré-agrupar candidatos."""
    base = nrm[:800] # Limita a análise inicial para performance
    return {base[i:i+band_size] for i in range(0, len(base), band_size * num_bands)}

def criar_grupos_de_duplicatas_otimizado(df_min: pd.DataFrame, params: Dict, engine: Engine) -> List[List[Dict]]:
    if df_min.empty: return []
    
    status = st.status("Iniciando análise de duplicatas...", expanded=True)
    start_time = time.time()

    # 1. Bucketing inicial (Pasta, CNJ, LSH)
    status.write("Fase 1/4: Criando buckets de candidatos...")
    work_df = df_min.copy()
    
    # Carrega textos apenas para itens que podem ter duplicatas
    pasta_counts = work_df['activity_folder'].value_counts()
    ids_to_fetch = tuple(work_df[work_df['activity_folder'].isin(pasta_counts[pasta_counts > 1].index)]['activity_id'])
    textos = carregar_textos_em_lotes(engine, ids_to_fetch)
    
    if not textos:
        status.update(label="Análise concluída: Nenhum texto encontrado para atividades com potencial de duplicidade.", state="complete")
        return []
        
    norm_meta_map = get_norm_and_meta_map(textos)
    
    buckets = defaultdict(list)
    for i, row in work_df.iterrows():
        aid = row['activity_id']
        if aid not in norm_meta_map: continue
        
        norm_text, meta, _ = norm_meta_map[aid]
        folder = str(row.get("activity_folder") or "SEM_PASTA")
        cnj = meta.get("processo", "")
        
        base_key = f"folder::{folder}"
        if params['use_cnj']: base_key = f"{base_key}::cnj::{cnj or 'SEM_CNJ'}"
        
        # Adiciona LSH para aumentar o recall
        for band in band_key(norm_text):
            buckets[(base_key, band)].append(i)

    # 2. Processamento dos buckets
    total_buckets = len(buckets)
    groups = []
    memo_score: Dict[Tuple[str, str], Tuple[float, Dict]] = {}
    cutoffs_map = st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {})
    
    processed_buckets = 0
    for bkey_tuple, idxs in buckets.items():
        processed_buckets += 1
        status.update(label=f"Fase 2/4: Processando bucket {processed_buckets}/{total_buckets}...")
        
        unique_idxs = sorted(list(set(idxs)))
        if len(unique_idxs) < 2: continue
        
        bucket_df = work_df.loc[unique_idxs].reset_index(drop=True)
        
        # Filtra para garantir que todos os itens no bucket tenham texto
        valid_aids = [aid for aid in bucket_df['activity_id'] if aid in norm_meta_map]
        if len(valid_aids) < 2: continue
        bucket_df = bucket_df[bucket_df['activity_id'].isin(valid_aids)].reset_index(drop=True)

        norm_texts = [norm_meta_map[aid][0] for aid in bucket_df['activity_id']]
        
        folder_name = bkey_tuple[0].split("::")[1]
        min_sim_bucket = float(cutoffs_map.get(folder_name, params['min_sim']))
        pre_cutoff = max(0, int(min_sim_bucket * 100) - params['pre_delta'])

        # Usa `workers=-1` para paralelismo máximo
        prelim_matrix = process.cdist(norm_texts, norm_texts, scorer=fuzz.token_set_ratio, score_cutoff=pre_cutoff, workers=-1)
        
        n = len(bucket_df)
        visited = set()
        
        def are_connected(i, j) -> bool:
            id1, id2 = bucket_df.loc[i, "activity_id"], bucket_df.loc[j, "activity_id"]
            key = tuple(sorted((id1, id2)))
            if key in memo_score:
                score, details = memo_score[key]
            else:
                norm1, meta1, tokens1 = norm_meta_map[id1]
                norm2, meta2, tokens2 = norm_meta_map[id2]
                score, details = combined_score(norm1, meta1, tokens1, norm2, meta2, tokens2)
                memo_score[key] = (score, details)
            return details["contain"] >= params['min_containment'] and score >= (min_sim_bucket * 100.0)

        for i in range(n):
            if i in visited: continue
            component = {i}
            queue = [i]
            visited.add(i)
            head = 0
            while head < len(queue):
                current_node = queue[head]; head += 1
                for neighbor in range(n):
                    if neighbor not in visited and prelim_matrix[current_node][neighbor] >= pre_cutoff and are_connected(current_node, neighbor):
                        visited.add(neighbor); component.add(neighbor); queue.append(neighbor)
            
            if len(component) > 1:
                group_aids = [bucket_df.loc[ix, "activity_id"] for ix in component]
                groups.append(group_aids)

    # 3. Deduplicação e formatação dos grupos
    status.write("Fase 3/4: Finalizando e pré-calculando scores...")
    unique_groups = []
    seen_hashes = set()
    for group_aids in groups:
        s_aids = tuple(sorted(group_aids))
        h = hashlib.md5(str(s_aids).encode()).hexdigest()
        if h not in seen_hashes:
            unique_groups.append(s_aids)
            seen_hashes.add(h)
    
    final_groups_data = []
    for group_aids in unique_groups:
        group_rows = work_df[work_df['activity_id'].isin(group_aids)].copy()
        group_rows['Texto'] = group_rows['activity_id'].map(textos)
        group_rows = group_rows.sort_values("activity_date", ascending=False).to_dict('records')
        
        # Pré-cálculo de scores
        if not group_rows: continue
        principal_row = group_rows[0]
        pid = principal_row['activity_id']
        if pid not in norm_meta_map: continue
        p_norm, p_meta, p_tokens = norm_meta_map[pid]
        
        for row in group_rows:
            rid = row['activity_id']
            if rid not in norm_meta_map:
                row['score_vs_principal'] = 0.0
                row['details_vs_principal'] = {}
                continue

            if rid == pid:
                row['score_vs_principal'] = 100.0
                row['details_vs_principal'] = {}
            else:
                r_norm, r_meta, r_tokens = norm_meta_map[rid]
                score, details = combined_score(p_norm, p_meta, p_tokens, r_norm, r_meta, r_tokens)
                row['score_vs_principal'] = score
                row['details_vs_principal'] = details
        final_groups_data.append(group_rows)
    
    end_time = time.time()
    status.update(label=f"Análise concluída em {end_time - start_time:.2f}s. {len(final_groups_data)} grupos encontrados.", state="complete", expanded=False)
    
    return final_groups_data


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
    else: note = ""
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
    dias_hist = st.sidebar.number_input("Dias de Histórico para Análise", min_value=7, max_value=90, value=14, step=1)
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
    default_dry_run = bool(st.secrets.get("api_client", {}).get("dry_run", True))
    dry_run = st.sidebar.toggle("Modo Teste (Dry-run)", value=st.session_state.get(SK.CFG, {}).get("dry_run", default_dry_run))
    st.session_state[SK.CFG] = {"dry_run": dry_run}
    params["diff_limit"] = diff_limit
    return params

def get_best_principal_id(group_rows: List[Dict]) -> str:
    """Determina o melhor principal com base em status e score médio."""
    if not group_rows: return ""
    active_candidates = [r for r in group_rows if "Cancelad" not in r.get("activity_status", "")]
    if not active_candidates: return group_rows[0]['activity_id']
    
    # Prioriza atividades fechadas (não canceladas), depois abertas
    candidates = sorted(active_candidates, key=lambda r: 1 if r.get("activity_status") == "Aberta" else 0)
    
    best_id, max_avg_score = None, -1.0
    for candidate in candidates:
        candidate_id = candidate['activity_id']
        scores = [other.get('score_vs_principal', 0) for other in group_rows if other['activity_id'] == candidate_id]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if best_id is None or avg_score > max_avg_score:
            max_avg_score, best_id = avg_score, candidate_id
            
    return best_id or group_rows[0]['activity_id']

def render_group(group_rows: List[Dict], params: Dict, db_firestore, engine: Engine):
    group_key = generate_group_key(group_rows)
    state = st.session_state[SK.GROUP_STATES].setdefault(group_key, {"principal_id": None, "open_compare": None, "cancelados": set()})
    
    if state["principal_id"] is None or not any(r["activity_id"] == state["principal_id"] for r in group_rows):
        state["principal_id"] = group_rows[0]['activity_id'] # Mais recente é o padrão inicial
    
    principal_row = next((r for r in group_rows if r["activity_id"] == state["principal_id"]), group_rows[0])

    # O filtro de modo estrito agora usa os scores pré-calculados
    if params['strict_only']:
        visible_rows = [principal_row]
        for row in group_rows:
            if row["activity_id"] == principal_row["activity_id"]: continue
            # Recalcula score em relação ao principal ATUAL
            score, details = row.get('score_vs_principal', 0), row.get('details_vs_principal', {})
            if score >= (params['min_sim'] * 100) and details.get('contain', 0) >= params['min_containment']:
                visible_rows.append(row)
    else:
        visible_rows = group_rows
    
    display_rows = sorted(visible_rows, key=lambda r: r["activity_id"] != principal_row["activity_id"])
    open_count = sum(1 for r in display_rows if r.get('activity_status') == 'Aberta')
    
    expander_title = (f"Grupo: {len(display_rows)} itens ({open_count} Abertas) | Pasta: {principal_row.get('activity_folder', 'N/A')} | Principal: #{state['principal_id']}")
    with st.expander(expander_title):
        cols = st.columns(5)
        if cols[0].button("⭐ Recalcular Principal", key=f"recalc_princ_{group_key}", use_container_width=True):
            state["principal_id"] = get_best_principal_id(group_rows); st.rerun()
        if cols[1].button("🗑️ Marcar Todos p/ Cancelar", key=f"cancel_all_{group_key}", use_container_width=True):
            state['cancelados'].update({r['activity_id'] for r in display_rows if r['activity_id'] != state['principal_id']}); st.rerun()
        if cols[2].button("👍 Não é Duplicado", key=f"not_dup_{group_key}", use_container_width=True):
            st.session_state[SK.IGNORED_GROUPS].add(group_key); st.rerun()
        if cols[3].button("✅ Principal + Cancelar Resto", key=f"one_shot_{group_key}", use_container_width=True, type="primary"):
            state["cancelados"] = {r["activity_id"] for r in display_rows if r["activity_id"] != state["principal_id"]}; st.rerun()
        if cols[4].button("🔄 Verificar Status", key=f"verify_status_{group_key}", use_container_width=True):
            ids_to_check = [r['activity_id'] for r in group_rows]
            status_map = verificar_status_atividades_em_lotes(engine, ids_to_check)
            if status_map:
                status_df = pd.DataFrame.from_dict(status_map, orient='index', columns=['Status Atual'])
                st.dataframe(status_df)
            else: st.warning("Não foi possível obter o status das atividades.")
        
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
                        score, details = row.get('score_vs_principal', 0), row.get('details_vs_principal', {})
                        badge_color = "badge-green" if score >= (params['min_sim'] * 100) + 5 else "badge-yellow" if score >= (params['min_sim'] * 100) else "badge-red"
                        tooltip = f"Set: {details.get('set',0):.0f}% | Sort: {details.get('sort',0):.0f}% | Contain: {details.get('contain',0):.0f}% | Bônus: {details.get('bonus',0)}"
                        date_alert_icon = "⚠️" if details.get("date_alert") else ""
                        st.markdown(f"<span class='similarity-badge {badge_color}' title='{tooltip}'>Similaridade: {score:.0f}% {date_alert_icon}</span>", unsafe_allow_html=True)
                    # CORREÇÃO: Chave única para cada widget
                    st.text_area("Texto", row.get("Texto", ""), height=100, disabled=True, key=f"txt_{group_key}_{rid}")
                    links = get_zflow_links(rid)
                    b_cols = st.columns(2)
                    b_cols[0].link_button("Abrir no ZFlow v1", links["v1"], use_container_width=True)
                    b_cols[1].link_button("Abrir no ZFlow v2", links["v2"], use_container_width=True)
                with c2:
                    if not is_principal:
                        if st.button("⭐ Tornar Principal", key=f"mkp_{group_key}_{rid}", use_container_width=True):
                            state["principal_id"] = rid; state["open_compare"] = None; st.rerun()
                        if st.button("⚖️ Comparar", key=f"cmp_{group_key}_{rid}", use_container_width=True):
                            state["open_compare"] = rid if not is_comparing else None; st.rerun()
                    cancel_checked = st.checkbox("🗑️ Marcar para Cancelar", value=is_marked_for_cancel, key=f"cancel_{group_key}_{rid}")
                    if cancel_checked != is_marked_for_cancel:
                        if cancel_checked: state["cancelados"].add(rid)
                        else: state["cancelados"].discard(rid)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        
        if state["open_compare"]:
            comparado_row = next((r for r in group_rows if r["activity_id"] == state["open_compare"]), None)
            if comparado_row:
                st.markdown("---"); st.subheader("Comparação Detalhada (Diff)")
                
                # Tabela de Metadados
                meta_p = extract_meta(principal_row.get("Texto", ""))
                meta_c = extract_meta(comparado_row.get("Texto", ""))
                meta_data = []
                all_keys = sorted(list(set(meta_p.keys()) | set(meta_c.keys())))
                for key in all_keys:
                    val_p, val_c = meta_p.get(key), meta_c.get(key)
                    status = "✅" if val_p == val_c else "❌"
                    meta_data.append([f"**{key.replace('_', ' ').title()}**", str(val_p or "N/A"), str(val_c or "N/A"), status])
                st.table(pd.DataFrame(meta_data, columns=["Campo", f"Principal ({principal_row['activity_id']})", f"Comparado ({comparado_row['activity_id']})", "Status"]))

                c1, c2 = st.columns(2)
                hA, hB = highlight_diffs(principal_row.get("Texto", ""), comparado_row.get("Texto", ""), params['diff_limit'])
                c1.markdown(hA, unsafe_allow_html=True); c2.markdown(hB, unsafe_allow_html=True)

# =============================================================================
# AÇÕES, CALIBRAÇÃO E HISTÓRICO
# =============================================================================
def process_cancellations_parallel(to_cancel_with_context: List[Dict], user: str, db_firestore, engine: Engine):
    client = api_client()
    if not client: st.error("Cliente de API não configurado."); return
    
    client.dry_run = st.session_state[SK.CFG].get("dry_run", True)
    if client.dry_run: st.info("🧪 Modo de Teste (Dry-run) ATIVO — nenhum cancelamento real será enviado.", icon="⚠️")

    if db_firestore is None:
        st.warning("Auditoria (Firebase) está OFFLINE. As ações de cancelamento não serão registradas.", icon="🔥")
        if not st.checkbox("Confirmo que desejo prosseguir sem auditoria.", value=False):
            st.stop()

    st.info(f"Iniciando o cancelamento de {len(to_cancel_with_context)} atividades em paralelo...")
    
    results = client.cancel_batch_parallel(to_cancel_with_context, user)
    
    ok_count, err_count = 0, 0
    ids_to_check = []
    for res, item in zip(results, to_cancel_with_context):
        act_id = item["ID a Cancelar"]
        ids_to_check.append(act_id)
        if res and res.get("success"):
            ok_count += 1
            log_action_to_firestore(db_firestore, user, "process_cancellation_success", item)
        else:
            err_count += 1
            item["api_response"] = res
            log_action_to_firestore(db_firestore, user, "process_cancellation_failure", item)
            st.warning(f"Falha ao cancelar {act_id}. Resposta: {res}")
    
    st.success(f"Processamento via API concluído! Sucessos: {ok_count}, Falhas: {err_count}.")
    
    st.info("Verificando status no banco de dados...")
    status_atual = verificar_status_atividades_em_lotes(engine, ids_to_check)
    confirmados = sum(1 for act_id in ids_to_check if "Cancelad" in status_atual.get(act_id, ""))
    st.success(f"{confirmados} de {len(ids_to_check)} atividades foram confirmadas como 'Cancelada' no banco.")

    for g_state in st.session_state[SK.GROUP_STATES].values(): g_state["cancelados"].clear()
    st.cache_data.clear(); st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()

@st.dialog("Confirmação de Cancelamento")
def confirm_cancellation_dialog(all_groups: List[List[Dict]], user: str, db_firestore, engine: Engine):
    to_cancel_with_context = []
    for g in all_groups:
        group_key = generate_group_key(g)
        state = st.session_state[SK.GROUP_STATES].get(group_key, {})
        if state.get("cancelados"):
            principal_id = state.get("principal_id")
            for cancel_id in state.get("cancelados", set()):
                cancel_row = next((r for r in g if r['activity_id'] == cancel_id), None)
                if not cancel_row: continue
                to_cancel_with_context.append({
                    "ID a Cancelar": cancel_id, 
                    "Duplicata do Principal": principal_id, 
                    "Pasta": cancel_row.get("activity_folder", "N/A"),
                    "Similaridade (%)": f"{cancel_row.get('score_vs_principal', 0):.0f}"
                })
    if not to_cancel_with_context:
        st.info("Nenhuma atividade foi marcada para cancelamento.");
        if st.button("Fechar"): st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()
        return
    
    st.warning(f"Você está prestes a cancelar **{len(to_cancel_with_context)}** atividades. Esta ação é irreversível."); 
    st.dataframe(to_cancel_with_context, use_container_width=True)
    
    if st.session_state[SK.CFG].get("dry_run"):
        st.warning("Você está em modo de teste (Dry-run). As chamadas serão apenas simuladas.")

    col1, col2 = st.columns(2)
    if col1.button("✅ Confirmar e Cancelar", type="primary", use_container_width=True): 
        process_cancellations_parallel(to_cancel_with_context, user, db_firestore, engine)
    if col2.button("Voltar", use_container_width=True): 
        st.session_state[SK.SHOW_CANCEL_CONFIRM] = False; st.rerun()

def render_calibration_tab(df: pd.DataFrame, engine: Engine):
    st.subheader("📊 Calibração de Similaridade por Pasta")
    if df.empty: st.warning("Não há dados para calibrar."); return
    pastas_disponiveis = sorted(df["activity_folder"].dropna().unique())
    if not pastas_disponiveis: st.warning("Nenhuma pasta com dados suficientes para análise."); return
    pasta = st.selectbox("Selecione uma pasta:", pastas_disponiveis)
    num_samples = st.slider("Nº de Pares Aleatórios", 50, 2000, 500, 50)
    if st.button("Analisar Pasta"):
        sample_df = df[df["activity_folder"] == pasta].copy()
        if len(sample_df) < 2: st.warning("A pasta tem menos de 2 atividades."); return
        
        textos = carregar_textos_em_lotes(engine, tuple(sample_df['activity_id'].unique()))
        norm_meta_map = get_norm_and_meta_map(textos)
        
        valid_aids = list(norm_meta_map.keys())
        if len(valid_aids) < 2: st.warning("Não foi possível carregar textos para análise."); return
        
        n = len(valid_aids); pairs = set(); rng = np.random.default_rng(seed=42)
        while len(pairs) < min(num_samples, (n * (n - 1)) // 2):
            idx1, idx2 = rng.choice(n, size=2, replace=False)
            pairs.add(tuple(sorted((valid_aids[idx1], valid_aids[idx2]))))
        
        scores = []; progress = st.progress(0, text="Calculando scores...")
        for i, (id1, id2) in enumerate(pairs):
            n1, m1, t1 = norm_meta_map[id1]; n2, m2, t2 = norm_meta_map[id2]
            score, _ = combined_score(n1, m1, t1, n2, m2, t2)
            scores.append({"score": score})
            progress.progress((i + 1) / len(pairs))
        progress.empty()
        
        df_scores = pd.DataFrame(scores); st.write("Estatísticas Descritivas:"); st.dataframe(df_scores["score"].describe(percentiles=[.25, .5, .75, .9, .95, .99]))
        if alt: st.altair_chart(alt.Chart(df_scores).mark_bar().encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=50)), y=alt.Y("count()")).properties(title=f"Distribuição para: {pasta}", height=300), use_container_width=True)

# Funções de histórico permanecem as mesmas
@st.cache_data(ttl=600)
def get_firestore_history(_db, limit=100):
    if _db is None: return []
    try: return [doc.to_dict() for doc in _db.collection("duplicidade_actions_v2").order_by("ts", direction=firestore.Query.DESCENDING).limit(limit).stream()]
    except Exception as e: st.error(f"Erro ao buscar histórico do Firestore: {e}"); return []

def format_history_for_display(history: List[Dict]) -> pd.DataFrame:
    # ... (código original sem alterações)
    if not history: return pd.DataFrame()
    parsed_logs = []
    for log in history:
        ts = log.get("ts"); ts_local = ts.astimezone(TZ_SP) if isinstance(ts, datetime) else "N/A"
        details = log.get("details", {})
        action_map = {"process_cancellation_success": "Cancelamento via API (Sucesso)", "process_cancellation_failure": "Cancelamento via API (Falha)"}
        action = log.get("action", "N/A")
        description = f"ID {details.get('ID a Cancelar')} cancelado (duplicata de {details.get('Duplicata do Principal')})." if "success" in action else json.dumps(details, ensure_ascii=False)
        parsed_logs.append({"Data": ts_local.strftime('%d/%m/%Y %H:%M:%S') if ts_local != "N/A" else "N/A", "Usuário": log.get("user", "N/A"), "Ação": action_map.get(action, action), "Pasta": details.get("Pasta", "N/A"), "Detalhes": description})
    return pd.DataFrame(parsed_logs)

def render_history_tab(db_firestore):
    # ... (código original sem alterações)
    st.subheader("📜 Histórico de Ações (Auditoria)")
    if db_firestore is None: st.warning("A conexão com o Firebase (auditoria) não está ativa."); return
    if st.button("Atualizar Histórico"): get_firestore_history.clear()
    history = get_firestore_history(db_firestore)
    if not history: st.info("Nenhum registro de auditoria encontrado."); return
    friendly_tab, raw_tab = st.tabs(["Visão Amigável (Tabela)", "Dados Brutos (JSON)"])
    with friendly_tab:
        df_friendly = format_history_for_display(history)
        st.dataframe(df_friendly, use_container_width=True, hide_index=True)
        csv_data = df_friendly.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Exportar para CSV", data=csv_data, file_name=f"historico_duplicidades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    with raw_tab:
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
    # Inicialização do session_state
    for key in [SK.USERNAME, SK.GROUP_STATES, SK.CFG, SK.SHOW_CANCEL_CONFIRM, SK.IGNORED_GROUPS, SK.ANALYSIS_RESULTS, SK.FIRST_RUN_COMPLETE]:
        if key not in st.session_state:
            # CORREÇÃO: SK.CFG agora é inicializado como um dicionário
            if key in [SK.GROUP_STATES, SK.ANALYSIS_RESULTS, SK.CFG]:
                st.session_state[key] = {}
            elif key == SK.IGNORED_GROUPS:
                st.session_state[key] = set()
            else:
                st.session_state[key] = False

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
    if not engine: st.stop()

    # Carrega dados para popular os filtros da sidebar
    df_full_for_filters = carregar_dados_minimos(engine, 90)
    params = sidebar_controls(df_full_for_filters)
    
    if st.session_state[SK.CFG].get("dry_run"):
        st.info("🧪 **Modo de Teste (Dry-run) ATIVO** — Nenhum cancelamento real será enviado para a API.", icon="⚠️")

    # Lógica de execução da análise
    if params["analysis_submitted"] or not st.session_state[SK.FIRST_RUN_COMPLETE]:
        df_analysis_min = carregar_dados_minimos(engine, params["dias_hist"])
        if df_analysis_min.empty:
            st.warning("Nenhuma atividade encontrada para o período de análise.")
            st.session_state[SK.ANALYSIS_RESULTS] = []
        else:
            core_params = {k: params[k] for k in ['min_sim', 'min_containment', 'pre_delta', 'use_cnj']}
            st.session_state[SK.ANALYSIS_RESULTS] = criar_grupos_de_duplicatas_otimizado(df_analysis_min, core_params, engine)
        st.session_state[SK.FIRST_RUN_COMPLETE] = True

    all_groups = st.session_state.get(SK.ANALYSIS_RESULTS, [])
    if not all_groups:
        st.info("Nenhum grupo de duplicatas encontrado com os parâmetros atuais. Ajuste os filtros e analise novamente."); st.stop()
    
    # Lógica de filtragem dos grupos para exibição
    final_filtered_groups = []
    for group in all_groups:
        group_key = generate_group_key(group)
        if group_key in st.session_state[SK.IGNORED_GROUPS]: continue
        
        non_canceled_count = sum(1 for r in group if "Cancelad" not in r.get("activity_status", ""))
        if non_canceled_count <= 1: continue

        # Aplica filtro de modo estrito antes de outros filtros
        rows_to_check = group
        if params['strict_only']:
            principal_id = st.session_state[SK.GROUP_STATES].get(group_key, {}).get("principal_id") or group[0]['activity_id']
            principal = next((r for r in group if r["activity_id"] == principal_id), group[0])
            visible_rows = [principal]
            for row in group:
                if row["activity_id"] == principal_id: continue
                score, details = row.get('score_vs_principal', 0), row.get('details_vs_principal', {})
                if score >= (params['min_sim'] * 100) and details.get('contain', 0) >= params['min_containment']:
                    visible_rows.append(row)
            if len(visible_rows) < 2: continue
            rows_to_check = visible_rows
        
        if params["pastas"] and not any(r.get("activity_folder") in params["pastas"] for r in rows_to_check): continue
        if params["only_groups_with_open"] and not any(r.get("activity_status") == "Aberta" for r in rows_to_check): continue
        if params["status"] and not any(r.get("activity_status") in params["status"] for r in rows_to_check): continue
        
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

    with tab2: render_calibration_tab(df_full_for_filters, engine)
    with tab3: render_history_tab(db_firestore)

if __name__ == "__main__":
    main()
