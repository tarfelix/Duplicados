import streamlit as st
import pandas as pd
from src.config import APP_TITLE, SK, DEFAULTS, TZ_SP, TZ_UTC
from src.database.mysql_client import get_mysql_engine, carregar_opcoes_mysql, carregar_dados_mysql
from src.database.firestore import init_firestore, log_action
from src.core.matcher import create_groups, combined_score
from src.components.ui import apply_styles, render_diff
from src.api.client import HttpClientRetry

# --- Initialize ---
st.set_page_config(layout="wide", page_title=APP_TITLE)
apply_styles()

db_firestore = init_firestore()
mysql_engine = get_mysql_engine()

if SK.USERNAME not in st.session_state:
    st.title(APP_TITLE)
    user = st.text_input("Nome do Usuário")
    if user and st.button("Entrar"):
        st.session_state[SK.USERNAME] = user
        st.rerun()
    st.stop()

# --- Sidebar ---
st.sidebar.header(f"👤 {st.session_state[SK.USERNAME]}")
if st.sidebar.button("🚪 Sair"):
    del st.session_state[SK.USERNAME]
    st.rerun()

pastas_opts, status_opts = carregar_opcoes_mysql(mysql_engine)
dias_hist = st.sidebar.number_input("Dias de Histórico", 7, 365, 10)
pastas_sel = st.sidebar.multiselect("Pastas", pastas_opts)
status_sel = st.sidebar.multiselect("Status", status_opts, default=[s for s in status_opts if "Cancelad" not in s])

strict_mode = st.sidebar.toggle("Modo Estrito", value=True)
use_cnj = st.sidebar.toggle("Filtrar por CNJ", value=True)

# --- Data Loading ---
df = carregar_dados_mysql(mysql_engine, dias_hist, pastas_sel, status_sel)

if df.empty:
    st.info("Nenhum dado encontrado para os filtros selecionados.")
else:
    params = {
        'min_sim': st.secrets.get("similarity", {}).get("min_sim_global", 0.9),
        'min_containment': st.secrets.get("similarity", {}).get("min_containment", 55),
        'use_cnj': use_cnj
    }
    
    # Init Group State
    if SK.GROUP_STATES not in st.session_state:
        st.session_state[SK.GROUP_STATES] = {}
    if SK.IGNORED_GROUPS not in st.session_state:
        st.session_state[SK.IGNORED_GROUPS] = set()

    groups = create_groups(df, params)
    
    st.title(f"🔍 {len(groups)} Grupos de Duplicatas")
    
    for i, group in enumerate(groups):
        g_id = group[0]["activity_id"]
        if g_id in st.session_state[SK.IGNORED_GROUPS]: continue
        
        with st.expander(f"Grupo {i+1}: {len(group)} itens | Pasta: {group[0]['activity_folder']}"):
            for row in group:
                st.write(f"ID: {row['activity_id']} | Status: {row['activity_status']}")
                st.text_area("Conteúdo", row["Texto"], height=100, key=f"txt_{row['activity_id']}")
                st.divider()

# Footer
st.sidebar.divider()
st.sidebar.caption(f"v5.1 Modular | Coolify Ready")
