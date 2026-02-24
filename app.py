import streamlit as st
import pandas as pd
from src.config import APP_TITLE, SK, DEFAULTS, TZ_SP, TZ_UTC, get_secret
from src.database.mysql_client import get_mysql_engine, carregar_opcoes_mysql, carregar_dados_mysql
from src.database.firestore import init_firestore, log_action
from src.core.matcher import create_groups, combined_score, get_best_principal_id
from src.components.ui import apply_styles, render_diff, render_group
from src.core.actions import export_groups_csv, process_cancellations
from src.api.client import get_api_client

# --- Initialize ---
st.set_page_config(layout="wide", page_title=APP_TITLE)
apply_styles()

db_firestore = init_firestore()
mysql_engine = get_mysql_engine()

if SK.USERNAME not in st.session_state:
    st.title(APP_TITLE)
    with st.container():
        user = st.text_input("Nome do Usuário")
        pwd = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if user and pwd:
                # Busca a senha para o usuário informado
                valid_pwd = get_secret(f"credentials.usernames.{user}")
                if valid_pwd and str(valid_pwd) == str(pwd):
                    st.session_state[SK.USERNAME] = user
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")
            else:
                st.warning("Por favor, preencha todos os campos.")
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
dry_run = st.sidebar.toggle("Modo Teste (Dry-run)", value=False)

# --- Data Loading ---
df = carregar_dados_mysql(mysql_engine, dias_hist, pastas_sel, status_sel)

if df.empty:
    st.info("Nenhum dado encontrado para os filtros selecionados.")
else:
    params = {
        'min_sim': float(get_secret("similarity.min_sim_global", 0.9)),
        'min_containment': int(get_secret("similarity.min_containment", 55)),
        'use_cnj': use_cnj,
        'diff_limit': int(get_secret("similarity.diff_hard_limit", 12000))
    }
    
    # Init Group State
    if SK.GROUP_STATES not in st.session_state:
        st.session_state[SK.GROUP_STATES] = {}
    if SK.IGNORED_GROUPS not in st.session_state:
        st.session_state[SK.IGNORED_GROUPS] = set()

    groups = create_groups(df, params)
    
    # --- Metrics ---
    st.title(f"🔍 {len(groups)} Grupos de Duplicatas")
    m1, m2, m3 = st.columns(3)
    abertas_total = sum(1 for _, row in df.iterrows() if row.get("activity_status") == "Aberta")
    total_marcados = sum(len(state.get('cancelados', [])) for state in st.session_state[SK.GROUP_STATES].values())
    
    m1.metric("Grupos", len(groups))
    m2.metric("Abertas", abertas_total)
    m3.metric("Marcados", total_marcados)
    
    st.divider()

    @st.dialog("Confirmar Cancelamento")
    def confirm_dialog():
        to_cancel = []
        for g in groups:
            gid = g[0]["activity_id"]
            state = st.session_state[SK.GROUP_STATES].get(gid, {})
            p_id = state.get("principal_id")
            for cid in state.get("cancelados", []):
                to_cancel.append({"ID a Cancelar": cid, "Duplicata do Principal": p_id})
        
        if not to_cancel:
            st.info("Nenhuma atividade marcada.")
            return
            
        st.warning(f"Você está prestes a cancelar **{len(to_cancel)}** atividades.")
        st.table(pd.DataFrame(to_cancel))
        if st.button("Confirmar e Processar", type="primary"):
            client = get_api_client(dry_run=dry_run)
            process_cancellations(to_cancel, st.session_state[SK.USERNAME], client, log_action)

    # --- Group Rendering ---
    for group in groups:
        render_group(group, params, get_best_principal_id, combined_score)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("⬇️ Baixar CSV", data=export_groups_csv(groups), file_name="duplicatas.csv", use_container_width=True)
    with col_b:
        if st.button("🚀 Processar Marcados", type="primary", use_container_width=True):
            confirm_dialog()

# Footer
st.sidebar.divider()
st.sidebar.caption(f"v5.1 Modular | Coolify Ready")
