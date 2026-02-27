import streamlit as st
import pandas as pd
from src.config import APP_TITLE, SK, DEFAULTS, TZ_SP, TZ_UTC, get_secret
from src.database.mysql_client import get_mysql_engine, carregar_opcoes_mysql, carregar_dados_mysql
from src.database.firestore import init_firestore, log_action, get_last_firestore_error
from src.database.users_firestore import (
    authenticate, has_any_user, create_user, list_users, update_user_password,
    update_user_role, delete_user, get_user, verify_password,
)
from src.core.matcher import create_groups, combined_score, get_best_principal_id
from src.components.ui import apply_styles, render_diff, render_group
from src.core.actions import export_groups_csv, process_cancellations
from src.api.client import get_api_client


# --- Initialize ---
st.set_page_config(layout="wide", page_title=APP_TITLE)
apply_styles()

db_firestore = init_firestore()
mysql_engine = get_mysql_engine()

# Sem Firebase não há gestão de usuários
if db_firestore is None:
    st.error("Configure o Firebase (variáveis de ambiente do Coolify) para usar login e gestão de usuários. A coleção **verificador_users** será usada para armazenar usuários e senhas (com hash).")
    err = get_last_firestore_error()
    if err:
        st.code(err, language=None)
    st.caption("Dica: Confira se as variáveis FIREBASE_CREDENTIALS_* estão no serviço correto do Coolify, salve e faça **Redeploy**. Se o erro acima aparecer vazio, faça um novo deploy do repositório e tente de novo.")
    st.stop()

if mysql_engine is None:
    st.error("Configure as credenciais do banco MySQL (variáveis de ambiente ou st.secrets).")
    st.stop()

# --- Primeiro acesso: criar administrador inicial ---
if not has_any_user(db_firestore):
    st.title(APP_TITLE)
    st.subheader("Configuração inicial")
    st.info("Não há usuários ainda. Crie o primeiro usuário (administrador).")
    with st.form("setup_admin"):
        u = st.text_input("Nome do usuário (administrador)")
        p = st.text_input("Senha", type="password")
        p2 = st.text_input("Confirmar senha", type="password")
        if st.form_submit_button("Criar administrador"):
            if not u or not p:
                st.error("Preencha usuário e senha.")
            elif p != p2:
                st.error("As senhas não coincidem.")
            else:
                ok, msg = create_user(db_firestore, u, p, role="admin")
                if ok:
                    st.session_state[SK.USERNAME] = u.strip().lower()
                    st.session_state[SK.USER_ROLE] = "admin"
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    st.stop()

# --- Login ---
if SK.USERNAME not in st.session_state:
    st.title(APP_TITLE)
    with st.container():
        user = st.text_input("Nome do Usuário")
        pwd = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if user and pwd:
                auth = authenticate(db_firestore, user, pwd)
                if auth:
                    st.session_state[SK.USERNAME] = auth["username"]
                    st.session_state[SK.USER_ROLE] = auth.get("role", "user")
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")
            else:
                st.warning("Por favor, preencha todos os campos.")
        st.markdown("---")
        st.caption("Esqueceu a senha? Peça ao administrador para redefinir no painel **Gerenciar usuários**.")
    st.stop()

# --- Sidebar ---
st.sidebar.header(f"👤 {st.session_state[SK.USERNAME]}")
if st.session_state.get(SK.USER_ROLE) == "admin":
    st.sidebar.caption("Administrador")
with st.sidebar:
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Atualizar"):
            carregar_dados_mysql.clear()
            carregar_opcoes_mysql.clear()
            st.rerun()
    with col_sb2:
        if st.button("🚪 Sair"):
            st.session_state.pop(SK.USERNAME, None)
            st.session_state.pop(SK.USER_ROLE, None)
            st.session_state.pop("_page", None)
            st.rerun()

if st.sidebar.button("🔑 Alterar minha senha"):
    st.session_state["_show_change_password"] = True
with st.sidebar.expander("Esqueci minha senha"):
    st.caption("Contate o administrador para redefinir sua senha em **Gerenciar usuários**.")

if st.session_state.get(SK.USER_ROLE) == "admin":
    st.sidebar.markdown("---")
    if st.sidebar.button("👥 Gerenciar usuários"):
        st.session_state["_page"] = "users"
        st.rerun()

# Diálogo Alterar senha
if st.session_state.get("_show_change_password"):
    @st.dialog("Alterar minha senha")
    def change_password_dialog():
        st.caption("Informe sua senha atual e a nova senha.")
        current = st.text_input("Senha atual", type="password")
        new1 = st.text_input("Nova senha", type="password")
        new2 = st.text_input("Confirmar nova senha", type="password")
        if st.button("Salvar"):
            if not current or not new1 or not new2:
                st.error("Preencha todos os campos.")
            elif new1 != new2:
                st.error("A nova senha e a confirmação não coincidem.")
            elif len(new1) < 4:
                st.error("A nova senha deve ter no mínimo 4 caracteres.")
            else:
                user_doc = get_user(db_firestore, st.session_state[SK.USERNAME])
                if not user_doc or not verify_password(current, user_doc.get("password_hash") or ""):
                    st.error("Senha atual incorreta.")
                else:
                    ok, msg = update_user_password(db_firestore, st.session_state[SK.USERNAME], new1)
                    if ok:
                        st.success(msg)
                        st.session_state["_show_change_password"] = False
                        st.rerun()
                    else:
                        st.error(msg)
        if st.button("Cancelar"):
            st.session_state["_show_change_password"] = False
            st.rerun()
    change_password_dialog()

# Página: Gerenciar usuários (só admin)
if st.session_state.get(SK.USER_ROLE) == "admin" and st.session_state.get("_page") == "users":
    st.title("👥 Gerenciar usuários")
    if st.button("← Voltar para Duplicidades"):
        st.session_state["_page"] = "main"
        st.rerun()
    st.markdown("---")
    users = list_users(db_firestore)
    if not users:
        st.info("Nenhum usuário cadastrado.")
    else:
        for u in users:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.text(f"**{u['username']}** — {u.get('role', 'user')}")
            with col2:
                new_role = st.selectbox(
                    "Perfil",
                    ["user", "admin"],
                    index=0 if u.get("role") == "user" else 1,
                    key=f"role_{u['username']}",
                    label_visibility="collapsed"
                )
                if st.button("Aplicar perfil", key=f"apply_role_{u['username']}"):
                    ok, msg = update_user_role(db_firestore, u["username"], new_role)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with col3:
                if st.button("Editar senha", key=f"pw_{u['username']}"):
                    st.session_state["_edit_user"] = u["username"]
            with col4:
                if u["username"] != st.session_state[SK.USERNAME] and u.get("role") != "admin":
                    if st.button("Excluir", key=f"del_{u['username']}"):
                        st.session_state["_delete_user"] = u["username"]
    st.markdown("---")
    st.subheader("Novo usuário")
    with st.form("new_user"):
        new_username = st.text_input("Nome de usuário")
        new_password = st.text_input("Senha", type="password")
        new_role = st.selectbox("Perfil", ["user", "admin"])
        if st.form_submit_button("Criar"):
            ok, msg = create_user(db_firestore, new_username, new_password, role=new_role)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    if st.session_state.get("_edit_user"):
        who = st.session_state["_edit_user"]
        with st.expander(f"Redefinir senha de **{who}**", expanded=True):
            new_p = st.text_input("Nova senha", type="password", key="ep")
            new_p2 = st.text_input("Confirmar nova senha", type="password", key="ep2")
            if st.button("Salvar nova senha"):
                if not new_p or len(new_p) < 4:
                    st.error("Senha deve ter no mínimo 4 caracteres.")
                elif new_p != new_p2:
                    st.error("As senhas não coincidem.")
                else:
                    ok, msg = update_user_password(db_firestore, who, new_p)
                    if ok:
                        st.success(msg)
                        st.session_state.pop("_edit_user", None)
                        st.rerun()
                    else:
                        st.error(msg)
    if st.session_state.get("_delete_user"):
        who = st.session_state["_delete_user"]
        st.warning(f"Confirmar exclusão do usuário **{who}**?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sim, excluir"):
                ok, msg = delete_user(db_firestore, who)
                if ok:
                    st.success(msg)
                    st.session_state.pop("_delete_user", None)
                    st.rerun()
                else:
                    st.error(msg)
        with c2:
            if st.button("Cancelar"):
                st.session_state.pop("_delete_user", None)
                st.rerun()
    st.stop()

pastas_opts, status_opts = carregar_opcoes_mysql(mysql_engine)
dias_hist = st.sidebar.number_input("Dias de Histórico", 7, 365, 10)
pastas_sel = st.sidebar.multiselect("Pastas", pastas_opts)
status_sel = st.sidebar.multiselect("Status", status_opts, default=[s for s in status_opts if "Cancelad" not in s])

strict_mode = st.sidebar.toggle("Modo Estrito", value=True)
use_cnj = st.sidebar.toggle("Filtrar por CNJ", value=True)
dry_run = st.sidebar.toggle("Modo Teste (Dry-run)", value=False)

# --- Data Loading ---
with st.spinner("Carregando atividades..."):
    df = carregar_dados_mysql(mysql_engine, dias_hist, pastas_sel, status_sel)

if df.empty:
    st.info("Nenhum dado encontrado para os filtros selecionados.")
else:
    raw_min_sim = float(get_secret("similarity.min_sim_global", 0.9))
    params = {
        'min_sim': raw_min_sim / 100.0 if raw_min_sim > 1 else raw_min_sim,
        'min_containment': int(get_secret("similarity.min_containment", 55)),
        'use_cnj': use_cnj,
        'diff_limit': int(get_secret("similarity.diff_hard_limit", 12000))
    }
    
    # Init Group State
    if SK.GROUP_STATES not in st.session_state:
        st.session_state[SK.GROUP_STATES] = {}
    if SK.IGNORED_GROUPS not in st.session_state:
        st.session_state[SK.IGNORED_GROUPS] = set()

    with st.spinner("Agrupando duplicatas..."):
        groups = create_groups(df, params)
    ignored = st.session_state.get(SK.IGNORED_GROUPS, set())
    groups = [g for g in groups if g[0]["activity_id"] not in ignored]
    
    # --- Metrics ---
    if dry_run:
        st.warning("Modo Teste (Dry-run) ativo – nenhum cancelamento será enviado à API.")
    
    st.title(f"🔍 {len(groups)} Grupos de Duplicatas")
    m1, m2, m3 = st.columns(3)
    abertas_total = sum(1 for _, row in df.iterrows() if row.get("activity_status") == "Aberta")
    total_marcados = sum(len(state.get('cancelados', [])) for state in st.session_state[SK.GROUP_STATES].values())
    
    m1.metric("Grupos", len(groups))
    m2.metric("Abertas", abertas_total)
    m3.metric("Marcados", total_marcados)
    
    if len(groups) == 0 and not df.empty:
        st.info("Nenhum grupo de duplicatas encontrado com os critérios atuais. Tente ajustar os filtros ou o modo estrito.")
    
    st.divider()

    @st.dialog("Confirmar Cancelamento")
    def confirm_dialog():
        to_cancel = []
        for g in groups:
            gid = g[0]["activity_id"]
            state = st.session_state[SK.GROUP_STATES].get(gid, {})
            p_id = state.get("principal_id")
            if not p_id:
                continue
            principal_row = next((r for r in g if r["activity_id"] == p_id), None)
            if not principal_row:
                continue
            p_norm = principal_row.get("_norm", "")
            p_meta = principal_row.get("_meta", {})
            for cid in state.get("cancelados", []):
                cancel_row = next((r for r in g if r["activity_id"] == cid), None)
                if not cancel_row:
                    continue
                score, _ = combined_score(p_norm, cancel_row.get("_norm", ""), p_meta, cancel_row.get("_meta", {}))
                to_cancel.append({
                    "ID a Cancelar": cid,
                    "Duplicata do Principal": p_id,
                    "Pasta": cancel_row.get("activity_folder", "N/A"),
                    "Similaridade (%)": f"{score:.0f}"
                })
        
        if not to_cancel:
            st.info("Nenhuma atividade marcada.")
            return
            
        st.warning("Atenção: esta ação é irreversível.")
        st.warning(f"Você está prestes a cancelar **{len(to_cancel)}** atividades.")
        st.table(pd.DataFrame(to_cancel))
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Confirmar e Processar", type="primary"):
                client = get_api_client(dry_run=dry_run)
                log_fn = (lambda u, a, d: log_action(db_firestore, u, a, d)) if db_firestore else (lambda u, a, d: None)
                process_cancellations(to_cancel, st.session_state[SK.USERNAME], client, log_fn)
        with col_btn2:
            if st.button("Voltar"):
                st.rerun()

    # --- Group Rendering ---
    for group in groups:
        render_group(group, params, get_best_principal_id, combined_score)

    st.divider()
    st.header("⚡ Ações em massa")
    st.caption("Exporte os grupos para CSV ou processe os cancelamentos marcados.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("⬇️ Baixar CSV", data=export_groups_csv(groups), file_name="duplicatas.csv", use_container_width=True)
    with col_b:
        if st.button("🚀 Processar Marcados", type="primary", use_container_width=True):
            confirm_dialog()

# Footer
st.sidebar.divider()
st.sidebar.caption(f"v5.1 Modular | Coolify Ready")
