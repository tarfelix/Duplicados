# -*- coding: utf-8 -*-
"""
Ferramenta de Apoio à Distribuição de Atividades 'Verificar'
===================================================================

Este aplicativo foi redesenhado para focar na distribuição inteligente
de atividades do tipo 'Verificar'. O objetivo principal é fornecer contexto
histórico para cada atividade que está atualmente em aberto.

Funcionalidades Principais:
- Login de Usuário: Acesso seguro utilizando credenciais armazenadas no
  Streamlit secrets.
- Visão Focada: Lista todas as atividades com status 'Aberta' ou 'Aguardando'.
- Ordenação Inteligente: Ordena as atividades por responsável e depois por pasta.
- Destaque Visual Preciso: Usa cores de fundo e texto informativo para
  diferenciar alertas de duplicidade e consistência.
- Contexto Histórico: Para cada atividade aberta, exibe todas as outras
  atividades da mesma pasta dentro do período de tempo selecionado.
- Filtros Inteligentes: Os filtros de responsável, pasta e texto se aplicam
  apenas às atividades ativas.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
from typing import Optional
import streamlit.components.v1 as components

# --- Chave de Sessão para Login ---
USERNAME_KEY = "username_distro_app"

# --- Configuração Geral da Página ---
st.set_page_config(
    layout="wide",
    page_title="Apoio à Distribuição de 'Verificar'"
)

# --- CSS Customizado para Cores de Fundo e Layout Compacto ---
st.markdown("""
<style>
    /* O marcador em si é invisível, serve apenas para o script encontrar. */
    .activity-item-marker {
        display: none;
    }

    /* Adiciona um espaço abaixo de cada expander para separá-los. */
    div[data-testid="stExpander"] {
        margin-bottom: 8px !important;
    }

    /* --- Classes de Cor que serão aplicadas pelo JavaScript --- */
    .alert-red-header {
        background-color: #ffcdd2 !important;
    }

    .alert-black-header {
        background-color: #BDBDBD !important;
    }
    .alert-black-header p { /* Garante que o texto seja branco no fundo escuro */
        color: white !important;
    }

    .alert-gray-header {
        background-color: #f5f5f5 !important;
    }

    /* --- Estilos da Legenda (sem alterações) --- */
    .legenda { display: flex; align-items: center; margin-bottom: 1rem; }
    .cor-box { width: 20px; height: 20px; margin-right: 10px; border: 1px solid #ccc; }
    .vermelho { background-color: #ffcdd2; }
    .preto { background-color: #BDBDBD; }
    .cinza { background-color: #f5f5f5; }
</style>
""", unsafe_allow_html=True)


st.title("Apoio à Distribuição de Atividades 'Verificar'")

# --- Conexão com o Banco de Dados ---
@st.cache_resource
def db_engine_mysql() -> Optional[Engine]:
    """
    Cria e gerencia a conexão com o banco de dados MySQL usando SQLAlchemy.
    """
    try:
        cfg = st.secrets.get("database", {})
        db_user, db_password, db_host, db_name = cfg.get("user"), cfg.get("password"), cfg.get("host"), cfg.get("name")
        if not all([db_user, db_password, db_host, db_name]):
            st.error("As credenciais do banco de dados (MySQL) não foram configuradas nos segredos.")
            return None
        connection_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
        engine = create_engine(connection_url, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        st.error(f"Ocorreu um erro ao conectar ao banco de dados (MySQL): {e}")
        return None

# --- Carregamento de Dados ---
@st.cache_data(ttl=300) # Cache de 5 minutos
def carregar_dados_contextuais(_eng: Engine, data_inicio: datetime.date, data_fim: datetime.date) -> pd.DataFrame:
    """
    Carrega dados de forma contextual, incluindo status 'Aguardando'.
    """
    if _eng is None: return pd.DataFrame()
    start_datetime = datetime.combine(data_inicio, datetime.min.time())
    end_datetime = datetime.combine(data_fim, datetime.max.time())
    
    active_statuses = ('Aberta', 'Aguardando')

    query = text(f"""
        WITH PastasAtivas AS (
            SELECT DISTINCT activity_folder
            FROM ViewGrdAtividadesTarcisio
            WHERE activity_type = 'Verificar' AND activity_status IN {active_statuses}
        )
        SELECT 
            v.activity_id, v.activity_folder, v.user_profile_name, 
            v.activity_date, v.activity_status, v.Texto
        FROM ViewGrdAtividadesTarcisio v
        JOIN PastasAtivas p ON v.activity_folder = p.activity_folder
        WHERE 
            v.activity_type = 'Verificar' 
            AND (
                v.activity_status IN {active_statuses} OR
                v.activity_date BETWEEN :start_datetime AND :end_datetime
            )
    """)
    try:
        with _eng.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_datetime": start_datetime, "end_datetime": end_datetime})
        if not df.empty:
            df["activity_id"] = df["activity_id"].astype(str)
            df["activity_date"] = pd.to_datetime(df["activity_date"], errors='coerce')
            df["Texto"] = df["Texto"].fillna("").astype(str)
        return df.sort_values("activity_date", ascending=False)
    except exc.SQLAlchemyError as e:
        st.error(f"Erro ao executar a consulta no banco de dados: {e}")
        return pd.DataFrame()

# --- Interface Principal ---
def main():
    if USERNAME_KEY not in st.session_state:
        st.session_state[USERNAME_KEY] = None

    if not st.session_state.get(USERNAME_KEY):
        st.sidebar.header("🔐 Login")
        with st.sidebar.form("login_form"):
            username = st.text_input("Nome de Usuário")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar")
            if submitted:
                creds = st.secrets.get("credentials", {})
                user_creds = creds.get("usernames", {})
                if username in user_creds and user_creds[username] == password:
                    st.session_state[USERNAME_KEY] = username
                    st.rerun()
                else:
                    st.sidebar.error("Usuário ou senha inválidos.")
        st.info("👋 Bem-vindo! Por favor, faça o login na barra lateral para continuar.")
        st.stop()

    st.sidebar.success(f"Logado como: **{st.session_state[USERNAME_KEY]}**")
    st.sidebar.header("🔍 Filtros da Consulta")

    data_fim_padrao = datetime.now().date()
    data_inicio_padrao = data_fim_padrao - timedelta(days=10)
    
    st.sidebar.info("O filtro de data define o período para buscar o **histórico de contexto** das atividades.")
    data_inicio = st.sidebar.date_input("📅 Início do Histórico", value=data_inicio_padrao)
    data_fim = st.sidebar.date_input("📅 Fim do Histórico", value=data_fim_padrao)

    if data_inicio > data_fim:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()

    if st.sidebar.button("🔄 Recarregar Dados", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache limpo! Os dados serão recarregados.")
        st.rerun()

    engine = db_engine_mysql()
    if engine is None: st.stop()
    
    with st.spinner("Carregando dados das atividades... Por favor, aguarde."):
        df_contexto_total = carregar_dados_contextuais(engine, data_inicio, data_fim)

    if df_contexto_total.empty:
        st.info("Nenhuma atividade 'Aberta' ou 'Aguardando' foi encontrada, ou não há histórico para elas no período selecionado.")
        st.stop()

    active_statuses = ['Aberta', 'Aguardando']
    df_ativas = df_contexto_total[df_contexto_total['activity_status'].isin(active_statuses)].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Filtrar Atividades Ativas")

    lista_pastas = sorted(df_ativas['activity_folder'].dropna().unique().tolist())
    pastas_selecionadas = st.sidebar.multiselect("📁 Pastas", options=lista_pastas)

    lista_responsaveis = sorted(df_ativas['user_profile_name'].dropna().unique().tolist())
    usuarios_selecionados = st.sidebar.multiselect("👤 Responsáveis", options=lista_responsaveis)
    
    texto_busca = st.sidebar.text_input("📝 Buscar no Texto")

    df_ativas_filtrado = df_ativas
    if pastas_selecionadas:
        df_ativas_filtrado = df_ativas_filtrado[df_ativas_filtrado['activity_folder'].isin(pastas_selecionadas)]
    if usuarios_selecionados:
        df_ativas_filtrado = df_ativas_filtrado[df_ativas_filtrado['user_profile_name'].isin(usuarios_selecionados)]
    if texto_busca:
        df_ativas_filtrado = df_ativas_filtrado[df_ativas_filtrado['Texto'].str.contains(texto_busca, case=False, na=False)]

    if not df_ativas_filtrado.empty:
        df_ativas_filtrado = df_ativas_filtrado.sort_values(
            by=['user_profile_name', 'activity_folder', 'activity_date'], 
            ascending=[True, True, False]
        )

    st.metric("Total de Atividades Ativas (após filtros)", len(df_ativas_filtrado))
    
    st.markdown("""
        <div class="legenda">
            <div class="cor-box vermelho"></div><span><b>Alerta Crítico (Vermelho):</b> A mesma pessoa tem mais de uma atividade ativa na mesma pasta.</span>
        </div>
        <div class="legenda">
            <div class="cor-box preto"></div><span><b>Alerta de Consistência (Preto):</b> Pessoas diferentes têm atividades ativas na mesma pasta.</span>
        </div>
        <div class="legenda">
            <div class="cor-box cinza"></div><span><b>Normal (Cinza):</b> Apenas uma atividade ativa nesta pasta.</span>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"Exibindo atividades ativas ('Aberta' ou 'Aguardando') e seu histórico de contexto.")
    st.markdown("---")

    for _, atividade_atual in df_ativas_filtrado.iterrows():
        conflitos_df = df_ativas[
            (df_ativas['activity_folder'] == atividade_atual['activity_folder']) &
            (df_ativas['activity_id'] != atividade_atual['activity_id'])
        ]

        classe_css = 'alert-gray'
        info_conflito = ""
        
        if not conflitos_df.empty:
            conflito_mesmo_resp = conflitos_df[conflitos_df['user_profile_name'] == atividade_atual['user_profile_name']]
            if not conflito_mesmo_resp.empty:
                classe_css = 'alert-red'
                outro = conflito_mesmo_resp.iloc[0]
                info_conflito = f" (Conflito com ID {outro['activity_id']} | Status: {outro['activity_status']})"
            else:
                classe_css = 'alert-black'
                outro = conflitos_df.iloc[0]
                info_conflito = f" (Conflito com ID {outro['activity_id']} | Resp: {outro['user_profile_name']})"

        expander_title = (
            f"ID: {atividade_atual['activity_id']} | Pasta: {atividade_atual['activity_folder']} | "
            f"Responsável: {atividade_atual['user_profile_name']} | Status: {atividade_atual['activity_status']}{info_conflito}"
        )
        
        # Colocamos o marcador invisível...
        st.markdown(f'<div class="activity-item-marker {classe_css}"></div>', unsafe_allow_html=True)
        
        # ...e o expander logo em seguida.
        with st.expander(expander_title, expanded=False):
            st.text_area("Conteúdo", atividade_atual['Texto'], key=f"texto_{atividade_atual['activity_id']}", height=150, disabled=True)
            st.subheader(f"Histórico da Pasta '{atividade_atual['activity_folder']}' no Período")
            df_historico_pasta = df_contexto_total[df_contexto_total['activity_folder'] == atividade_atual['activity_folder']]
            st.dataframe(df_historico_pasta, use_container_width=True, hide_index=True,
                column_config={
                    "activity_id": "ID", "activity_folder": None, "user_profile_name": "Responsável",
                    "activity_date": st.column_config.DatetimeColumn("Data", format="DD/MM/YYYY HH:mm"),
                    "activity_status": "Status", "Texto": None
                })

    # --- SCRIPT INJECTION ---
    # REVISÃO 10: Solução final, estável e segura.
    # Este script espera a renderização do Streamlit terminar e aplica as cores uma única vez.
    js_script = """
    <script>
    const applyColors = () => {
        const markers = document.querySelectorAll('.activity-item-marker');
        const expanderHeaders = document.querySelectorAll('[data-testid="stExpander"] > div:first-child');

        if (markers.length === 0 || expanderHeaders.length === 0 || markers.length !== expanderHeaders.length) {
            return false; // Indica que não foi bem-sucedido
        }

        markers.forEach((marker, index) => {
            const header = expanderHeaders[index];
            if (!header) return;

            header.classList.remove('alert-red-header', 'alert-black-header', 'alert-gray-header');

            let colorClass = '';
            if (marker.classList.contains('alert-red')) {
                colorClass = 'alert-red-header';
            } else if (marker.classList.contains('alert-black')) {
                colorClass = 'alert-black-header';
            } else if (marker.classList.contains('alert-gray')) {
                colorClass = 'alert-gray-header';
            }
            
            if (colorClass) {
                header.classList.add(colorClass);
            }
        });
        return true; // Indica que foi bem-sucedido
    }

    // A abordagem mais segura: um verificador que tenta aplicar as cores
    // e para assim que consegue, evitando sobrecarga.
    const runWhenReady = () => {
        const intervalId = setInterval(() => {
            // Tenta aplicar as cores. Se for bem-sucedido, a função retorna true.
            if (applyColors()) {
                // Para o verificador assim que as cores forem aplicadas.
                clearInterval(intervalId);
            }
        }, 250); // Tenta a cada 250ms

        // Como segurança, para o verificador após 5 segundos, independentemente do resultado.
        setTimeout(() => {
            clearInterval(intervalId);
        }, 5000);
    };

    // Roda a função principal quando o iframe do Streamlit carregar.
    window.addEventListener('load', runWhenReady);
    </script>
    """
    components.html(js_script, height=0, width=0)

if __name__ == "__main__":
    main()
