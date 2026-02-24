import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import date, timedelta
from typing import List, Tuple, Dict, Any, Optional
import logging

from src.config import get_secret

@st.cache_resource
def get_mysql_engine() -> Optional[Engine]:
    """Cria e armazena em cache a engine de conexão com o MySQL."""
    host = get_secret("database.host")
    user = get_secret("database.user")
    password = get_secret("database.password")
    name = get_secret("database.name")
    
    if not all([host, user, password, name]):
        st.error("Credenciais do banco de dados (MySQL) ausentes. Verifique st.secrets ou variáveis de ambiente.")
        return None
    
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{user}:{password}@{host}/{name}",
            pool_pre_ping=True,
            pool_recycle=3600
        )
        with engine.connect():
            pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e)
        st.error(f"Erro ao conectar no banco de dados (MySQL): {e}")
        return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_opcoes_mysql(_eng: Engine) -> Tuple[List[str], List[str]]:
    """Carrega apenas os filtros distintos de pastas e status."""
    query_pastas = text("SELECT DISTINCT activity_folder FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND activity_folder IS NOT NULL")
    query_status = text("SELECT DISTINCT activity_status FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND activity_status IS NOT NULL")
    
    try:
        with _eng.connect() as conn:
            pastas = [row[0] for row in conn.execute(query_pastas).fetchall()]
            status = [row[0] for row in conn.execute(query_status).fetchall()]
        return sorted(pastas), sorted(status)
    except exc.SQLAlchemyError as e:
        logging.exception(e)
        return [], []

@st.cache_data(ttl=1800, hash_funcs={Engine: lambda _: None})
def carregar_dados_mysql(_eng: Engine, dias_historico: int, pastas: List[str] = None, status: List[str] = None) -> pd.DataFrame:
    """Carrega atividades do banco usando filtros SQL."""
    limite = date.today() - timedelta(days=dias_historico)
    
    base_query = """
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar'
          AND (activity_status='Aberta' OR DATE(activity_date) >= :limite)
    """
    
    params: Dict[str, Any] = {"limite": limite}
    if pastas:
        base_query += " AND activity_folder IN :pastas"
        params["pastas"] = tuple(pastas)
    if status:
        base_query += " AND activity_status IN :status"
        params["status"] = tuple(status)

    try:
        with _eng.connect() as conn:
            df = pd.read_sql(text(base_query), conn, params=params)
        
        if df.empty:
            return pd.DataFrame()
        
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)
        
        # Manter apenas a atividade mais relevante (Aberta > Fechada)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        df = df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
        
        return df
    except exc.SQLAlchemyError as e:
        logging.exception(e)
        st.error(f"Erro ao carregar dados do banco: {e}")
        return pd.DataFrame()
