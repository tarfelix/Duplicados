import logging
from datetime import date, timedelta
from functools import lru_cache
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text, bindparam, exc
from sqlalchemy.engine import Engine

from backend.config import get_settings

_mysql_engine: Optional[Engine] = None


def get_mysql_engine() -> Optional[Engine]:
    global _mysql_engine
    if _mysql_engine is not None:
        return _mysql_engine

    settings = get_settings()
    if not all([settings.database_host, settings.database_user, settings.database_password, settings.database_name]):
        logging.error("MySQL credentials missing")
        return None

    try:
        _mysql_engine = create_engine(
            settings.mysql_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        with _mysql_engine.connect():
            pass
        return _mysql_engine
    except exc.SQLAlchemyError as e:
        logging.exception(f"MySQL connection error: {e}")
        return None


def carregar_opcoes_mysql() -> Tuple[List[str], List[str]]:
    engine = get_mysql_engine()
    if not engine:
        return [], []

    query_pastas = text(
        "SELECT DISTINCT activity_folder FROM ViewGrdAtividadesTarcisio "
        "WHERE activity_type='Verificar' AND activity_folder IS NOT NULL"
    )
    query_status = text(
        "SELECT DISTINCT activity_status FROM ViewGrdAtividadesTarcisio "
        "WHERE activity_type='Verificar' AND activity_status IS NOT NULL"
    )

    try:
        with engine.connect() as conn:
            pastas = [row[0] for row in conn.execute(query_pastas).fetchall()]
            status = [row[0] for row in conn.execute(query_status).fetchall()]
        return sorted(pastas), sorted(status)
    except exc.SQLAlchemyError as e:
        logging.exception(f"Error loading filter options: {e}")
        return [], []


def carregar_dados_mysql(
    dias_historico: int,
    pastas: Optional[List[str]] = None,
    status: Optional[List[str]] = None,
) -> pd.DataFrame:
    engine = get_mysql_engine()
    if not engine:
        return pd.DataFrame()

    limite = date.today() - timedelta(days=dias_historico)

    base_query = """
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar'
          AND (activity_status='Aberta' OR DATE(activity_date) >= :limite)
    """

    params = {"limite": limite}

    if pastas:
        base_query += " AND activity_folder IN :pastas"
        params["pastas"] = list(pastas)
    if status:
        base_query += " AND activity_status IN :status"
        params["status"] = list(status)

    stmt = text(base_query)
    if pastas:
        stmt = stmt.bindparams(bindparam("pastas", expanding=True))
    if status:
        stmt = stmt.bindparams(bindparam("status", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(stmt, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)

        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        df = df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])

        return df
    except exc.SQLAlchemyError as e:
        logging.exception(f"Error loading activities: {e}")
        return pd.DataFrame()
