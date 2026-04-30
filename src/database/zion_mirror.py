"""
zion_mirror.py — Cliente HTTP para a API sp-zion-mirror.

Substitui as 2 queries SQL diretas no Zion MySQL por chamadas GET /activities
ao mirror local. Quando ZION_MIRROR_ENABLED=true (env), usa mirror primeiro
e cai pro MySQL em qualquer erro.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import List, Optional, Tuple

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

MIRROR_URL = os.getenv("ZION_MIRROR_API_URL", "").rstrip("/")
MIRROR_TOKEN = os.getenv("ZION_MIRROR_API_TOKEN", "")
MIRROR_ENABLED = os.getenv("ZION_MIRROR_ENABLED", "false").lower() in ("true", "1", "yes")
MIRROR_TIMEOUT = float(os.getenv("ZION_MIRROR_TIMEOUT_SECONDS", "30"))


def is_enabled() -> bool:
    return bool(MIRROR_ENABLED and MIRROR_URL and MIRROR_TOKEN)


def _list_all(params: dict, page_size: int = 5000) -> list[dict]:
    """Pagina /activities com ?with_total=false (mirror salta SELECT count(*)
    e fica 6-9x mais rapido em queries amplas). Para quando vier pagina < page_size.
    """
    items: list[dict] = []
    offset = 0
    while True:
        p = {**params, "limit": page_size, "offset": offset, "with_total": "false"}
        r = httpx.get(
            f"{MIRROR_URL}/activities",
            params=p,
            headers={"Authorization": f"Bearer {MIRROR_TOKEN}"},
            timeout=MIRROR_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        page = data.get("items", [])
        items.extend(page)
        if len(page) < page_size:
            break
        total = data.get("total")
        if total is not None and len(items) >= total:
            break
        offset += page_size
        if offset > 100000:
            logger.warning("[mirror] paginacao cortada em 100k")
            break
    return items


def carregar_opcoes_mirror() -> Optional[Tuple[List[str], List[str]]]:
    """
    Substitui SELECT DISTINCT activity_folder e activity_status WHERE type='Verificar'.

    Mirror não tem endpoint DISTINCT — pegamos amostra grande e fazemos DISTINCT
    no cliente. Para 'Verificar' temos ~100k linhas, paginamos.
    """
    if not is_enabled():
        return None
    try:
        items = _list_all({"type": "Verificar", "limit": 5000})
    except Exception as e:
        logger.warning(f"[mirror] carregar_opcoes falhou: {type(e).__name__}: {e}")
        return None

    pastas: set = set()
    status: set = set()
    for it in items:
        folder = it.get("activity_folder")
        if folder:
            pastas.add(folder)
        st = it.get("activity_status")
        if st:
            status.add(st)
    return sorted(pastas), sorted(status)


def carregar_dados_mirror(
    dias_historico: int,
    pastas: List[str] = None,
    status: List[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Substitui SELECT id, folder, user, date, status, Texto WHERE type='Verificar'
    AND (status='Aberta' OR date >= :limite) [AND folder IN ...] [AND status IN ...].

    Mirror tem date_from mas não suporta OR composto. Estratégia: 2 chamadas paralelas:
    1. status=Aberta + type=Verificar (todos abertos, sem filtro data)
    2. type=Verificar + date_from=:limite + status_in=... ou todos exceto Aberta
    Depois deduplicar e aplicar filtros pastas/status localmente.

    Para simplicidade: 1 call grande pegando todos com date_from=limite (cobre tudo
    aberto + tudo recente até :limite) e filtro local.
    """
    if not is_enabled():
        return None
    limite = date.today() - timedelta(days=dias_historico)
    try:
        # Buscar tudo type=Verificar dos últimos N dias OR Aberta (sem date filter)
        # Estratégia: 2 calls — Aberta (sem date) + finalizadas com date >= limite
        items_aberta = _list_all({"type": "Verificar", "status": "Aberta"})
        items_recente = _list_all({
            "type": "Verificar",
            "date_from": str(limite),
        })
        # Dedup por activity_id
        seen: set = set()
        items: list[dict] = []
        for arr in (items_aberta, items_recente):
            for it in arr:
                aid = it.get("activity_id")
                if aid in seen:
                    continue
                seen.add(aid)
                items.append(it)
    except Exception as e:
        logger.warning(f"[mirror] carregar_dados falhou: {type(e).__name__}: {e}")
        return None

    if not items:
        return pd.DataFrame()

    # Filtros locais por pastas e status
    if pastas:
        pastas_set = set(pastas)
        items = [it for it in items if it.get("activity_folder") in pastas_set]
    if status:
        status_set = set(status)
        items = [it for it in items if it.get("activity_status") in status_set]

    if not items:
        return pd.DataFrame()

    # Construir DataFrame com mesmas colunas do SQL original
    rows = []
    for it in items:
        rows.append({
            "activity_id": str(it.get("activity_id", "")),
            "activity_folder": it.get("activity_folder", ""),
            "user_profile_name": it.get("user_profile_name", ""),
            "activity_date": it.get("activity_date"),
            "activity_status": it.get("activity_status", ""),
            "Texto": it.get("texto") or "",  # mirror retorna minúsculo
        })
    df = pd.DataFrame(rows)
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df["Texto"] = df["Texto"].fillna("").astype(str)

    # Mesmo dedup e ordenação do SQL original
    df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
    df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
    df = df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    return df
