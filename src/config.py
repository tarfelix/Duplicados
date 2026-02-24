import streamlit as st
from zoneinfo import ZoneInfo
from typing import Dict

# Constantes Globais
APP_TITLE = "Verificador de Duplicidade Avançado"
TZ_SP = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# Chaves Session State
SUFFIX = "_v5_final"

class SK:
    USERNAME = f"username_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_{SUFFIX}"
    PAGE_NUMBER = f"page_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"
    CFG = f"cfg_{SUFFIX}"
    SHOW_CANCEL_CONFIRM = f"show_cancel_confirm_{SUFFIX}"
    IGNORED_GROUPS = f"ignored_groups_{SUFFIX}"

# Valores Padrão
DEFAULTS = {
    "itens_por_pagina": 10,
    "dias_filtro_inicio": 7,
    "dias_filtro_fim": 14,
    "min_sim_global": 90,
    "min_containment": 55,
    "pre_cutoff_delta": 10,
    "diff_hard_limit": 12000,
}

def get_secret(key_path: str, default=None):
    """Auxiliar para ler secrets com segurança."""
    parts = key_path.split('.')
    val = st.secrets
    try:
        for part in parts:
            val = val[part]
        return val
    except (KeyError, AttributeError):
        return default
