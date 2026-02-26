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

# Valores Padrão (min_sim_global em escala 0-1, ex.: 0.9 = 90%)
DEFAULTS = {
    "itens_por_pagina": 10,
    "dias_filtro_inicio": 7,
    "dias_filtro_fim": 14,
    "min_sim_global": 0.9,
    "min_containment": 55,
    "pre_cutoff_delta": 10,
    "diff_hard_limit": 12000,
}

def get_secret(key_path: str, default=None):
    """
    Busca um segredo no Streamlit de forma ultra-resiliente.
    Tenta o formato aninhado, o formato flat e o fallback para variáveis de ambiente (OS).
    Isso evita o erro 'StreamlitSecretNotFoundError' no Coolify/Docker.
    """
    import os
    parts = key_path.split('.')
    flat_key = "_".join(parts).upper()

    # 1. Tentar via Streamlit (Nested ou Flat)
    try:
        # Tenta percorrer o caminho (ex: database -> host)
        val = st.secrets
        for part in parts:
            val = val[part]
        if val is not None:
            return val
    except Exception:
        # Se falhar qualquer acesso ao st.secrets, ignoramos e tentamos o próximo
        pass

    try:
        # Tenta o flat key direto no st.secrets (ex: DATABASE_HOST)
        val_flat = st.secrets.get(flat_key)
        if val_flat is not None:
            return val_flat
    except Exception:
        pass

    # 2. Resgate definitivo via Variáveis de Ambiente do Sistema (Docker/Coolify)
    return os.environ.get(flat_key, default)
