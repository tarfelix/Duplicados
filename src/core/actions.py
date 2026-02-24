import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from src.config import SK

def export_groups_csv(groups: List[List[Dict]]) -> bytes:
    """Gera um arquivo CSV a partir dos grupos de duplicatas."""
    rows = []
    for i, g in enumerate(groups):
        for r in g:
            rows.append({
                "group_index": i + 1,
                "group_size": len(g),
                "activity_id": r.get("activity_id"),
                "activity_folder": r.get("activity_folder"),
                "activity_date": r.get("activity_date"),
                "activity_status": r.get("activity_status"),
                "user_profile_name": r.get("user_profile_name"),
                "Texto": r.get("Texto", "")
            })
    if not rows: return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def process_cancellations(to_cancel_list: List[Dict], user: str, api_client: Any, log_fn: Any):
    """Processa os cancelamentos via API."""
    if not api_client:
        st.error("Cliente de API não configurado.")
        return

    st.info(f"Iniciando o cancelamento de {len(to_cancel_list)} atividades...")
    progress = st.progress(0)
    
    for i, item in enumerate(to_cancel_list):
        act_id = item["ID a Cancelar"]
        principal_id = item["Duplicata do Principal"]
        
        try:
            response = api_client.activity_canceled(activity_id=act_id, user_name=user, principal_id=principal_id)
            if response and (response.get("ok") or response.get("success") or response.get("code") == '200'):
                log_fn(user, "process_cancellation_success", item)
            else:
                log_fn(user, "process_cancellation_failure", {**item, "response": response})
                st.warning(f"Falha ao cancelar {act_id}. Resposta: {response}")
        except Exception as e:
            log_fn(user, "process_cancellation_exception", {**item, "error": str(e)})
            st.error(f"Erro ao cancelar {act_id}: {e}")
        
        progress.progress((i + 1) / len(to_cancel_list))

    st.success("Processamento concluído!")
    if api_client.dry_run:
        st.warning("Atenção: Modo Teste (Dry-run) ativo. Nada foi alterado.")
    
    # Clear states
    st.session_state[SK.GROUP_STATES] = {}
    st.rerun()
