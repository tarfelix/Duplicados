import streamlit as st
import logging
from typing import Dict
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None

@st.cache_resource
def init_firestore():
    """Inicializa a conexão com o Firebase."""
    if not firebase_admin:
        return None
    
    try:
        if not firebase_admin._apps:
            creds_config = st.secrets.get("firebase_credentials")
            if not creds_config or 'project_id' not in creds_config:
                return None
            
            creds_dict = dict(creds_config)
            if 'private_key' in creds_dict:
                creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        
        return firestore.client()
    except Exception as e:
        logging.error(f"Falha ao conectar no Firebase: {e}")
        return None

def log_action(db, user: str, action: str, details: Dict):
    """Registra uma ação no Firestore."""
    if db is None:
        return
    
    try:
        doc_ref = db.collection("duplicidade_actions").document()
        log_entry = {
            "ts": firestore.SERVER_TIMESTAMP,
            "user": user,
            "action": action,
            "details": details
        }
        doc_ref.set(log_entry)
    except Exception as e:
        logging.error(f"Erro ao registrar ação no Firestore: {e}")
