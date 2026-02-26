import streamlit as st
import logging
from typing import Dict
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None

from src.config import get_secret

@st.cache_resource
def init_firestore():
    """Inicializa a conexão com o Firebase."""
    if not firebase_admin:
        return None
    
    try:
        if not firebase_admin._apps:
            # Tenta pegar o dicionário completo (secrets.toml)
            creds_dict = get_secret("firebase_credentials")
            
            # Se não for um dict completo (comum em env vars), tenta montar a partir de chaves individuais
            if not isinstance(creds_dict, dict) or 'project_id' not in creds_dict:
                creds_dict = {
                    "type": get_secret("firebase_credentials.type", "service_account"),
                    "project_id": get_secret("firebase_credentials.project_id"),
                    "private_key_id": get_secret("firebase_credentials.private_key_id"),
                    "private_key": get_secret("firebase_credentials.private_key"),
                    "client_email": get_secret("firebase_credentials.client_email"),
                    "client_id": get_secret("firebase_credentials.client_id"),
                    "auth_uri": get_secret("firebase_credentials.auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                    "token_uri": get_secret("firebase_credentials.token_uri", "https://oauth2.googleapis.com/token"),
                    "auth_provider_x509_cert_url": get_secret("firebase_credentials.auth_provider_x509_cert_url"),
                    "client_x509_cert_url": get_secret("firebase_credentials.client_x509_cert_url"),
                    "universe_domain": get_secret("firebase_credentials.universe_domain", "googleapis.com")
                }

            if not creds_dict.get('project_id') or not creds_dict.get('private_key'):
                return None
            
            # Corrige a private_key (escapes de \n)
            if 'private_key' in creds_dict and isinstance(creds_dict['private_key'], str):
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
        try:
            st.toast(f"⚠️ Erro ao salvar log de auditoria: {e}", icon="🔥")
        except Exception:
            pass
