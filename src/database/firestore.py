import streamlit as st
import json
import logging
import os
from typing import Dict, Optional
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None

from src.config import get_secret

_last_firestore_error: Optional[str] = None

def get_last_firestore_error() -> Optional[str]:
    """Retorna a última mensagem de erro do Firebase (para exibir na UI)."""
    return _last_firestore_error

def _firebase_creds_from_env() -> Optional[dict]:
    """Monta o dict de credenciais a partir das variáveis de ambiente (Coolify/Docker)."""
    project_id = os.environ.get("FIREBASE_CREDENTIALS_PROJECT_ID") or get_secret("firebase_credentials.project_id")
    private_key = os.environ.get("FIREBASE_CREDENTIALS_PRIVATE_KEY") or get_secret("firebase_credentials.private_key")
    if not project_id or not private_key:
        return None
    return {
        "type": os.environ.get("FIREBASE_CREDENTIALS_TYPE") or get_secret("firebase_credentials.type", "service_account"),
        "project_id": project_id,
        "private_key_id": os.environ.get("FIREBASE_CREDENTIALS_PRIVATE_KEY_ID") or get_secret("firebase_credentials.private_key_id"),
        "private_key": private_key,
        "client_email": os.environ.get("FIREBASE_CREDENTIALS_CLIENT_EMAIL") or get_secret("firebase_credentials.client_email"),
        "client_id": os.environ.get("FIREBASE_CREDENTIALS_CLIENT_ID") or get_secret("firebase_credentials.client_id"),
        "auth_uri": os.environ.get("FIREBASE_CREDENTIALS_AUTH_URI") or get_secret("firebase_credentials.auth_uri", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.environ.get("FIREBASE_CREDENTIALS_TOKEN_URI") or get_secret("firebase_credentials.token_uri", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.environ.get("FIREBASE_CREDENTIALS_AUTH_PROVIDER_X509_CERT_URL") or get_secret("firebase_credentials.auth_provider_x509_cert_url"),
        "client_x509_cert_url": os.environ.get("FIREBASE_CREDENTIALS_CLIENT_X509_CERT_URL") or get_secret("firebase_credentials.client_x509_cert_url"),
        "universe_domain": os.environ.get("FIREBASE_CREDENTIALS_UNIVERSE_DOMAIN") or get_secret("firebase_credentials.universe_domain", "googleapis.com"),
    }

@st.cache_resource
def init_firestore():
    """Inicializa a conexão com o Firebase."""
    global _last_firestore_error
    _last_firestore_error = None
    if not firebase_admin:
        _last_firestore_error = "Biblioteca firebase-admin não instalada. Adicione ao requirements.txt."
        return None
    
    try:
        if not firebase_admin._apps:
            # 1) Prioridade: variáveis de ambiente (Coolify/Docker)
            creds_dict = _firebase_creds_from_env()
            
            # 2) Fallback: get_secret (secrets.toml ou JSON em uma variável)
            if not creds_dict:
                raw = get_secret("firebase_credentials")
                if isinstance(raw, str):
                    try:
                        creds_dict = json.loads(raw)
                    except json.JSONDecodeError as e:
                        _last_firestore_error = f"FIREBASE_CREDENTIALS (JSON) inválido: {e}"
                        logging.error(_last_firestore_error)
                        return None
                if not isinstance(creds_dict, dict) or not creds_dict.get('project_id'):
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

            if not creds_dict or not creds_dict.get('project_id') or not creds_dict.get('private_key'):
                _last_firestore_error = (
                    "Variáveis do Firebase não encontradas. No Coolify, confira se FIREBASE_CREDENTIALS_PROJECT_ID e "
                    "FIREBASE_CREDENTIALS_PRIVATE_KEY estão definidas (e se o serviço foi redeployado após salvar)."
                )
                return None
            
            # Corrige a private_key (escapes de \n)
            if isinstance(creds_dict.get('private_key'), str):
                creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        
        return firestore.client()
    except Exception as e:
        _last_firestore_error = str(e)
        logging.error(f"Falha ao conectar no Firebase: {e}", exc_info=True)
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
