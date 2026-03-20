"""
Gestão de usuários do Verificador de Duplicidade.
Usuários são armazenados no Firestore (coleção verificador_users).
Senhas são armazenadas com hash seguro usando passlib.
"""
import logging
import re
import concurrent.futures
from typing import Optional, List, Dict, Any

try:
    from passlib.context import CryptContext
except ImportError:
    CryptContext = None

COLLECTION_USERS = "verificador_users"
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
FIRESTORE_TIMEOUT = 15  # segundos


def _run_with_timeout(fn, timeout=FIRESTORE_TIMEOUT):
    """Executa fn() com timeout. Retorna o resultado ou levanta TimeoutError."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        return future.result(timeout=timeout)

PWD_CONTEXT = (
    CryptContext(
        schemes=["pbkdf2_sha256", "bcrypt"],
        default="pbkdf2_sha256",
        deprecated="auto",
    )
    if CryptContext
    else None
)

def _hash_password(plain) -> str:
    if not PWD_CONTEXT:
        raise RuntimeError("Instale passlib para gestão de usuários.")
    plain = str(plain) if plain is not None else ""
    return PWD_CONTEXT.hash(plain)

def verify_password(plain, hashed: str) -> bool:
    if not PWD_CONTEXT or not hashed:
        return False
    try:
        plain = str(plain) if plain is not None else ""
        return PWD_CONTEXT.verify(plain, hashed)
    except Exception:
        return False

def get_user(db, username: str) -> Optional[Dict[str, Any]]:
    """Retorna o documento do usuário ou None."""
    if not db or not username:
        return None
    try:
        doc = _run_with_timeout(
            lambda: db.collection(COLLECTION_USERS).document(username.strip().lower()).get()
        )
        if doc.exists:
            return doc.to_dict()
        return None
    except concurrent.futures.TimeoutError:
        logging.error(f"Timeout ({FIRESTORE_TIMEOUT}s) ao buscar usuário {username} no Firestore")
        return None
    except Exception as e:
        logging.error(f"Erro ao buscar usuário {username}: {e}")
        return None

def list_users(db) -> List[Dict[str, Any]]:
    """Lista todos os usuários (username, role, created_at). Sem expor senha."""
    if not db:
        return []
    try:
        docs = _run_with_timeout(
            lambda: list(db.collection(COLLECTION_USERS).stream())
        )
        return [
            {"username": doc.id, "role": doc.to_dict().get("role", "user"), "created_at": doc.to_dict().get("created_at")}
            for doc in docs
        ]
    except concurrent.futures.TimeoutError:
        logging.error(f"Timeout ({FIRESTORE_TIMEOUT}s) ao listar usuários no Firestore")
        return []
    except Exception as e:
        logging.error(f"Erro ao listar usuários: {e}")
        return []

def create_user(db, username: str, password: str, role: str = "user") -> tuple[bool, str]:
    """
    Cria um usuário. role: 'admin' ou 'user'.
    Retorna (sucesso, mensagem).
    """
    if not db:
        return False, "Firebase não configurado."
    if not PWD_CONTEXT:
        return False, "Dependência passlib não instalada."
    username = username.strip()
    if not username:
        return False, "Nome de usuário obrigatório."
    if not USERNAME_PATTERN.match(username):
        return False, "Nome de usuário só pode conter letras, números, ponto, hífen e underscore."
    if len(username) < 2:
        return False, "Nome de usuário muito curto."
    password = str(password).strip() if password is not None else ""
    if not password or len(password) < 4:
        return False, "Senha deve ter no mínimo 4 caracteres."
    if role not in ("admin", "user"):
        role = "user"
    username_lower = username.lower()
    try:
        ref = db.collection(COLLECTION_USERS).document(username_lower)
        if ref.get().exists:
            return False, "Este nome de usuário já existe."
        from firebase_admin import firestore
        ref.set({
            "username": username_lower,
            "password_hash": _hash_password(password),
            "role": role,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        return True, "Usuário criado com sucesso."
    except Exception as e:
        logging.exception(e)
        return False, f"Erro ao criar usuário: {e}"

def _get_timestamp():
    try:
        from firebase_admin import firestore
        return firestore.SERVER_TIMESTAMP
    except Exception:
        return None

def update_user_password(db, username: str, new_password: str) -> tuple[bool, str]:
    """Altera a senha de um usuário (admin ou o próprio usuário com senha atual verificada antes)."""
    if not db:
        return False, "Firebase não configurado."
    if not PWD_CONTEXT:
        return False, "Dependência passlib não instalada."
    username_lower = username.strip().lower()
    if not new_password or len(new_password) < 4:
        return False, "Nova senha deve ter no mínimo 4 caracteres."
    new_password = str(new_password)
    try:
        ref = db.collection(COLLECTION_USERS).document(username_lower)
        doc = ref.get()
        if not doc.exists:
            return False, "Usuário não encontrado."
        ref.update({
            "password_hash": _hash_password(new_password),
            "updated_at": _get_timestamp(),
        })
        return True, "Senha alterada com sucesso."
    except Exception as e:
        logging.exception(e)
        return False, f"Erro ao alterar senha: {e}"

def update_user_role(db, username: str, role: str) -> tuple[bool, str]:
    """Altera o role de um usuário (apenas admin)."""
    if not db:
        return False, "Firebase não configurado."
    if role not in ("admin", "user"):
        return False, "Role inválido."
    username_lower = username.strip().lower()
    try:
        ref = db.collection(COLLECTION_USERS).document(username_lower)
        if not ref.get().exists:
            return False, "Usuário não encontrado."
        ref.update({"role": role, "updated_at": _get_timestamp()})
        return True, "Perfil atualizado."
    except Exception as e:
        logging.exception(e)
        return False, f"Erro ao atualizar perfil: {e}"

def delete_user(db, username: str) -> tuple[bool, str]:
    """Remove um usuário. Não permite excluir o próprio admin se for o único admin."""
    if not db:
        return False, "Firebase não configurado."
    username_lower = username.strip().lower()
    try:
        ref = db.collection(COLLECTION_USERS).document(username_lower)
        doc = ref.get()
        if not doc.exists:
            return False, "Usuário não encontrado."
        data = doc.to_dict()
        if data.get("role") == "admin":
            admins = [u["username"] for u in list_users(db) if u.get("role") == "admin"]
            if len(admins) <= 1:
                return False, "Não é possível excluir o único administrador."
        ref.delete()
        return True, "Usuário excluído."
    except Exception as e:
        logging.exception(e)
        return False, f"Erro ao excluir usuário: {e}"

def authenticate(db, username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Autentica usuário. Retorna dict com username e role se ok, senão None.
    """
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.get("password_hash") or ""):
        return None
    return {"username": user.get("username", username), "role": user.get("role", "user")}

def has_any_user(db) -> bool:
    """Verifica se já existe pelo menos um usuário (para primeiro acesso)."""
    if not db:
        return False
    try:
        result = _run_with_timeout(
            lambda: len(list(db.collection(COLLECTION_USERS).limit(1).stream())) > 0
        )
        return result
    except concurrent.futures.TimeoutError:
        logging.error(f"Timeout ({FIRESTORE_TIMEOUT}s) ao verificar usuários no Firestore")
        return False
    except Exception:
        return False
