from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth import hash_password, verify_password, create_access_token
from backend.database.postgres import get_db
from backend.database.models import User
from backend.dependencies import get_current_user
from backend.schemas import (
    LoginRequest, TokenResponse, ChangePasswordRequest, MeResponse,
    HasUsersResponse, CreateUserRequest,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/has-users", response_model=HasUsersResponse)
def has_users(db: Session = Depends(get_db)):
    exists = db.query(User).first() is not None
    return HasUsersResponse(has_users=exists)


@router.post("/setup", response_model=TokenResponse)
def setup_first_admin(req: CreateUserRequest, db: Session = Depends(get_db)):
    """Create the first admin user (only works when no users exist)."""
    if db.query(User).first() is not None:
        raise HTTPException(status_code=400, detail="Já existem usuários cadastrados.")

    user = User(
        username=req.username.strip().lower(),
        password_hash=hash_password(req.password),
        role="admin",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(access_token=token, username=user.username, role=user.role)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    username = req.username.strip().lower()
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuário ou senha inválidos.")

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(access_token=token, username=user.username, role=user.role)


@router.get("/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)):
    return MeResponse(username=user.username, role=user.role)


@router.post("/change-password")
def change_password(req: ChangePasswordRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Senha atual incorreta.")

    user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"message": "Senha alterada com sucesso."}
