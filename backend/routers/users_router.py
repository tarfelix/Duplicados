from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.auth import hash_password
from backend.database.postgres import get_db
from backend.database.models import User
from backend.dependencies import require_admin
from backend.schemas import (
    CreateUserRequest, UpdateRoleRequest, AdminPasswordResetRequest, UserResponse,
)

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=List[UserResponse])
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.username).all()
    return [UserResponse(username=u.username, role=u.role, created_at=u.created_at) for u in users]


@router.post("", response_model=UserResponse, status_code=201)
def create_user(req: CreateUserRequest, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    username = req.username.strip().lower()
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Este nome de usuário já existe.")

    user = User(
        username=username,
        password_hash=hash_password(req.password),
        role=req.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(username=user.username, role=user.role, created_at=user.created_at)


@router.patch("/{username}/role")
def update_role(username: str, req: UpdateRoleRequest, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username.lower()).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    user.role = req.role
    db.commit()
    return {"message": "Perfil atualizado."}


@router.patch("/{username}/password")
def reset_password(username: str, req: AdminPasswordResetRequest, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username.lower()).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"message": "Senha alterada com sucesso."}


@router.delete("/{username}")
def delete_user(username: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.username == username.lower()).first()
    if not target:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")

    if target.role == "admin":
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Não é possível excluir o único administrador.")

    if target.username == admin.username:
        raise HTTPException(status_code=400, detail="Não é possível excluir seu próprio usuário.")

    db.delete(target)
    db.commit()
    return {"message": "Usuário excluído."}
