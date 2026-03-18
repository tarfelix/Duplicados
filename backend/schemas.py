from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# --- Auth ---
class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=200)


class MeResponse(BaseModel):
    username: str
    role: str


# --- Users ---
class CreateUserRequest(BaseModel):
    username: str = Field(min_length=2, max_length=100, pattern=r"^[a-zA-Z0-9_.\-]+$")
    password: str = Field(min_length=8, max_length=200)
    role: str = Field(default="user", pattern=r"^(admin|user)$")


class UpdateRoleRequest(BaseModel):
    role: str = Field(pattern=r"^(admin|user)$")


class AdminPasswordResetRequest(BaseModel):
    new_password: str = Field(min_length=8, max_length=200)


class UserResponse(BaseModel):
    username: str
    role: str
    created_at: Optional[datetime] = None


class HasUsersResponse(BaseModel):
    has_users: bool


# --- Activities ---
class FiltersResponse(BaseModel):
    pastas: List[str]
    status: List[str]


# --- Groups ---
class GroupParams(BaseModel):
    dias: int = Field(default=10, ge=1, le=365)
    pastas: Optional[List[str]] = None
    status: Optional[List[str]] = None
    min_sim: int = Field(default=90, ge=0, le=100)
    min_containment: int = Field(default=55, ge=0, le=100)
    use_cnj: bool = True


class ActivityItem(BaseModel):
    activity_id: str
    activity_folder: Optional[str] = None
    user_profile_name: Optional[str] = None
    activity_date: Optional[str] = None
    activity_status: Optional[str] = None
    texto: str = ""
    score: Optional[float] = None
    score_details: Optional[Dict[str, Any]] = None
    is_principal: bool = False


class GroupResponse(BaseModel):
    group_id: str
    items: List[ActivityItem]
    folder: Optional[str] = None
    open_count: int = 0
    best_principal_id: str = ""
    sources: Optional[List[str]] = None
    is_retificacao: bool = False
    is_cross_djen: bool = False


class GroupsListResponse(BaseModel):
    groups: List[GroupResponse]
    total_groups: int
    total_abertas: int


class CancelItem(BaseModel):
    activity_id: str
    principal_id: str


class CancelRequest(BaseModel):
    items: List[CancelItem]
    dry_run: bool = False


class CancelResult(BaseModel):
    ok: int
    err: int
    details: List[Dict[str, Any]]


# --- Diff ---
class DiffRequest(BaseModel):
    text_a: str
    text_b: str
    limit: int = Field(default=12000, le=50000)


class DiffResponse(BaseModel):
    html_a: str
    html_b: str


class ExplainDiffRequest(BaseModel):
    text_a: str
    text_b: str


class ExplainDiffResponse(BaseModel):
    explanation: Optional[str] = None
