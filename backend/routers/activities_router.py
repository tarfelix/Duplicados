from fastapi import APIRouter, Depends, Query
from typing import List, Optional

from backend.database.models import User
from backend.database.mysql_client import carregar_opcoes_mysql
from backend.dependencies import get_current_user
from backend.schemas import FiltersResponse

router = APIRouter(prefix="/api/activities", tags=["activities"])


@router.get("/filters", response_model=FiltersResponse)
def get_filters(user: User = Depends(get_current_user)):
    pastas, status = carregar_opcoes_mysql()
    return FiltersResponse(pastas=pastas, status=status)
