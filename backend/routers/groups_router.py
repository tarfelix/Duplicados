import csv
import io
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional

import pandas as pd

from backend.database.models import User, AuditLog
from backend.database.mysql_client import carregar_dados_mysql
from backend.database.postgres import get_db
from backend.dependencies import get_current_user
from backend.config import get_settings
from backend.services.matcher import (
    create_groups, combined_score, get_best_principal_id, normalize_text, extract_meta,
)
from backend.services.api_client import get_api_client
from backend.services.diff import render_diff
from backend.services.ai_explain import explain_differences, is_ai_available
from backend.schemas import (
    GroupsListResponse, GroupResponse, ActivityItem,
    CancelRequest, CancelResult,
    DiffRequest, DiffResponse,
    ExplainDiffRequest, ExplainDiffResponse,
)

router = APIRouter(prefix="/api/groups", tags=["groups"])


def _build_groups(
    dias: int,
    pastas: Optional[List[str]],
    status: Optional[List[str]],
    min_sim: int,
    min_containment: int,
    use_cnj: bool,
    hide_closed: bool = True,
):
    settings = get_settings()
    df = carregar_dados_mysql(dias, pastas or None, status or None)
    if df.empty:
        return [], 0, df

    params = {
        "min_sim": min_sim / 100.0,
        "min_containment": min_containment,
        "use_cnj": use_cnj,
        "stopwords_extra": settings.stopwords_extra_list,
        "cutoffs_map": settings.cutoffs_map,
        "min_tokens": settings.similarity_min_tokens_to_match,
    }

    groups = create_groups(df, params)

    if hide_closed:
        groups = [g for g in groups if any(r.get("activity_status") == "Aberta" for r in g)]

    # Hide already-resolved groups: if there are cancelled items and only
    # 1 non-cancelled item remains, the duplicate was already handled.
    def _is_resolved(g):
        cancelled = sum(1 for r in g if r.get("activity_status") == "Cancelada")
        non_cancelled = sum(1 for r in g if r.get("activity_status") != "Cancelada")
        return cancelled >= 1 and non_cancelled <= 1

    groups = [g for g in groups if not _is_resolved(g)]

    def sort_key(g):
        open_count = sum(1 for r in g if r.get("activity_status") == "Aberta")
        latest = max((pd.to_datetime(r.get("activity_date"), errors="coerce") for r in g), default=pd.Timestamp.min)
        return (-open_count, -latest.value)

    groups = sorted(groups, key=sort_key)

    total_abertas = int((df["activity_status"] == "Aberta").sum())
    return groups, total_abertas, df


@router.get("", response_model=GroupsListResponse)
def get_groups(
    dias: int = Query(default=10, ge=1, le=365),
    pastas: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    min_sim: int = Query(default=90, ge=0, le=100),
    min_containment: int = Query(default=55, ge=0, le=100),
    use_cnj: bool = Query(default=True),
    hide_closed: bool = Query(default=True),
    user: User = Depends(get_current_user),
):
    pastas_list = [p.strip() for p in pastas.split(",") if p.strip()] if pastas else None
    status_list = [s.strip() for s in status.split(",") if s.strip()] if status else None

    groups_raw, total_abertas, df = _build_groups(dias, pastas_list, status_list, min_sim, min_containment, use_cnj, hide_closed)

    groups_response = []
    for g in groups_raw:
        group_id = str(g[0]["activity_id"])
        best_pid = get_best_principal_id(g, min_sim, min_containment)
        principal = next((r for r in g if r["activity_id"] == best_pid), g[0])
        p_norm = principal.get("_norm", "")
        p_meta = principal.get("_meta", {})

        items = []
        for row in g:
            rid = row["activity_id"]
            is_p = rid == best_pid
            score = None
            score_details = None
            if not is_p:
                score, score_details = combined_score(p_norm, row.get("_norm", ""), p_meta, row.get("_meta", {}))

            dt = pd.to_datetime(row.get("activity_date"), errors="coerce") if row.get("activity_date") is not None else None
            items.append(ActivityItem(
                activity_id=rid,
                activity_folder=row.get("activity_folder"),
                user_profile_name=row.get("user_profile_name"),
                activity_date=dt.isoformat() if dt is not None and pd.notna(dt) else None,
                activity_status=row.get("activity_status"),
                texto=row.get("Texto", ""),
                score=score,
                score_details=score_details,
                is_principal=is_p,
            ))

        open_count = sum(1 for r in g if r.get("activity_status") == "Aberta")
        group_sources = g[0].get("_group_sources", [])
        group_is_retif = g[0].get("_group_is_retificacao", False)
        group_cross_djen = g[0].get("_group_cross_djen", False)
        groups_response.append(GroupResponse(
            group_id=group_id,
            items=items,
            folder=g[0].get("activity_folder"),
            open_count=open_count,
            best_principal_id=best_pid,
            sources=group_sources if group_sources else None,
            is_retificacao=group_is_retif,
            is_cross_djen=group_cross_djen,
        ))

    return GroupsListResponse(
        groups=groups_response,
        total_groups=len(groups_response),
        total_abertas=total_abertas,
    )


@router.post("/cancel", response_model=CancelResult)
def cancel_activities(
    req: CancelRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    client = get_api_client(dry_run=req.dry_run)
    if not client:
        raise HTTPException(status_code=400, detail="API de cancelamento não configurada.")

    results = {"ok": 0, "err": 0}
    details = []

    for item in req.items:
        try:
            response = client.activity_canceled(
                activity_id=item.activity_id,
                user_name=user.username,
                principal_id=item.principal_id,
            )
            success = response and (
                response.get("ok") or response.get("success") or
                str(response.get("code", "")) == "200"
            )
            if success:
                results["ok"] += 1
                action = "process_cancellation_success"
            else:
                results["err"] += 1
                action = "process_cancellation_failure"

            details.append({"activity_id": item.activity_id, "success": bool(success), "response": response})

            log = AuditLog(
                username=user.username,
                action=action,
                details={
                    "activity_id": item.activity_id,
                    "principal_id": item.principal_id,
                    "dry_run": req.dry_run,
                },
            )
            db.add(log)
        except Exception as e:
            logging.exception(f"Error cancelling {item.activity_id}: {e}")
            results["err"] += 1
            details.append({"activity_id": item.activity_id, "success": False, "error": str(e)})

            log = AuditLog(
                username=user.username,
                action="process_cancellation_exception",
                details={"activity_id": item.activity_id, "error": str(e)},
            )
            db.add(log)

    db.commit()
    return CancelResult(ok=results["ok"], err=results["err"], details=details)


def _csv_row_generator(groups_raw):
    """Yield CSV rows as strings for true streaming."""
    output = io.StringIO()
    writer = csv.writer(output)
    header = ["group_index", "group_size", "activity_id", "activity_folder", "activity_date", "activity_status", "user_profile_name", "Texto"]
    writer.writerow(header)
    yield output.getvalue()
    output.seek(0)
    output.truncate(0)

    for i, g in enumerate(groups_raw):
        for r in g:
            writer.writerow([
                i + 1,
                len(g),
                r.get("activity_id", ""),
                r.get("activity_folder", ""),
                r.get("activity_date", ""),
                r.get("activity_status", ""),
                r.get("user_profile_name", ""),
                r.get("Texto", ""),
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)


@router.get("/export-csv")
def export_csv(
    dias: int = Query(default=10, ge=1, le=365),
    pastas: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    min_sim: int = Query(default=90, ge=0, le=100),
    min_containment: int = Query(default=55, ge=0, le=100),
    use_cnj: bool = Query(default=True),
    hide_closed: bool = Query(default=True),
    user: User = Depends(get_current_user),
):
    pastas_list = [p.strip() for p in pastas.split(",") if p.strip()] if pastas else None
    status_list = [s.strip() for s in status.split(",") if s.strip()] if status else None

    groups_raw, _, _ = _build_groups(dias, pastas_list, status_list, min_sim, min_containment, use_cnj, hide_closed)

    return StreamingResponse(
        _csv_row_generator(groups_raw),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=duplicatas.csv"},
    )


@router.post("/diff", response_model=DiffResponse)
def compute_diff(req: DiffRequest, user: User = Depends(get_current_user)):
    try:
        html_a, html_b = render_diff(req.text_a, req.text_b, req.limit)
        return DiffResponse(html_a=html_a, html_b=html_b)
    except Exception:
        logging.exception("Error in /diff endpoint")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar diff.")


@router.post("/explain-diff", response_model=ExplainDiffResponse)
def explain_diff(req: ExplainDiffRequest, user: User = Depends(get_current_user)):
    if not is_ai_available():
        raise HTTPException(status_code=400, detail="IA não configurada.")
    explanation = explain_differences(req.text_a, req.text_b)
    return ExplainDiffResponse(explanation=explanation)
