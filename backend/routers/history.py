"""Valuation history and dealer profile endpoints.

These replace direct browser-to-Postgres writes. Previously the React app called
`supabase.from('evaluations').insert(...)` itself, which meant:

  * the row shape was defined in frontend code and had drifted from the schema
    documented in the README (it sent `variant` and `locality` columns that did
    not exist there, so every insert failed);
  * guest sessions wrote the literal string 'guest' into a `uuid` column, which
    also failed, silently, in a `console.warn`;
  * any validation was advisory, because the client chose what to send.

Now the server owns the row shape, the id, and the timestamp, and derives
`user_id` from a verified token rather than trusting the request body.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from backend import db
from backend.auth import AuthenticatedUser, get_current_user

log = logging.getLogger("priceref.history")

router = APIRouter(prefix="/api", tags=["history"])


class _CamelModel(BaseModel):
    """Accepts and emits camelCase, stores snake_case internally.

    The frontend's record objects are camelCase and the database columns are
    snake_case; doing the translation here means neither side has to know about
    the other's convention, and the mapping lives in one place instead of two
    hand-written converter functions in AppContext.jsx.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",  # tolerate extra UI-only fields rather than 422 the request
        # This domain is about vehicle models and ML model variants, so fields
        # named `model` and `model_variant` are unavoidable. Pydantic reserves the
        # `model_` prefix by default and warns on collision; clearing the
        # namespace is the documented way to opt out.
        protected_namespaces=(),
    )


class EvaluationIn(_CamelModel):
    """A valuation the client wants persisted.

    Note what is absent: `id`, `created_at` and `user_id`. All three are assigned
    server-side. Accepting them would let a caller backdate a record, collide
    with an existing id, or attribute a row to another user.
    """

    source: str = Field(default="Single Vehicle", max_length=64)
    brand: str = Field(default="Unknown", max_length=64)
    model: str = Field(default="Unknown", max_length=64)
    variant: str = Field(default="", max_length=128)
    year: int = Field(default=0, ge=0, le=2100)
    fuel: str = Field(default="Unknown", max_length=32)
    transmission: str = Field(default="Unknown", max_length=32)
    city: str = Field(default="Unknown", max_length=64)
    locality: str = Field(default="", max_length=128)
    odometer: int = Field(default=0, ge=0, le=10_000_000)
    fuel_efficiency: float = Field(default=0, ge=0, le=1000)
    owner_count: int = Field(default=1, ge=0, le=20)
    engine_cc: int = Field(default=0, ge=0, le=20_000)
    condition: str = Field(default="Good", max_length=32)
    seller_asking_price: float = Field(default=0, ge=0)
    market_value: float = Field(default=0, ge=0)
    buy_price: float = Field(default=0, ge=0)
    sell_price: float = Field(default=0, ge=0)
    expected_profit: float = 0
    margin_pct: float = 0
    risk_score: float = Field(default=0, ge=0, le=100)
    confidence_score: float = Field(default=0, ge=0, le=100)
    deal_quality_score: float = Field(default=0, ge=0, le=100)
    action: str = Field(default="MANUAL REVIEW", max_length=64)
    urgency_score: float = Field(default=0, ge=0, le=100)
    is_ml_powered: bool = True
    model_variant: str = Field(default="", max_length=64)
    positive_factors: list[Any] = Field(default_factory=list)
    negative_factors: list[Any] = Field(default_factory=list)


class EvaluationOut(EvaluationIn):
    id: str
    created_at: datetime


class HistoryResponse(_CamelModel):
    evaluations: list[EvaluationOut]
    count: int


class DeleteResponse(_CamelModel):
    deleted: int


class ProfileIn(_CamelModel):
    name: str = Field(min_length=1, max_length=120)
    avatar: str = Field(default="U", min_length=1, max_length=8)
    role: str = Field(default="Dealer", max_length=32)


class ProfileOut(ProfileIn):
    id: str
    created_at: datetime | None = None


def _to_db_row(payload: EvaluationIn, user_id: str) -> dict[str, Any]:
    """Build the insert row. Server-owned fields are set here, not copied in."""
    row = payload.model_dump(by_alias=False)
    row["id"] = str(uuid.uuid4())
    row["user_id"] = user_id
    row["created_at"] = datetime.now(UTC).isoformat()
    return row


def _from_db_row(row: dict[str, Any]) -> EvaluationOut:
    return EvaluationOut.model_validate(row)


@router.get("/history", response_model=HistoryResponse)
async def list_history(
    limit: int = Query(default=200, ge=1, le=500),
    user: AuthenticatedUser = Depends(get_current_user),
) -> HistoryResponse:
    """Return the caller's valuation history, newest first."""
    rows = await db.list_evaluations(
        user_id=user.id, access_token=user.access_token, limit=limit
    )

    evaluations: list[EvaluationOut] = []
    for row in rows:
        try:
            evaluations.append(_from_db_row(row))
        except Exception as exc:
            # One unparseable legacy row must not blank out the whole dashboard.
            log.warning("history | skipping unreadable row id=%s: %s", row.get("id"), exc)

    return HistoryResponse(evaluations=evaluations, count=len(evaluations))


@router.post("/history", response_model=EvaluationOut, status_code=status.HTTP_201_CREATED)
async def create_history_entry(
    payload: EvaluationIn,
    user: AuthenticatedUser = Depends(get_current_user),
) -> EvaluationOut:
    """Persist one valuation for the caller."""
    inserted = await db.insert_evaluation(
        row=_to_db_row(payload, user.id), access_token=user.access_token
    )
    return _from_db_row(inserted)


@router.delete("/history", response_model=DeleteResponse)
async def clear_history(
    user: AuthenticatedUser = Depends(get_current_user),
) -> DeleteResponse:
    """Delete all of the caller's valuation history."""
    deleted = await db.delete_evaluations(user_id=user.id, access_token=user.access_token)
    log.info("history | cleared %d rows for user=%s", deleted, user.id)
    return DeleteResponse(deleted=deleted)


@router.delete("/history/{evaluation_id}", response_model=DeleteResponse)
async def delete_history_entry(
    evaluation_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> DeleteResponse:
    """Delete a single valuation."""
    try:
        uuid.UUID(evaluation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation id must be a UUID.",
        ) from None

    deleted = await db.delete_evaluation(
        user_id=user.id, evaluation_id=evaluation_id, access_token=user.access_token
    )
    if deleted == 0:
        # Deliberately identical whether the row belongs to someone else or does
        # not exist, so this cannot be used to probe for valid ids.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    return DeleteResponse(deleted=deleted)


@router.get("/profile", response_model=ProfileOut)
async def read_profile(
    user: AuthenticatedUser = Depends(get_current_user),
) -> ProfileOut:
    """Return the caller's dealer profile, synthesising one if absent.

    A missing row is normal — a user can exist in auth.users before a profile is
    created — so this derives a sensible default from the token's email rather
    than returning 404 and making the client handle it.
    """
    row = await db.get_profile(user_id=user.id, access_token=user.access_token)
    if row is not None:
        return ProfileOut.model_validate(row)

    fallback_name = (user.email or "dealer").split("@")[0]
    return ProfileOut(
        id=user.id,
        name=fallback_name,
        avatar=fallback_name[:2].upper() or "U",
        role="Dealer",
        created_at=None,
    )


@router.put("/profile", response_model=ProfileOut)
async def write_profile(
    payload: ProfileIn,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ProfileOut:
    """Create or update the caller's dealer profile."""
    row = await db.upsert_profile(
        user_id=user.id,
        access_token=user.access_token,
        fields=payload.model_dump(by_alias=False),
    )
    return ProfileOut.model_validate(row)
