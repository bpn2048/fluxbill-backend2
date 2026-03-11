from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Type
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from db import get_session
from models import AppSetting, Customer, Invoice, Subscription


router = APIRouter(prefix="/api", tags=["billing"])
ID_GENERATION_MAX_RETRIES = 6


class InvoiceCreate(BaseModel):
  customer_code: Optional[str] = None
  customer_id: Optional[str] = None
  customer: Optional[str] = None
  customer_name: Optional[str] = None
  amount: int = Field(gt=0)
  currency: str = "INR"
  status: str = "draft"
  due: Optional[date] = None
  method: str = "-"


class InvoiceUpdate(BaseModel):
  customer: Optional[str] = None
  amount: Optional[int] = Field(default=None, ge=0)
  currency: Optional[str] = None
  status: Optional[str] = None
  due: Optional[date] = None
  method: Optional[str] = None


class CustomerCreate(BaseModel):
  id: Optional[str] = None
  name: str
  tier: str = "SMB"
  invoices: int = 0
  status: str = "new"


class CustomerUpdate(BaseModel):
  name: Optional[str] = None
  tier: Optional[str] = None
  invoices: Optional[int] = Field(default=None, ge=0)
  status: Optional[str] = None


class SubscriptionCreate(BaseModel):
  id: Optional[str] = None
  plan: str = "Starter"
  customer: Optional[str] = None
  customer_code: Optional[str] = None
  customer_id: Optional[str] = None
  customer_name: Optional[str] = None
  mrr: int = Field(default=0, ge=0)
  status: str = "active"


class SubscriptionUpdate(BaseModel):
  plan: Optional[str] = None
  customer: Optional[str] = None
  customer_code: Optional[str] = None
  customer_id: Optional[str] = None
  customer_name: Optional[str] = None
  mrr: Optional[int] = Field(default=None, ge=0)
  status: Optional[str] = None


class SettingsUpdate(BaseModel):
  company_name: Optional[str] = None
  invoice_prefix: Optional[str] = Field(default=None, min_length=1, max_length=12)


def _match(q: str, *values: str) -> bool:
  ql = q.strip().lower()
  return any(ql in (v or "").lower() for v in values)


def _score_value(q: str, value: Any) -> int:
  ql = q.strip().lower()
  if not ql:
    return 0

  text = str(value or "").strip().lower()
  if not text:
    return 0

  if text == ql:
    return 120
  if text.startswith(ql):
    return 90
  if ql in text:
    return 60
  return 0


def _score_match(q: str, *values: Any) -> int:
  return max((_score_value(q, v) for v in values), default=0)


_TAB_KEYWORDS = {
  "invoices": ("invoice", "invoices", "inv"),
  "subscriptions": ("subscription", "subscriptions", "sub"),
  "customers": ("customer", "customers", "cust"),
}

_TAB_ALIAS_TEXT = {
  tab: " ".join(keywords) for tab, keywords in _TAB_KEYWORDS.items()
}


def _extract_entity_hints(q: str) -> Set[str]:
  lowered = str(q or "").strip().lower()
  if not lowered:
    return set()

  hints: Set[str] = set()
  for tab, keywords in _TAB_KEYWORDS.items():
    for keyword in keywords:
      if re.search(rf"\b{re.escape(keyword)}\b", lowered):
        hints.add(tab)
        break
  return hints


def _extract_reference_digits(q: str, has_entity_hints: bool) -> Optional[str]:
  lowered = str(q or "").strip().lower()
  if not lowered:
    return None

  explicit_marker = re.search(r"(?:\b(?:no|number)\b|#)", lowered) is not None
  if not explicit_marker and not has_entity_hints:
    return None

  nums = re.findall(r"\d+", lowered)
  if not nums:
    return None
  return nums[-1]


def _id_suffix_digits(raw_id: Any) -> Optional[str]:
  m = re.search(r"(\d+)$", str(raw_id or "").strip())
  if not m:
    return None
  return m.group(1)


def _search_score(
  query: str,
  tab: str,
  row_id: Any,
  entity_hints: Set[str],
  number_hint: Optional[str],
  *values: Any,
) -> int:
  if entity_hints and tab not in entity_hints:
    return 0

  score = _score_match(query, *values)
  if number_hint is None:
    return score

  id_digits = _id_suffix_digits(row_id)
  if id_digits is not None and id_digits.endswith(number_hint):
    return score + 260

  # Query asks for a specific record number in this tab; hide non-matching rows.
  if entity_hints:
    return 0

  return score


def _invoice_search_item(row: Invoice, score: int) -> Dict[str, Any]:
  return {
    "tab": "invoices",
    "target": "field.search.invoices",
    "id": row.id,
    "title": row.customer,
    "subtitle": f"{row.status} | {row.currency} {row.amount}",
    "meta": {
      "status": row.status,
      "amount": row.amount,
      "currency": row.currency,
      "due": str(row.due),
      "created": str(row.created),
      "method": row.method,
    },
    "score": score,
  }


def _customer_search_item(row: Customer, score: int) -> Dict[str, Any]:
  return {
    "tab": "customers",
    "target": "field.search.customers",
    "id": row.id,
    "title": row.name,
    "subtitle": f"{row.tier} | {row.status}",
    "meta": {
      "tier": row.tier,
      "status": row.status,
      "invoices": row.invoices,
      "created_at": row.created_at.isoformat() if row.created_at else None,
    },
    "score": score,
  }


def _subscription_search_item(row: Subscription, score: int) -> Dict[str, Any]:
  return {
    "tab": "subscriptions",
    "target": "field.search.subscriptions",
    "id": row.id,
    "title": row.customer,
    "subtitle": f"{row.plan} | {row.status} | MRR {row.mrr}",
    "meta": {
      "plan": row.plan,
      "status": row.status,
      "mrr": row.mrr,
    },
    "score": score,
  }


def _normalize_prefix(prefix: str, default: str) -> str:
  cleaned = re.sub(r"[^A-Za-z0-9]", "", (prefix or "").upper())
  return cleaned or default


def _next_code(existing_ids: List[str], prefix: str, width: int = 4) -> str:
  max_num = 0
  pat = re.compile(rf"^{re.escape(prefix)}-(\d+)$", re.IGNORECASE)
  for raw in existing_ids:
    m = pat.match(str(raw or ""))
    if m:
      max_num = max(max_num, int(m.group(1)))
  return f"{prefix}-{str(max_num + 1).zfill(width)}"


def _next_model_code(session: Session, model: Type[Any], prefix: str, width: int) -> str:
  existing_ids = session.exec(select(model.id)).all()
  return _next_code(existing_ids, prefix=prefix, width=width)


def _normalize_lookup_text(value: Any) -> str:
  return re.sub(r"\s+", " ", str(value or "")).strip()


def _resolve_customer(
  session: Session,
  customer_code: Optional[str] = None,
  customer_name: Optional[str] = None,
) -> Optional[Customer]:
  code = _normalize_lookup_text(customer_code)
  name = _normalize_lookup_text(customer_name)

  if code:
    customer = session.get(Customer, code)
    if customer:
      return customer
    if not name:
      # Keep compatibility when UI sends a customer name in customer_code.
      name = code

  if name:
    wanted = name.lower()
    for row in session.exec(select(Customer)).all():
      if _normalize_lookup_text(row.name).lower() == wanted:
        return row

  return None


def _adjust_customer_invoice_count(session: Session, customer_id: str, delta: int) -> Optional[Customer]:
  customer = session.exec(
    select(Customer).where(Customer.id == customer_id).with_for_update()
  ).first()
  if not customer:
    return None

  current = int(customer.invoices or 0)
  customer.invoices = max(current + int(delta), 0)
  session.add(customer)
  return customer


def _get_settings(session: Session) -> AppSetting:
  settings = session.get(AppSetting, 1)
  if settings:
    return settings

  settings = AppSetting(id=1, company_name="FluxBill", invoice_prefix="INV")
  session.add(settings)
  session.commit()
  session.refresh(settings)
  return settings


def _serialize_settings(settings: AppSetting) -> dict:
  return {
    "company_name": settings.company_name,
    "invoice_prefix": settings.invoice_prefix,
  }


@router.get("/intial-state", summary="Initial State")
def intial_state(session: Session = Depends(get_session)):
  return {
    "settings": _serialize_settings(_get_settings(session)),
    "customers": session.exec(select(Customer)).all(),
    "subscriptions": session.exec(select(Subscription)).all(),
  }


@router.get("/invoices", response_model=List[Invoice], summary="List Invoices")
def list_invoices(
  q: Optional[str] = None,
  min_amount: Optional[int] = None,
  max_amount: Optional[int] = None,
  session: Session = Depends(get_session),
):
  rows = session.exec(select(Invoice)).all()

  if min_amount is not None:
    rows = [r for r in rows if (r.amount or 0) >= min_amount]
  if max_amount is not None:
    rows = [r for r in rows if (r.amount or 0) <= max_amount]
  if q:
    rows = [r for r in rows if _match(q, r.id, r.customer, r.status, r.method)]

  return sorted(rows, key=lambda r: str(r.created), reverse=True)


@router.get("/search", summary="Unified Search")
def unified_search(
  q: str = Query(..., min_length=1),
  active_tab: Optional[str] = Query(default=None),
  limit_per_tab: int = Query(default=5, ge=1, le=50),
  session: Session = Depends(get_session),
):
  query = q.strip()
  active = str(active_tab or "").strip().lower() or None
  tabs = ["invoices", "customers", "subscriptions"]
  entity_hints = _extract_entity_hints(query)
  number_hint = _extract_reference_digits(query, bool(entity_hints))

  invoice_hits: List[Dict[str, Any]] = []
  for row in session.exec(select(Invoice)).all():
    score = _search_score(
      query,
      "invoices",
      row.id,
      entity_hints,
      number_hint,
      row.id,
      row.customer,
      row.status,
      row.currency,
      row.method,
      row.amount,
      row.due,
      row.created,
      _TAB_ALIAS_TEXT["invoices"],
    )
    if score > 0:
      invoice_hits.append(_invoice_search_item(row, score))
  invoice_hits.sort(key=lambda x: (x["score"], str(x["id"])), reverse=True)

  customer_hits: List[Dict[str, Any]] = []
  for row in session.exec(select(Customer)).all():
    score = _search_score(
      query,
      "customers",
      row.id,
      entity_hints,
      number_hint,
      row.id,
      row.name,
      row.tier,
      row.status,
      row.invoices,
      _TAB_ALIAS_TEXT["customers"],
    )
    if score > 0:
      customer_hits.append(_customer_search_item(row, score))
  customer_hits.sort(key=lambda x: (x["score"], str(x["id"])), reverse=True)

  subscription_hits: List[Dict[str, Any]] = []
  for row in session.exec(select(Subscription)).all():
    score = _search_score(
      query,
      "subscriptions",
      row.id,
      entity_hints,
      number_hint,
      row.id,
      row.plan,
      row.customer,
      row.status,
      row.mrr,
      _TAB_ALIAS_TEXT["subscriptions"],
    )
    if score > 0:
      subscription_hits.append(_subscription_search_item(row, score))
  subscription_hits.sort(key=lambda x: (x["score"], str(x["id"])), reverse=True)

  raw_by_tab = {
    "invoices": invoice_hits,
    "customers": customer_hits,
    "subscriptions": subscription_hits,
  }
  by_tab = {tab: raw_by_tab[tab][:limit_per_tab] for tab in tabs}

  tab_order = tabs[:]
  if active in tab_order:
    tab_order.remove(active)
    tab_order.insert(0, active)

  results: List[Dict[str, Any]] = []
  for tab in tab_order:
    results.extend(by_tab[tab])

  returned_counts = {tab: len(by_tab[tab]) for tab in tabs}
  matched_counts = {tab: len(raw_by_tab[tab]) for tab in tabs}

  return {
    "query": query,
    "active_tab": active,
    "tab_order": tab_order,
    "results": results,
    "by_tab": by_tab,
    "counts": {
      "returned": returned_counts,
      "matched": matched_counts,
      "returned_total": sum(returned_counts.values()),
      "matched_total": sum(matched_counts.values()),
    },
  }


@router.post("/invoices", response_model=Invoice, summary="Create Invoice")
def create_invoice(payload: InvoiceCreate, session: Session = Depends(get_session)):
  settings = _get_settings(session)
  invoice_prefix = _normalize_prefix(settings.invoice_prefix, "INV")

  customer_code = str(payload.customer_code or payload.customer_id or "").strip()
  customer_name = str(payload.customer_name or payload.customer or "").strip()
  customer = _resolve_customer(session, customer_code=customer_code, customer_name=customer_name)
  if not customer:
    raise HTTPException(status_code=400, detail="Customer not found for invoice")

  customer_ref = customer.id
  created = date.today()
  due = payload.due or (created + timedelta(days=14))
  amount = int(payload.amount)
  currency = (payload.currency or "INR").upper()
  status = (payload.status or "draft").lower()
  method = payload.method or "-"

  for attempt in range(ID_GENERATION_MAX_RETRIES):
    invoice_id = _next_model_code(session, Invoice, invoice_prefix, width=4)
    inv = Invoice(
      id=invoice_id,
      customer=customer_ref,
      amount=amount,
      currency=currency,
      status=status,
      due=due,
      created=created,
      method=method,
    )
    session.add(inv)

    tracked_customer = _adjust_customer_invoice_count(session, customer_ref, +1)
    if not tracked_customer:
      session.rollback()
      raise HTTPException(status_code=400, detail="Customer not found for invoice")

    try:
      session.commit()
      session.refresh(inv)
      return inv
    except IntegrityError:
      session.rollback()
      if attempt == ID_GENERATION_MAX_RETRIES - 1:
        raise HTTPException(status_code=409, detail="Could not allocate invoice id. Retry request.")

  raise HTTPException(status_code=409, detail="Could not allocate invoice id. Retry request.")


@router.patch("/invoices/{invoice_id}", response_model=Invoice, summary="Update Invoice")
def patch_invoice(invoice_id: str, payload: InvoiceUpdate, session: Session = Depends(get_session)):
  inv = session.get(Invoice, invoice_id)
  if not inv:
    raise HTTPException(status_code=404, detail="Invoice not found")

  updates = payload.model_dump(exclude_unset=True)
  old_customer_ref = inv.customer

  if "customer" in updates:
    requested_customer = _normalize_lookup_text(updates.get("customer"))
    if not requested_customer:
      raise HTTPException(status_code=400, detail="Customer is required for invoice update")
    customer = _resolve_customer(session, customer_code=requested_customer, customer_name=requested_customer)
    if not customer:
      raise HTTPException(status_code=400, detail="Customer not found for invoice update")
    updates["customer"] = customer.id

  for key, value in updates.items():
    setattr(inv, key, value)

  new_customer_ref = inv.customer
  if new_customer_ref != old_customer_ref:
    old_customer = _adjust_customer_invoice_count(session, old_customer_ref, -1)
    if old_customer:
      session.add(old_customer)

    new_customer = _adjust_customer_invoice_count(session, new_customer_ref, +1)
    if not new_customer:
      session.rollback()
      raise HTTPException(status_code=400, detail="Customer not found for invoice update")
    session.add(new_customer)

  session.add(inv)
  session.commit()
  session.refresh(inv)
  return inv


@router.delete("/invoices/{invoice_id}", summary="Delete Invoice")
def delete_invoice(invoice_id: str, session: Session = Depends(get_session)):
  inv = session.get(Invoice, invoice_id)
  if not inv:
    raise HTTPException(status_code=404, detail="Invoice not found")

  customer = _adjust_customer_invoice_count(session, inv.customer, -1)
  if customer:
    session.add(customer)

  session.delete(inv)
  session.commit()
  return {"ok": True, "invoice_id": invoice_id}


@router.post("/customers", response_model=Customer, summary="Create Customer")
def create_customer(payload: CustomerCreate, session: Session = Depends(get_session)):
  name = payload.name.strip()
  if not name:
    raise HTTPException(status_code=400, detail="Customer name is required")

  provided_customer_id = str(payload.id or "").strip()
  if provided_customer_id:
    customer = Customer(
      id=provided_customer_id,
      name=name,
      tier=payload.tier,
      invoices=int(payload.invoices or 0),
      status=payload.status,
    )
    session.add(customer)
    try:
      session.commit()
      session.refresh(customer)
      return customer
    except IntegrityError:
      session.rollback()
      raise HTTPException(status_code=409, detail="Customer id already exists")

  for attempt in range(ID_GENERATION_MAX_RETRIES):
    customer_id = _next_model_code(session, Customer, "CUST", width=3)
    customer = Customer(
      id=customer_id,
      name=name,
      tier=payload.tier,
      invoices=int(payload.invoices or 0),
      status=payload.status,
    )
    session.add(customer)
    try:
      session.commit()
      session.refresh(customer)
      return customer
    except IntegrityError:
      session.rollback()
      if attempt == ID_GENERATION_MAX_RETRIES - 1:
        raise HTTPException(status_code=409, detail="Could not allocate customer id. Retry request.")

  raise HTTPException(status_code=409, detail="Could not allocate customer id. Retry request.")


@router.put("/customers/{customer_id}", response_model=Customer, summary="Update Customer")
def put_customer(customer_id: str, payload: CustomerUpdate, session: Session = Depends(get_session)):
  customer = session.get(Customer, customer_id)
  if not customer:
    raise HTTPException(status_code=404, detail="Customer not found")

  updates = payload.model_dump(exclude_unset=True)
  if "name" in updates:
    name = str(updates.get("name") or "").strip()
    if not name:
      raise HTTPException(status_code=400, detail="Customer name is required")
    updates["name"] = name
  if "tier" in updates:
    updates["tier"] = str(updates.get("tier") or "").strip() or customer.tier
  if "status" in updates:
    updates["status"] = str(updates.get("status") or "").strip() or customer.status

  for key, value in updates.items():
    setattr(customer, key, value)

  session.add(customer)
  session.commit()
  session.refresh(customer)
  return customer


@router.delete("/customers/{customer_id}", summary="Delete Customer")
def delete_customer(customer_id: str, session: Session = Depends(get_session)):
  customer = session.get(Customer, customer_id)
  if not customer:
    raise HTTPException(status_code=404, detail="Customer not found")
  session.delete(customer)
  session.commit()
  return {"ok": True, "customer_id": customer_id}


@router.get("/subscriptions", response_model=List[Subscription], summary="List Subscriptions")
def list_subscriptions(q: Optional[str] = None, session: Session = Depends(get_session)):
  rows = session.exec(select(Subscription)).all()
  if not q:
    return rows
  return [r for r in rows if _match(q, r.id, r.plan, r.customer, r.status)]


@router.post("/subscriptions", response_model=Subscription, summary="Create Subscription")
def create_subscription(payload: SubscriptionCreate, session: Session = Depends(get_session)):
  customer_code = str(payload.customer_code or payload.customer_id or "").strip()
  customer_name = str(payload.customer_name or payload.customer or "").strip()
  customer = _resolve_customer(session, customer_code=customer_code, customer_name=customer_name)
  if not customer:
    raise HTTPException(status_code=400, detail="Customer not found for subscription")

  customer_ref = customer.id
  plan = payload.plan
  mrr = int(payload.mrr or 0)
  status = payload.status

  provided_subscription_id = str(payload.id or "").strip()
  if provided_subscription_id:
    sub = Subscription(
      id=provided_subscription_id,
      plan=plan,
      customer=customer_ref,
      mrr=mrr,
      status=status,
    )
    session.add(sub)
    try:
      session.commit()
      session.refresh(sub)
      return sub
    except IntegrityError:
      session.rollback()
      raise HTTPException(status_code=409, detail="Subscription id already exists")

  for attempt in range(ID_GENERATION_MAX_RETRIES):
    sub_id = _next_model_code(session, Subscription, "SUB", width=4)
    sub = Subscription(
      id=sub_id,
      plan=plan,
      customer=customer_ref,
      mrr=mrr,
      status=status,
    )
    session.add(sub)
    try:
      session.commit()
      session.refresh(sub)
      return sub
    except IntegrityError:
      session.rollback()
      if attempt == ID_GENERATION_MAX_RETRIES - 1:
        raise HTTPException(status_code=409, detail="Could not allocate subscription id. Retry request.")

  raise HTTPException(status_code=409, detail="Could not allocate subscription id. Retry request.")


@router.put("/subscriptions/{sub_id}", response_model=Subscription, summary="Update Subscription")
def put_subscription(sub_id: str, payload: SubscriptionUpdate, session: Session = Depends(get_session)):
  subscription = session.get(Subscription, sub_id)
  if not subscription:
    raise HTTPException(status_code=404, detail="Subscription not found")

  updates = payload.model_dump(exclude_unset=True)

  has_customer_change = any(
    key in updates for key in ("customer", "customer_name", "customer_code", "customer_id")
  )
  if has_customer_change:
    raw_customer = str(updates.get("customer") or "").strip()
    customer_code = str(
      updates.get("customer_code")
      or updates.get("customer_id")
      or raw_customer
      or ""
    ).strip()
    customer_name = str(
      updates.get("customer_name")
      or raw_customer
      or ""
    ).strip()
    customer = _resolve_customer(session, customer_code=customer_code, customer_name=customer_name)
    if not customer:
      raise HTTPException(status_code=400, detail="Customer not found for subscription update")

    updates["customer"] = customer.id

  # These fields are internal input helpers; do not store directly.
  updates.pop("customer_name", None)
  updates.pop("customer_code", None)
  updates.pop("customer_id", None)

  if "plan" in updates:
    updates["plan"] = str(updates.get("plan") or "").strip() or subscription.plan
  if "status" in updates:
    updates["status"] = str(updates.get("status") or "").strip() or subscription.status

  for key, value in updates.items():
    setattr(subscription, key, value)

  session.add(subscription)
  session.commit()
  session.refresh(subscription)
  return subscription


@router.delete("/subscriptions/{sub_id}", summary="Delete Subscription")
def delete_subscription(sub_id: str, session: Session = Depends(get_session)):
  subscription = session.get(Subscription, sub_id)
  if not subscription:
    raise HTTPException(status_code=404, detail="Subscription not found")
  session.delete(subscription)
  session.commit()
  return {"ok": True, "sub_id": sub_id}


@router.patch("/settings", summary="Update Settings")
def patch_settings(payload: SettingsUpdate, session: Session = Depends(get_session)):
  settings = _get_settings(session)

  if payload.company_name is not None:
    company_name = payload.company_name.strip()
    settings.company_name = company_name or settings.company_name or "FluxBill"
  if payload.invoice_prefix is not None:
    settings.invoice_prefix = _normalize_prefix(payload.invoice_prefix, "INV")

  settings.updated_at = datetime.utcnow()
  session.add(settings)
  session.commit()
  session.refresh(settings)
  return {"company_name": settings.company_name, "invoice_prefix": settings.invoice_prefix}
