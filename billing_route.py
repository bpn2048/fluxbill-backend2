from datetime import date, datetime, timedelta
from typing import List, Optional
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from db import get_session
from models import AppSetting, Customer, Invoice, Subscription


router = APIRouter(prefix="/api", tags=["billing"])


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


class SubscriptionCreate(BaseModel):
  id: Optional[str] = None
  plan: str = "Starter"
  customer: Optional[str] = None
  customer_code: Optional[str] = None
  customer_name: Optional[str] = None
  mrr: int = Field(default=0, ge=0)
  status: str = "active"


class SettingsUpdate(BaseModel):
  company_name: Optional[str] = None
  invoice_prefix: Optional[str] = Field(default=None, min_length=1, max_length=12)


def _match(q: str, *values: str) -> bool:
  ql = q.strip().lower()
  return any(ql in (v or "").lower() for v in values)


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


def _get_settings(session: Session) -> AppSetting:
  settings = session.get(AppSetting, 1)
  if settings:
    return settings

  settings = AppSetting(id=1, company_name="FluxBill", invoice_prefix="INV")
  session.add(settings)
  session.commit()
  session.refresh(settings)
  return settings


def _seed_if_empty(session: Session) -> bool:
  has_invoice = session.exec(select(Invoice.id)).first()
  if has_invoice:
    return False

  session.add_all([
    Invoice(
      id="INV-10428",
      customer="Apex Retail Pvt Ltd",
      amount=48900,
      currency="INR",
      status="paid",
      due=date(2025, 12, 2),
      created=date(2025, 11, 25),
      method="UPI",
    ),
    Invoice(
      id="INV-10429",
      customer="BlueSky Logistics",
      amount=125000,
      currency="INR",
      status="overdue",
      due=date(2025, 12, 8),
      created=date(2025, 11, 28),
      method="Card",
    ),
    Invoice(
      id="INV-10430",
      customer="Nimbus Clinics",
      amount=76000,
      currency="INR",
      status="sent",
      due=date(2025, 12, 20),
      created=date(2025, 12, 1),
      method="NetBanking",
    ),
  ])

  session.add_all([
    Subscription(id="SUB-2201", plan="Growth", customer="Nimbus Clinics", mrr=6999, status="active"),
    Subscription(id="SUB-2202", plan="Starter", customer="Orchid Education", mrr=1999, status="active"),
  ])

  session.add_all([
    Customer(id="CUST-901", name="Apex Retail Pvt Ltd", tier="Mid-market", invoices=12, status="healthy"),
    Customer(id="CUST-902", name="BlueSky Logistics", tier="Enterprise", invoices=21, status="at_risk"),
  ])

  _get_settings(session)
  session.commit()
  return True


def _serialize_settings(settings: AppSetting) -> dict:
  return {
    "company_name": settings.company_name,
    "invoice_prefix": settings.invoice_prefix,
  }


def _serialize_state(session: Session) -> dict:
  _seed_if_empty(session)
  settings = _get_settings(session)
  customers = session.exec(select(Customer)).all()
  subscriptions = session.exec(select(Subscription)).all()
  invoices = sorted(session.exec(select(Invoice)).all(), key=lambda r: str(r.created), reverse=True)
  return {
    "settings": _serialize_settings(settings),
    "customers": customers,
    "subscriptions": subscriptions,
    "invoices": invoices,
  }


@router.get("/state", summary="State")
def state(session: Session = Depends(get_session)):
  return _serialize_state(session)


@router.get("/bootstrap", summary="Bootstrap")
def bootstrap(session: Session = Depends(get_session)):
  data = _serialize_state(session)
  return {
    "settings": data["settings"],
    "customers": data["customers"],
    "subscriptions": data["subscriptions"],
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


@router.post("/invoices", response_model=Invoice, summary="Create Invoice")
def create_invoice(payload: InvoiceCreate, session: Session = Depends(get_session)):
  _seed_if_empty(session)
  settings = _get_settings(session)
  invoice_prefix = _normalize_prefix(settings.invoice_prefix, "INV")

  customer_name = (payload.customer_name or payload.customer or "").strip()
  customer = None
  customer_code = payload.customer_code or payload.customer_id
  if customer_code:
    customer = session.get(Customer, customer_code)
    if customer:
      customer_name = customer.name

  if not customer_name:
    fallback_customer = session.exec(select(Customer)).first()
    if fallback_customer:
      customer_name = fallback_customer.name
      customer = fallback_customer

  if not customer_name:
    raise HTTPException(status_code=400, detail="No customer available for invoice")

  existing_ids = session.exec(select(Invoice.id)).all()
  invoice_id = _next_code(existing_ids, invoice_prefix, width=4)
  created = date.today()
  due = payload.due or (created + timedelta(days=14))

  inv = Invoice(
    id=invoice_id,
    customer=customer_name,
    amount=int(payload.amount),
    currency=(payload.currency or "INR").upper(),
    status=(payload.status or "draft").lower(),
    due=due,
    created=created,
    method=(payload.method or "-"),
  )
  session.add(inv)

  if customer:
    customer.invoices = int(customer.invoices or 0) + 1
    session.add(customer)

  session.commit()
  session.refresh(inv)
  return inv


def _update_invoice(invoice_id: str, payload: InvoiceUpdate, session: Session = Depends(get_session)) -> Invoice:
  inv = session.get(Invoice, invoice_id)
  if not inv:
    raise HTTPException(status_code=404, detail="Invoice not found")

  updates = payload.model_dump(exclude_unset=True)
  for key, value in updates.items():
    setattr(inv, key, value)

  session.add(inv)
  session.commit()
  session.refresh(inv)
  return inv


@router.patch("/invoices/{invoice_id}", response_model=Invoice, summary="Update Invoice")
def patch_invoice(invoice_id: str, payload: InvoiceUpdate, session: Session = Depends(get_session)):
  return _update_invoice(invoice_id, payload, session)


@router.delete("/invoices/{invoice_id}", summary="Delete Invoice")
def delete_invoice(invoice_id: str, session: Session = Depends(get_session)):
  inv = session.get(Invoice, invoice_id)
  if not inv:
    raise HTTPException(status_code=404, detail="Invoice not found")
  session.delete(inv)
  session.commit()
  return {"ok": True, "invoice_id": invoice_id}


@router.post("/customers", response_model=Customer, summary="Create Customer")
def create_customer(payload: CustomerCreate, session: Session = Depends(get_session)):
  if payload.id:
    existing = session.get(Customer, payload.id)
    if existing:
      raise HTTPException(status_code=409, detail="Customer id already exists")
    customer_id = payload.id
  else:
    ids = session.exec(select(Customer.id)).all()
    customer_id = _next_code(ids, "CUST", width=3)

  customer = Customer(
    id=customer_id,
    name=payload.name.strip(),
    tier=payload.tier,
    invoices=int(payload.invoices or 0),
    status=payload.status,
  )
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
  customer_name = (payload.customer_name or payload.customer or "").strip()
  if payload.customer_code:
    customer = session.get(Customer, payload.customer_code)
    if customer:
      customer_name = customer.name

  if not customer_name:
    fallback_customer = session.exec(select(Customer)).first()
    if fallback_customer:
      customer_name = fallback_customer.name

  if not customer_name:
    raise HTTPException(status_code=400, detail="No customer available for subscription")

  if payload.id:
    existing = session.get(Subscription, payload.id)
    if existing:
      raise HTTPException(status_code=409, detail="Subscription id already exists")
    sub_id = payload.id
  else:
    ids = session.exec(select(Subscription.id)).all()
    sub_id = _next_code(ids, "SUB", width=4)

  sub = Subscription(
    id=sub_id,
    plan=payload.plan,
    customer=customer_name,
    mrr=int(payload.mrr or 0),
    status=payload.status,
  )
  session.add(sub)
  session.commit()
  session.refresh(sub)
  return sub


@router.delete("/subscriptions/{sub_id}", summary="Delete Subscription")
def delete_subscription(sub_id: str, session: Session = Depends(get_session)):
  subscription = session.get(Subscription, sub_id)
  if not subscription:
    raise HTTPException(status_code=404, detail="Subscription not found")
  session.delete(subscription)
  session.commit()
  return {"ok": True, "sub_id": sub_id}


def _update_settings(payload: SettingsUpdate, session: Session = Depends(get_session)) -> dict:
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


@router.patch("/settings", summary="Update Settings")
def patch_settings(payload: SettingsUpdate, session: Session = Depends(get_session)):
  return _update_settings(payload, session)

