# models.py
from datetime import date, datetime

from sqlmodel import Field, SQLModel


class Customer(SQLModel, table=True):
  id: str = Field(primary_key=True, index=True)
  name: str
  tier: str = "SMB"
  invoices: int = 0
  status: str = "new"
  created_at: datetime = Field(default_factory=datetime.utcnow)


class Invoice(SQLModel, table=True):
  id: str = Field(primary_key=True, index=True)  # INV-10428
  customer: str = Field(foreign_key="customer.id", index=True)
  amount: int
  currency: str = "INR"
  status: str = "sent"  # draft|sent|overdue|paid
  due: date
  created: date
  method: str = "-"


class Subscription(SQLModel, table=True):
  id: str = Field(primary_key=True, index=True)  # SUB-2201
  plan: str
  customer: str = Field(foreign_key="customer.id", index=True)
  mrr: int
  status: str = "active"  # active|past_due|canceled


class AppSetting(SQLModel, table=True):
  id: int = Field(default=1, primary_key=True)
  company_name: str = "FluxBill"
  invoice_prefix: str = "INV"
  updated_at: datetime = Field(default_factory=datetime.utcnow)
