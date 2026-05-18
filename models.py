import os
from datetime import date, datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class AppSetting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    company_name: str
    invoice_prefix: str
    # Made optional to safely handle missing or empty initialization timestamps
    updated_at: Optional[datetime] = Field(default=None)


class Customer(SQLModel, table=True):
    # Some pre-existing setups might pass explicit code keys (e.g., 'CUST-001')
    id: str = Field(primary_key=True)
    name: str
    tier: str = "SMB"  # SMB | Mid-market | Enterprise
    invoices: int = 0
    status: str = "new"  # active | new | at_risk
    # Made optional to handle records imported without explicit creation history
    created_at: Optional[datetime] = Field(default=None)


class Invoice(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer: str  # References Customer.id
    amount: int
    currency: str = "INR"
    status: str = "draft"  # draft | sent | paid | over-due
    created: date
    # Made optional to handle old historical or un-sent invoice rows cleanly
    due: Optional[date] = Field(default=None)
    method: Optional[str] = Field(default="-")  # UPI | Card | NetBanking | -


class Subscription(SQLModel, table=True):
    id: str = Field(primary_key=True)
    plan: str = "Starter"  # Starter | Growth | Enterprise
    customer: str  # References Customer.id
    mrr: int = 0
    status: str = "active"  # active | past_due | canceled
    # Made optional to gracefully handle subscriptions generated without explicit date bounds
    created_at: Optional[datetime] = Field(default=None)