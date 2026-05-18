import os
from datetime import date, datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class AppSetting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    company_name: str
    invoice_prefix: str
    # Explicitly mark nullable on both the database and pydantic layers
    updated_at: Optional[datetime] = Field(default=None, sa_column_kwargs={"nullable": True})


class Customer(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    tier: str = "SMB"  # SMB | Mid-market | Enterprise
    invoices: int = 0
    status: str = "new"  # active | new | at_risk
    created_at: Optional[datetime] = Field(default=None, sa_column_kwargs={"nullable": True})


class Invoice(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer: str  # References Customer.id
    amount: int
    currency: str = "INR"
    status: str = "draft"  # draft | sent | paid | overdue
    created: date
    due: Optional[date] = Field(default=None, sa_column_kwargs={"nullable": True})
    method: Optional[str] = Field(default="-", sa_column_kwargs={"nullable": True})


class Subscription(SQLModel, table=True):
    id: str = Field(primary_key=True)
    plan: str = "Starter"  # Starter | Growth | Enterprise
    customer: str  # References Customer.id
    mrr: int = 0
    status: str = "active"  # active | past_due | canceled
    # Force direct null safety evaluation for your imported database rows
    created_at: Optional[datetime] = Field(default=None, sa_column_kwargs={"nullable": True})