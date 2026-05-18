# db.py
import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

load_dotenv()

# 1. Look for the Railway cloud variable, fallback to local dev string if not found
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://postgres:1234@localhost:5432/fluxbill_demo2"
).strip()

# 2. Production safe-check: Railway sometimes structures strings as 'postgres://' 
# instead of 'postgresql://', which causes SQLAlchemy to throw an error.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session