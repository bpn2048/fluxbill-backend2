# db.py
import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

load_dotenv()

# 1. Fetch Railway cloud variable or fallback to local dev credentials
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://postgres:1234@localhost:5432/fluxbill_demo2"
).strip()

# 2. Production safe-check for legacy structural strings
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3. Enhanced production pool configs to avoid dropped connections or query blocks
engine = create_engine(
    DATABASE_URL, 
    echo=False, 
    pool_pre_ping=True,       # Verifies connections are alive before serving requests
    pool_size=10,             # Maintains a steady pool of active database connections
    max_overflow=20           # Temporarily scales connections during peak dashboard loading
)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session