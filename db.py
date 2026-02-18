# db.py
import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
  raise RuntimeError("DATABASE_URL is not set in backend .env")

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

def init_db() -> None:
  SQLModel.metadata.create_all(engine)

def get_session():
  with Session(engine) as session:
    yield session
