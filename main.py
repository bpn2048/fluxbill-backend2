import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlmodel import Session

from billing_route import router as billing_router
from db import engine, init_db

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small").strip()
CORS_ORIGINS = [
  x.strip()
  for x in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
  if x.strip()
]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$").strip() or None

_whisper_model = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
  init_db()
  yield


app = FastAPI(title="FluxBill Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
  CORSMiddleware,
  allow_origins=CORS_ORIGINS,
  allow_origin_regex=CORS_ORIGIN_REGEX,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
app.include_router(billing_router)
init_db()

ACTIONS = [
  "click",
  "type",
  "none",
  "create_invoice",
  "delete_invoice",
  "update_invoice",
  "filter_invoices",
  "create_customer",
  "delete_customer",
  "create_subscription",
  "delete_subscription",
]

TARGETS = [
  "nav.dashboard",
  "nav.invoices",
  "nav.subscriptions",
  "nav.customers",
  "nav.reports",
  "nav.settings",
  "field.search",
  "action.createInvoice",
  "action.collectPayment",
]


class AssistantTextRequest(BaseModel):
  text: str
  active_tab: str = "dashboard"


class Command(BaseModel):
  action: str = Field(default="none")
  target: Optional[str] = None
  args: Dict[str, Any] = Field(default_factory=dict)
  reply: str = "ok"


def _get_whisper_model():
  global _whisper_model
  if _whisper_model is None:
    _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
  return _whisper_model


def _system_prompt() -> str:
  return f"""
You are an intent + entity extractor for a React billing dashboard.

Return ONLY valid JSON with this schema:
{{
  "action": "one of: {", ".join(ACTIONS)}",
  "target": "one of: {", ".join(TARGETS)} or null",
  "args": {{ "any": "key-values" }},
  "reply": "short human message"
}}

Rules:
- Use action="click" when the user wants to navigate tabs or press a UI button.
- Use action="type" only for search input, set target="field.search" and args={{"text":"..."}}.
- For create/delete/update/filter actions, put extracted fields inside args and set target=null.
- If user uses verbs like create/add/new, delete/remove, update/edit/change for invoice/customer/subscription, choose the matching CRUD action and DO NOT return action="click".
- Only use action="click" for explicit navigation intent (open/go to/switch tab).
- If user says a bare phrase like "delete acme", infer delete_customer with args={{"name":"Acme"}}.
- Never ask for confirmation (for example "are you sure?") when user asks create/update/delete; treat it as confirmed and return the corresponding action directly.
- If user intent is unclear, output action="none" and ask a short question in reply.
- Always respond in English.

Extraction formats:
- create_customer args: {{ "name": "...", "tier": "SMB|Mid-market|Enterprise", "status": "active|new|at_risk" }}
- delete_customer args: {{ "customer_id": "CUST-0901" }} OR {{ "name": "Apex Retail Pvt Ltd" }}
- create_invoice args: {{ "customer_id": "...", "customer_name": "...", "amount": 25000, "currency": "INR", "status": "draft|sent|paid|overdue" }}
- update_invoice args: {{ "invoice_id": "INV-10431", "status": "paid", "amount": 25000, "currency": "INR", "due": "YYYY-MM-DD", "method": "UPI", "customer_name": "..." }}
- delete_invoice args: {{ "invoice_id": "INV-10431" }}
- filter_invoices args: {{ "amount_min": 10000, "amount_max": 50000, "status": "paid|sent|overdue|draft" }}
- create_subscription args: {{ "customer_id": "...", "customer_name": "...", "plan": "Starter|Growth|Enterprise", "mrr": 6999, "status": "active|past_due|canceled" }}
- delete_subscription args: {{ "subscription_id": "SUB-2201" }}
""".strip()


def _extract_json(text_in: str) -> Optional[Dict[str, Any]]:
  try:
    obj = json.loads(text_in)
    if isinstance(obj, dict):
      return obj
  except Exception:
    pass

  m = re.search(r"\{[\s\S]*\}", text_in)
  if not m:
    return None
  try:
    obj = json.loads(m.group(0))
    return obj if isinstance(obj, dict) else None
  except Exception:
    return None


async def plan_command(user_text: str, active_tab: str) -> Command:
  if not OPENROUTER_API_KEY:
    raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

  url = "https://openrouter.ai/api/v1/chat/completions"
  headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
  }
  payload = {
    "model": OPENROUTER_MODEL,
    "messages": [
      {"role": "system", "content": _system_prompt()},
      {"role": "user", "content": f"Active tab: {active_tab}\nUser: {user_text}"},
    ],
    "temperature": 0.1,
  }

  async with httpx.AsyncClient(timeout=60) as client:
    r = await client.post(url, headers=headers, json=payload)
    if r.status_code >= 400:
      raise HTTPException(status_code=500, detail=f"OpenRouter error: {r.status_code} {r.text}")

  content = r.json()["choices"][0]["message"]["content"]
  obj = _extract_json(content)
  if not obj:
    return Command(action="none", reply='I could not understand. Try: "open invoices", "search apex".')

  try:
    cmd = Command(**obj)
  except ValidationError:
    return Command(action="none", reply="I got an invalid command format. Try again.")

  if cmd.action not in ACTIONS:
    return Command(action="none", reply="That action is not supported.")
  if cmd.target is not None and cmd.target not in TARGETS:
    return Command(action="none", reply="That UI target is not supported.")

  if cmd.action == "type":
    text_value = (cmd.args.get("text") or "").strip()
    if cmd.target != "field.search" or not text_value:
      return Command(action="none", reply="Tell me what to search for.")
    cmd.args["text"] = text_value

  return cmd


def transcribe_audio(file_path: str) -> str:
  whisper = _get_whisper_model()
  segments, _info = whisper.transcribe(
    file_path,
    language="en",
    task="transcribe",
    vad_filter=True,
    initial_prompt="This is English. Transcribe only English words.",
  )
  parts = []
  for s in segments:
    if s.text:
      parts.append(s.text.strip())
  return " ".join([p for p in parts if p]).strip()


@app.get("/health", include_in_schema=False)
def health():
  return {"ok": True, "whisper_model": WHISPER_MODEL_NAME, "openrouter_model": OPENROUTER_MODEL}


@app.get("/health/db", include_in_schema=False)
def health_db():
  try:
    with Session(engine) as session:
      session.exec(text("SELECT 1"))
    return {"ok": True}
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"DB connection failed: {exc}") from exc


@app.post("/assistant/text", include_in_schema=False)
async def assistant_text(req: AssistantTextRequest):
  cmd = await plan_command(req.text, req.active_tab)
  return {"transcript": req.text, "command": cmd.model_dump()}


@app.post("/assistant/voice", include_in_schema=False)
async def assistant_voice(file: UploadFile = File(...), active_tab: str = Form("dashboard")):
  filename = file.filename or "voice"
  _, ext = os.path.splitext(filename)
  if not ext:
    ext = ".webm"

  with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp_path = tmp.name
    tmp.write(await file.read())

  try:
    transcript = transcribe_audio(tmp_path)
    if not transcript:
      return {"transcript": "", "command": Command(action="none", reply="I could not hear anything. Try again.").model_dump()}

    cmd = await plan_command(transcript, active_tab)
    return {"transcript": transcript, "command": cmd.model_dump()}
  finally:
    try:
      os.remove(tmp_path)
    except Exception:
      pass
