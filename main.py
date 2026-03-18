import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from billing_route import router as billing_router
from db import init_db

load_dotenv()


def _env_float(name: str, default: float) -> float:
  raw = os.getenv(name, "").strip()
  if not raw:
    return default
  try:
    return float(raw)
  except ValueError:
    return default


def _env_int(name: str, default: int) -> int:
  raw = os.getenv(name, "").strip()
  if not raw:
    return default
  try:
    return int(raw)
  except ValueError:
    return default


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://127.0.0.1:8000").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "FluxBill Backend").strip()
OPENROUTER_TIMEOUT_SECONDS = _env_float("OPENROUTER_TIMEOUT_SECONDS", 60.0)

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small").strip()
ASSISTANT_HISTORY_MAX_MESSAGES = max(_env_int("ASSISTANT_HISTORY_MAX_MESSAGES", 12), 2)

CORS_ORIGINS = [
  x.strip()
  for x in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
  if x.strip()
]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$").strip() or None

_whisper_model = None
_history_lock = Lock()
_planner_lock = Lock()
_session_histories: Dict[str, InMemoryChatMessageHistory] = {}
_planner_chain: Optional[RunnableWithMessageHistory] = None


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


@app.get("/", include_in_schema=False)
async def root_ping():
  # Browsers frequently probe "/" when opening the backend origin.
  return {"ok": True, "service": "FluxBill Backend"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
  # Avoid repeated 404s for automatic favicon probes.
  return Response(status_code=204)


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

NAVIGATION_TARGET_ALIASES = {
  "nav.dashboard": {"dashboard", "dash", "home", "overview"},
  "nav.invoices": {"invoice", "invoices", "in voice", "in voices"},
  "nav.subscriptions": {"subscription", "subscriptions", "plan", "plans"},
  "nav.customers": {"customer", "customers", "client", "clients"},
  "nav.reports": {"report", "reports", "analytics"},
  "nav.settings": {"setting", "settings", "preferences"},
}

NAVIGATION_VERBS = (
  "switch tab to",
  "navigate to",
  "take me to",
  "switch to",
  "go to",
  "goto",
  "open",
  "show",
)

INCOMPLETE_COMMAND_FILLER_WORDS = {
  "a",
  "an",
  "the",
  "my",
  "me",
  "please",
  "now",
  "tab",
  "page",
  "screen",
  "section",
  "item",
  "thing",
  "one",
  "something",
  "anything",
  "new",
  "it",
  "this",
  "that",
}

INCOMPLETE_COMMAND_PROMPTS = (
  (
    NAVIGATION_VERBS,
    "What should I open: dashboard, invoices, customers, subscriptions, reports, or settings?",
  ),
  (
    ("create", "add", "new"),
    "What should I create: invoice, customer, or subscription?",
  ),
  (
    ("delete", "remove"),
    "What should I delete: invoice, customer, or subscription?",
  ),
  (
    ("update", "edit", "change"),
    "What should I update: invoice, customer, or subscription?",
  ),
  (
    ("search", "find"),
    "What should I search for?",
  ),
  (
    ("filter",),
    "What should I filter: invoices by amount or status?",
  ),
)

VOICE_HOTWORDS = (
  "invoice, invoices, customer, customers, subscription, subscriptions, "
  "dashboard, reports, settings, billing"
)

VOICE_TRANSCRIPT_REPLACEMENTS = (
  (r"\bin voices\b", "invoices"),
  (r"\bin voice\b", "invoice"),
  (r"\bopen invoices\b", "open invoices"),
  (r"\bopen invoice\b", "open invoice"),
  (r"\bgo to in voices\b", "go to invoices"),
  (r"\bgo to in voice\b", "go to invoice"),
  (r"\bswitch to in voices\b", "switch to invoices"),
  (r"\bswitch to in voice\b", "switch to invoice"),
  (r"\bshow in voices\b", "show invoices"),
  (r"\bshow in voice\b", "show invoice"),
  (r"\bdelete in voices\b", "delete invoices"),
  (r"\bdelete in voice\b", "delete invoice"),
  (r"\bupdate in voices\b", "update invoices"),
  (r"\bupdate in voice\b", "update invoice"),
  (r"\bcreate in voice\b", "create invoice"),
)


class AssistantTextRequest(BaseModel):
  text: str
  active_tab: str = "dashboard"
  available_targets: List[str] = Field(default_factory=list)
  session_id: Optional[str] = None


class Command(BaseModel):
  action: str = Field(default="none")
  target: Optional[str] = None
  args: Dict[str, Any] = Field(default_factory=dict)
  reply: str = "ok"


_command_parser = PydanticOutputParser(pydantic_object=Command)


def _normalize_session_id(raw_session_id: Optional[str]) -> str:
  cleaned = re.sub(r"[^A-Za-z0-9._:-]", "", str(raw_session_id or "").strip())
  return cleaned[:80] or "default"


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
  sid = _normalize_session_id(session_id)
  with _history_lock:
    history = _session_histories.get(sid)
    if history is None:
      history = InMemoryChatMessageHistory()
      _session_histories[sid] = history
    if len(history.messages) > ASSISTANT_HISTORY_MAX_MESSAGES:
      history.messages = history.messages[-ASSISTANT_HISTORY_MAX_MESSAGES:]
    return history


def _build_planner_chain() -> RunnableWithMessageHistory:
  headers: Dict[str, str] = {}
  if OPENROUTER_SITE_URL:
    headers["HTTP-Referer"] = OPENROUTER_SITE_URL
  if OPENROUTER_APP_NAME:
    headers["X-Title"] = OPENROUTER_APP_NAME

  llm = ChatOpenAI(
    model_name=OPENROUTER_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0.1,
    request_timeout=OPENROUTER_TIMEOUT_SECONDS,
    max_retries=1,
    default_headers=headers,
  )
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "{system_prompt}\n\nFormat instructions:\n{format_instructions}"),
      MessagesPlaceholder(variable_name="history"),
      ("human", "{user_message}"),
    ]
  )
  base_chain = prompt | llm | StrOutputParser()
  return RunnableWithMessageHistory(
    base_chain,
    _get_session_history,
    input_messages_key="user_message",
    history_messages_key="history",
  )


def _get_planner_chain() -> RunnableWithMessageHistory:
  global _planner_chain
  if _planner_chain is None:
    with _planner_lock:
      if _planner_chain is None:
        _planner_chain = _build_planner_chain()
  return _planner_chain


def _is_supported_target(target: str) -> bool:
  t = (target or "").strip()
  if not t:
    return False
  return t in TARGETS or t.startswith("field.search.")


def _normalize_targets(raw_targets: List[str]) -> List[str]:
  seen = set()
  out: List[str] = []
  for raw in raw_targets or []:
    t = str(raw or "").strip().lower()
    if not t or t in seen:
      continue
    seen.add(t)
    out.append(t)
  return out


def _build_user_message(user_text: str, active_tab: str, available_targets: List[str]) -> str:
  return (
    f"Active tab: {active_tab}\n"
    f"Available targets: {', '.join(_normalize_targets(available_targets)) or 'none'}\n"
    f"User: {user_text}"
  )


def _normalize_phrase(text: str) -> str:
  collapsed = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
  return re.sub(r"\s+", " ", collapsed).strip()


def _match_navigation_target(user_text: str) -> Optional[str]:
  normalized = _normalize_phrase(user_text)
  if not normalized:
    return None

  matched_verb = next(
    (verb for verb in NAVIGATION_VERBS if normalized == verb or normalized.startswith(f"{verb} ")),
    None,
  )
  if not matched_verb:
    return None

  remainder = normalized[len(matched_verb):].strip()
  remainder = re.sub(r"^(the|my)\s+", "", remainder)
  remainder = re.sub(r"\s+(tab|page|screen|section)$", "", remainder).strip()
  if not remainder:
    return None

  for target, aliases in NAVIGATION_TARGET_ALIASES.items():
    if remainder in aliases:
      return target
  return None


def _build_navigation_command(target: str) -> Command:
  label = target.split(".", 1)[1]
  return Command(action="click", target=target, reply=f"opening {label}.")


def _match_incomplete_command_reply(user_text: str) -> Optional[str]:
  normalized = _normalize_phrase(user_text)
  if not normalized:
    return None

  for verbs, reply in INCOMPLETE_COMMAND_PROMPTS:
    matched_verb = next(
      (verb for verb in verbs if normalized == verb or normalized.startswith(f"{verb} ")),
      None,
    )
    if not matched_verb:
      continue

    remainder = normalized[len(matched_verb):].strip()
    if not remainder:
      return reply

    tokens = [token for token in remainder.split(" ") if token]
    if tokens and all(token in INCOMPLETE_COMMAND_FILLER_WORDS for token in tokens):
      return reply

  return None


def _normalize_voice_transcript(text: str) -> str:
  normalized = str(text or "").strip()
  if not normalized:
    return ""

  for pattern, replacement in VOICE_TRANSCRIPT_REPLACEMENTS:
    normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

  normalized = re.sub(r"\s+", " ", normalized).strip()
  return normalized


def _apply_search_routing(cmd: Command, active_tab: str, available_targets: List[str]) -> Command:
  if cmd.action != "type":
    return cmd

  text_value = str(cmd.args.get("text") or "").strip()
  if not text_value:
    return cmd

  tab_key = str(active_tab or "").strip().lower()
  requested_target = (cmd.target or "field.search").strip().lower()
  available = _normalize_targets(available_targets)
  available_set = set(available)

  candidates: List[str] = []
  page_target = f"field.search.{tab_key}" if tab_key else ""

  if available:
    if page_target and page_target in available_set:
      candidates.append(page_target)
    if requested_target and requested_target in available_set:
      candidates.append(requested_target)
    if "field.search" in available_set:
      candidates.append("field.search")
  else:
    if page_target:
      candidates.append(page_target)
    if requested_target:
      candidates.append(requested_target)
    candidates.append("field.search")

  deduped: List[str] = []
  seen = set()
  for t in candidates:
    if not t or t in seen:
      continue
    seen.add(t)
    deduped.append(t)

  if not deduped:
    deduped = ["field.search"]

  cmd.target = deduped[0]
  cmd.args["text"] = text_value
  cmd.args["search_targets"] = deduped
  return cmd


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
- Use prior conversation turns in history for follow-up requests.
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


async def plan_command(
  user_text: str,
  active_tab: str,
  available_targets: List[str],
  session_id: Optional[str] = None,
) -> Command:
  incomplete_reply = _match_incomplete_command_reply(user_text)
  if incomplete_reply:
    return Command(action="none", reply=incomplete_reply)

  nav_target = _match_navigation_target(user_text)
  if nav_target:
    return _build_navigation_command(nav_target)

  if not OPENROUTER_API_KEY:
    raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

  normalized_session_id = _normalize_session_id(session_id)
  user_message = _build_user_message(user_text, active_tab, available_targets)

  try:
    content = await _get_planner_chain().ainvoke(
      {
        "system_prompt": _system_prompt(),
        "format_instructions": _command_parser.get_format_instructions(),
        "user_message": user_message,
      },
      config={
        "configurable": {"session_id": normalized_session_id},
        "metadata": {"component": "assistant_command_planner", "session_id": normalized_session_id},
      },
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"LangChain planner error: {exc}") from exc

  try:
    cmd = _command_parser.parse(content)
  except OutputParserException:
    obj = _extract_json(content)
    if not obj:
      return Command(action="none", reply='I could not understand. Try: "open invoices", "search apex".')
    try:
      cmd = Command(**obj)
    except ValidationError:
      return Command(action="none", reply="I got an invalid command format. Try again.")

  if cmd.action not in ACTIONS:
    return Command(action="none", reply="That action is not supported.")
  if cmd.target is not None and not _is_supported_target(cmd.target):
    return Command(action="none", reply="That UI target is not supported.")

  if cmd.action == "type":
    text_value = (cmd.args.get("text") or "").strip()
    if (cmd.target is not None and not str(cmd.target).startswith("field.search")) or not text_value:
      return Command(action="none", reply="Tell me what to search for.")
    cmd.args["text"] = text_value
    cmd = _apply_search_routing(cmd, active_tab, available_targets)

  return cmd


def transcribe_audio(file_path: str) -> str:
  whisper = _get_whisper_model()
  segments, _info = whisper.transcribe(
    file_path,
    language="en",
    task="transcribe",
    vad_filter=True,
    initial_prompt=(
      "This is English for a billing dashboard. Important words include invoice, invoices, "
      "customer, customers, subscription, subscriptions, dashboard, reports, and settings."
    ),
    hotwords=VOICE_HOTWORDS,
  )
  parts = []
  for s in segments:
    if s.text:
      parts.append(s.text.strip())
  return _normalize_voice_transcript(" ".join([p for p in parts if p]).strip())


@app.post("/assistant/text", include_in_schema=False)
async def assistant_text(req: AssistantTextRequest):
  session_id = _normalize_session_id(req.session_id)
  cmd = await plan_command(req.text, req.active_tab, req.available_targets, session_id=session_id)
  return {"session_id": session_id, "transcript": req.text, "command": cmd.model_dump()}


@app.post("/assistant/voice", include_in_schema=False)
async def assistant_voice(
  file: UploadFile = File(...),
  active_tab: str = Form("dashboard"),
  available_targets_json: str = Form("[]"),
  session_id: str = Form("default"),
):
  normalized_session_id = _normalize_session_id(session_id)
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
      return {
        "session_id": normalized_session_id,
        "transcript": "",
        "command": Command(action="none", reply="I could not hear anything. Try again.").model_dump(),
      }

    available_targets: List[str]
    try:
      parsed = json.loads(available_targets_json or "[]")
      available_targets = parsed if isinstance(parsed, list) else []
    except Exception:
      available_targets = []

    cmd = await plan_command(transcript, active_tab, available_targets, session_id=normalized_session_id)
    return {"session_id": normalized_session_id, "transcript": transcript, "command": cmd.model_dump()}
  finally:
    try:
      os.remove(tmp_path)
    except Exception:
      pass
