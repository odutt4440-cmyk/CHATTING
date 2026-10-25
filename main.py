"""
main.py — AI Conversation Room Manager (single-file)

Modes:
  python main.py                  -> run the Telegram bot + userbot runtime
  python main.py --link <uid>     -> interactive, terminal-only account authorization

Permissions:
  - START / PAUSE / RESUME / STOP  -> creator AND partner (both)
  - INVITE / CANCEL / LOCK / UNLOCK / GROUP CONFIG -> creator only

Conversation behaviour:
  - STOP then START continues from where the conversation was stopped.
  - Manual replies by participants are ingested and the AI continues
    naturally from them — no disturbance to the turn flow.
  - Strict English-only output (prompt-level rule + language guard).

Stack:
  - Kurigram (Pyrogram-compatible fork, imported via the `pyrogram` path)
  - Telethon (StringSession) for authorized user accounts
  - aiosqlite (SQLite) for persistence
  - aiohttp for an OpenAI-compatible AI endpoint (Groq/OpenRouter/OpenAI/...)
  - cryptography (Fernet) for session encryption at rest
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import getpass
import logging
import os
import random
import re
import signal
import sys
import time
from typing import Any, Optional

import aiohttp
import aiosqlite
from cryptography.fernet import Fernet, InvalidToken
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.errors import (
    FloodWaitError,
    SessionPasswordNeededError,
    PhoneCodeInvalidError,
    PhoneCodeExpiredError,
    AuthKeyUnregisteredError,
)

# Kurigram is installed as `kurigram` but exposes the pyrogram import path.
from pyrogram import Client, filters, enums, errors as pyro_errors
from pyrogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    Message,
)

# ----------------------------------------------------------------------------
# Logging (never logs session material, OTPs, or passwords)
# ----------------------------------------------------------------------------
class SensitiveFilter(logging.Filter):
    PATTERNS = re.compile(
        r"(StringSession|session_string|otp|password|auth_key)", re.IGNORECASE
    )

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.PATTERNS.sub("[REDACTED]", str(record.msg))
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
for _h in logging.getLogger().handlers:
    _h.addFilter(SensitiveFilter())
log = logging.getLogger("rooms")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN", "")
    API_ID = int(os.getenv("API_ID", "0") or 0)
    API_HASH = os.getenv("API_HASH", "")
    AI_API_KEY = os.getenv("AI_API_KEY", "")
    AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()
    AI_MODEL = os.getenv("AI_MODEL", "")
    AI_BASE_URL = os.getenv("AI_BASE_URL", "")
    DATABASE_URL = os.getenv("DATABASE_URL", "rooms.db")  # sqlite file path
    OWNER_ID = int(os.getenv("OWNER_ID", "0") or 0)
    SESSION_KEY = os.getenv("SESSION_ENCRYPTION_KEY", "")
    INVITE_TTL_MINUTES = int(os.getenv("INVITE_TTL_MINUTES", "120"))
    DEFAULT_INTERVAL = int(os.getenv("DEFAULT_INTERVAL", "45"))
    DEFAULT_MAX_MESSAGES = int(os.getenv("DEFAULT_MAX_MESSAGES", "60"))
    DEFAULT_SESSION_MINUTES = int(os.getenv("DEFAULT_SESSION_MINUTES", "60"))
    DEFAULT_MAX_LEN = int(os.getenv("DEFAULT_MAX_LEN", "400"))
    DEFAULT_HISTORY_LEN = int(os.getenv("DEFAULT_HISTORY_LEN", "20"))
    DEFAULT_DAILY_LIMIT = int(os.getenv("DEFAULT_DAILY_LIMIT", "120"))
    MAX_AI_RETRIES = 3
    AI_TIMEOUT = 60

    PROVIDERS = {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "together": "https://api.together.xyz/v1",
        "custom": "",  # must set AI_BASE_URL
    }
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "groq": "llama-3.3-70b-versatile",
        "openrouter": "openai/gpt-4o-mini",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "custom": "default",
    }

    @property
    def ai_base(self) -> str:
        return self.AI_BASE_URL or self.PROVIDERS.get(self.AI_PROVIDER, "")

    @property
    def ai_model(self) -> str:
        return self.AI_MODEL or self.DEFAULT_MODELS.get(self.AI_PROVIDER, "gpt-4o-mini")

    @property
    def fernet(self) -> Optional[Fernet]:
        if not self.SESSION_KEY:
            return None
        try:
            return Fernet(self.SESSION_KEY.encode())
        except Exception:
            log.error("SESSION_ENCRYPTION_KEY is not a valid Fernet key.")
            return None


CFG = Config()

# ----------------------------------------------------------------------------
# Database (SQLite via aiosqlite)
# ----------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rooms (
    room_id TEXT PRIMARY KEY,
    creator_id INTEGER NOT NULL,
    partner_id INTEGER,
    status TEXT NOT NULL,             -- waiting|linking|config|ready|active|paused|stopped|cancelled
    group_id INTEGER,
    locked INTEGER NOT NULL DEFAULT 0,
    topic TEXT,
    mode TEXT NOT NULL DEFAULT 'topic',
    created_at TEXT NOT NULL,
    started_at TEXT,
    stopped_at TEXT,
    msg_count_a INTEGER NOT NULL DEFAULT 0,
    msg_count_b INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS invites (
    code TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_by INTEGER,
    used_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    user_id INTEGER PRIMARY KEY,
    session_enc TEXT NOT NULL,
    linked_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    room_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL DEFAULT 'the future of technology and society',
    interval INTEGER NOT NULL,
    max_messages INTEGER NOT NULL,
    session_minutes INTEGER NOT NULL,
    max_len INTEGER NOT NULL,
    history_len INTEGER NOT NULL,
    mention_policy TEXT NOT NULL DEFAULT 'first_only',  -- first_only|always|never
    daily_limit INTEGER NOT NULL,
    personality_a TEXT NOT NULL,
    personality_b TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL,
    role TEXT NOT NULL,               -- A|B
    text TEXT NOT NULL,
    tg_msg_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_room ON history(room_id, id);
CREATE TABLE IF NOT EXISTS usage_daily (
    user_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS seen_msgs (
    room_id TEXT NOT NULL,
    tg_msg_id INTEGER NOT NULL,
    PRIMARY KEY (room_id, tg_msg_id)
);
"""

STATUS_LABEL = {
    "waiting": "Waiting for partner",
    "linking": "Accounts linking",
    "config": "Configure target group",
    "ready": "READY",
    "active": "ACTIVE",
    "paused": "PAUSED",
    "stopped": "Stopped",
    "cancelled": "Cancelled",
}


class Database:
    def __init__(self, path: str):
        self.path = path
        self.db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        # Railway volume support: create parent dir if it doesn't exist
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript(SCHEMA)
        await self.db.commit()

    async def close(self) -> None:
        if self.db:
            await self.db.close()

    async def execute(self, sql: str, params: tuple = ()) -> None:
        assert self.db
        await self.db.execute(sql, params)
        await self.db.commit()

    async def fetchone(self, sql: str, params: tuple = ()) -> Optional[aiosqlite.Row]:
        assert self.db
        cur = await self.db.execute(sql, params)
        return await cur.fetchone()

    async def fetchall(self, sql: str, params: tuple = ()) -> list:
        assert self.db
        cur = await self.db.execute(sql, params)
        return await cur.fetchall()

    # ---- users -------------------------------------------------------------
    async def upsert_user(self, user_id: int, username: Optional[str]) -> None:
        await self.execute(
            "INSERT INTO users(user_id, username, created_at) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET username=excluded.username",
            (user_id, username or "", dt.datetime.utcnow().isoformat()),
        )

    # ---- rooms -------------------------------------------------------------
    async def create_room(self, room_id: str, creator_id: int) -> None:
        await self.execute(
            "INSERT INTO rooms(room_id, creator_id, status, created_at) VALUES(?,?,?,?)",
            (room_id, creator_id, "waiting", dt.datetime.utcnow().isoformat()),
        )
        await self.execute(
            "INSERT OR IGNORE INTO settings(room_id, interval, max_messages, "
            "session_minutes, max_len, history_len, daily_limit) VALUES(?,?,?,?,?,?,?)",
            (room_id, CFG.DEFAULT_INTERVAL, CFG.DEFAULT_MAX_MESSAGES,
             CFG.DEFAULT_SESSION_MINUTES, CFG.DEFAULT_MAX_LEN,
             CFG.DEFAULT_HISTORY_LEN, CFG.DEFAULT_DAILY_LIMIT),
        )

    async def get_room(self, room_id: str) -> Optional[dict]:
        row = await self.fetchone("SELECT * FROM rooms WHERE room_id=?", (room_id,))
        return dict(row) if row else None

    async def get_room_by_creator(self, user_id: int) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT * FROM rooms WHERE creator_id=? AND status NOT IN "
            "('stopped','cancelled') ORDER BY created_at DESC LIMIT 1", (user_id,))
        return dict(row) if row else None

    async def get_room_of_participant(self, user_id: int) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT * FROM rooms WHERE (creator_id=? OR partner_id=?) AND "
            "status NOT IN ('stopped','cancelled') ORDER BY created_at DESC LIMIT 1",
            (user_id, user_id))
        return dict(row) if row else None

    async def update_room(self, room_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        await self.execute(
            f"UPDATE rooms SET {sets} WHERE room_id=?",
            (*fields.values(), room_id),
        )

    async def all_active_rooms(self) -> list:
        rows = await self.fetchall(
            "SELECT * FROM rooms WHERE status IN ('active','paused','ready') AND group_id IS NOT NULL")
        return [dict(r) for r in rows]

    # ---- invites -----------------------------------------------------------
    async def create_invite(self, code: str, room_id: str, ttl_minutes: int) -> None:
        now = dt.datetime.utcnow()
        await self.execute(
            "INSERT OR REPLACE INTO invites(code, room_id, created_at, expires_at) VALUES(?,?,?,?)",
            (code, room_id, now.isoformat(),
             (now + dt.timedelta(minutes=ttl_minutes)).isoformat()),
        )

    async def get_valid_invite(self, code: str) -> Optional[dict]:
        row = await self.fetchone("SELECT * FROM invites WHERE code=?", (code,))
        if not row:
            return None
        inv = dict(row)
        if inv["used_by"] is not None:
            return None
        exp = dt.datetime.fromisoformat(inv["expires_at"])
        if dt.datetime.utcnow() > exp:
            return None
        return inv

    async def consume_invite(self, code: str, user_id: int) -> None:
        await self.execute(
            "UPDATE invites SET used_by=?, used_at=? WHERE code=?",
            (user_id, dt.datetime.utcnow().isoformat(), code),
        )

    # ---- sessions ----------------------------------------------------------
    async def save_session(self, user_id: int, encrypted: str) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO sessions(user_id, session_enc, linked_at) VALUES(?,?,?)",
            (user_id, encrypted, dt.datetime.utcnow().isoformat()),
        )

    async def get_session_enc(self, user_id: int) -> Optional[str]:
        row = await self.fetchone("SELECT session_enc FROM sessions WHERE user_id=?", (user_id,))
        return row["session_enc"] if row else None

    async def delete_session(self, user_id: int) -> None:
        await self.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))

    # ---- settings ----------------------------------------------------------
    async def get_settings(self, room_id: str) -> dict:
        row = await self.fetchone("SELECT * FROM settings WHERE room_id=?", (room_id,))
        return dict(row) if row else {}

    async def update_settings(self, room_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        await self.execute(
            f"UPDATE settings SET {sets} WHERE room_id=?", (*fields.values(), room_id))

    # ---- history -----------------------------------------------------------
    async def add_history(self, room_id: str, role: str, text: str,
                          tg_msg_id: Optional[int] = None) -> None:
        await self.execute(
            "INSERT INTO history(room_id, role, text, tg_msg_id, created_at) VALUES(?,?,?,?,?)",
            (room_id, role, text, tg_msg_id, dt.datetime.utcnow().isoformat()))

    async def get_history(self, room_id: str, limit: int) -> list:
        rows = await self.fetchall(
            "SELECT role, text FROM history WHERE room_id=? ORDER BY id DESC LIMIT ?",
            (room_id, limit))
        return list(reversed([dict(r) for r in rows]))

    async def clear_history(self, room_id: str) -> None:
        await self.execute("DELETE FROM history WHERE room_id=?", (room_id,))

    async def seen_before(self, room_id: str, tg_msg_id: int) -> bool:
        row = await self.fetchone(
            "SELECT 1 FROM seen_msgs WHERE room_id=? AND tg_msg_id=?", (room_id, tg_msg_id))
        return bool(row)

    async def mark_seen(self, room_id: str, tg_msg_id: int) -> None:
        await self.execute(
            "INSERT OR IGNORE INTO seen_msgs(room_id, tg_msg_id) VALUES(?,?)",
            (room_id, tg_msg_id))

    # ---- usage -------------------------------------------------------------
    async def incr_usage(self, user_id: int) -> int:
        day = dt.date.today().isoformat()
        await self.execute(
            "INSERT INTO usage_daily(user_id, day, count) VALUES(?,?,1) "
            "ON CONFLICT(user_id, day) DO UPDATE SET count=count+1", (user_id, day))
        row = await self.fetchone(
            "SELECT count FROM usage_daily WHERE user_id=? AND day=?", (user_id, day))
        return row["count"] if row else 0

    async def usage_today(self, user_id: int) -> int:
        row = await self.fetchone(
            "SELECT count FROM usage_daily WHERE user_id=? AND day=?",
            (user_id, dt.date.today().isoformat()))
        return row["count"] if row else 0

    # ---- stats -------------------------------------------------------------
    async def room_stats(self, room_id: str) -> dict:
        room = await self.get_room(room_id)
        row = await self.fetchone(
            "SELECT COUNT(*) AS n FROM history WHERE room_id=?", (room_id,))
        return {
            "messages": row["n"] if row else 0,
            "a": room["msg_count_a"] if room else 0,
            "b": room["msg_count_b"] if room else 0,
        }


# ----------------------------------------------------------------------------
# Room IDs / invitation codes
# ----------------------------------------------------------------------------
ROOM_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

def gen_room_id() -> str:
    suffix = "".join(random.choice(ROOM_ALPHABET) for _ in range(4))
    return f"AURA-{suffix}"

def gen_invite_code() -> str:
    return "".join(random.choice(ROOM_ALPHABET) for _ in range(8))

# ----------------------------------------------------------------------------
# Language guard — enforce English-only AI output
# ----------------------------------------------------------------------------
# Detects meaningful use of non-Latin scripts (Devanagari, Arabic, CJK,
# Cyrillic, etc.). Emoji, punctuation, and digits are ignored.
_NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F"
    r"\u0E00-\u0E7F\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")

# Common Hinglish transliteration markers ("yaar", "kya", "hai", ...)
_HINGLISH_MARKERS = re.compile(
    r"\b(yaar|yaar|kya|kyu|kyun|hai|hain|nahi|nahin|acha|achha|theek|matlab"
    r"|bhai|bro|bkl|mujhe|tumhe|karo|karna|kar|raha|rahi|hogaya|ho+ ?gaya"
    r"|chal|chalo|dekh|sun|suno|abhi|kal|aaj|bahut|bohot|thoda|zada|zyada"
    r"|haan|haa|nhi| nhi|samjha|samjh|bata|batao|chal+ ?ra)\b", re.IGNORECASE)


def is_english(text: str) -> bool:
    """True if the text is English (Latin script, no Hinglish/Hindi markers)."""
    if not text:
        return False
    if _NON_LATIN.search(text):
        return False
    if _HINGLISH_MARKERS.search(text):
        return False
    return True

# ----------------------------------------------------------------------------
# AI engine (OpenAI-compatible chat completions) — logical, English-only
# ----------------------------------------------------------------------------
class AIEngine:
    ENGLISH_RULE = (
        "LANGUAGE RULE (STRICT): You MUST write ONLY in clean, natural, "
        "grammatically correct English. NEVER use Hindi, Hinglish, Roman Hindi, "
        "or any other language. Never mix languages, never use transliterated "
        "Hindi words (like yaar, kya, hai, matlab, bhai, chal, acha), never "
        "code-switch. Pure English only, every single message."
    )

    PERSONALITY_A = (
        "You are Participant A in a realistic two-person Telegram conversation between two "
        "thoughtful humans.\n"
        "Your role:\n"
        "- Initiate topics and introduce fresh angles as the discussion evolves.\n"
        "- Ask thoughtful, specific questions that push the discussion deeper.\n"
        "- Respond directly to B's most recent message before adding anything new.\n"
        "- Build logically on the conversation: reference earlier points when relevant, "
        "challenge weak reasoning politely, and give concrete examples or reasoning for "
        "your opinions — never vague filler.\n"
        "- Keep the thread coherent; do not randomly change subjects mid-flow.\n"
        "Style: natural, grammatically correct conversational English, 2-4 sentences, "
        "no hashtags, no signatures, no role labels, never mention you are an AI.\n"
    ) + ENGLISH_RULE

    PERSONALITY_B = (
        "You are Participant B in a realistic two-person Telegram conversation between two "
        "thoughtful humans.\n"
        "Your role:\n"
        "- Respond to A's most recent message with a direct, substantive reply.\n"
        "- Give independent, well-reasoned opinions with clear logic or examples.\n"
        "- Ask relevant follow-up questions that move the discussion forward.\n"
        "- Disagree respectfully when you genuinely think A is wrong, and say why.\n"
        "- Build on earlier points in the conversation; keep continuity.\n"
        "Style: natural, grammatically correct conversational English, 2-4 sentences, "
        "no hashtags, no signatures, no role labels, never mention you are an AI.\n"
    ) + ENGLISH_RULE

    def __init__(self):
        self._http: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=CFG.AI_TIMEOUT),
            headers={"Authorization": f"Bearer {CFG.AI_API_KEY}"},
        )

    async def close(self) -> None:
        if self._http:
            await self._http.close()

    async def chat(self, system: str, history: list[dict], user_prompt: str) -> str:
        if not CFG.AI_API_KEY or not CFG.ai_base:
            raise RuntimeError("AI provider not configured (AI_API_KEY / AI_PROVIDER).")
        messages = [{"role": "system", "content": system}]
        # history roles map to the two conversation sides
        for h in history:
            messages.append({"role": "user" if h["role"] == "B" else "assistant",
                             "content": h["text"]})
        messages.append({"role": "user", "content": user_prompt})
        payload = {"model": CFG.ai_model, "messages": messages,
                   "max_tokens": 350, "temperature": 0.9,
                   "presence_penalty": 0.6, "frequency_penalty": 0.4}

        last_err: Optional[Exception] = None
        for attempt in range(1, CFG.MAX_AI_RETRIES + 1):
            try:
                async with self._http.post(
                    f"{CFG.ai_base.rstrip('/')}/chat/completions", json=payload
                ) as resp:
                    if resp.status == 429:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status)
                    resp.raise_for_status()
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    return re.sub(r"^(Participant [AB]:\s*)+", "", text)
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError) as e:
                last_err = e
                await asyncio.sleep(2 ** attempt)  # bounded exponential backoff
        raise RuntimeError(f"AI request failed after {CFG.MAX_AI_RETRIES} retries: {last_err}")


# ----------------------------------------------------------------------------
# Keyboards
# ----------------------------------------------------------------------------
class K:
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 CREATE ROOM", callback_data="room:create"),
             InlineKeyboardButton("🤝 JOIN ROOM", callback_data="room:join")],
            [InlineKeyboardButton("👤 MY ROOMS", callback_data="room:mine"),
             InlineKeyboardButton("📊 STATISTICS", callback_data="room:stats")],
            [InlineKeyboardButton("⚙️ SETTINGS", callback_data="set:global"),
             InlineKeyboardButton("❓ HELP", callback_data="help")],
        ])

    @staticmethod
    def room_waiting(room_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📨 INVITE PARTNER", callback_data=f"room:invite:{room_id}")],
            [InlineKeyboardButton("⚙️ ROOM SETTINGS", callback_data=f"set:room:{room_id}"),
             InlineKeyboardButton("❌ CANCEL ROOM", callback_data=f"room:cancel:{room_id}")],
            [InlineKeyboardButton("⬅️ BACK", callback_data="home")],
        ])

    @staticmethod
    def linking(room_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 REFRESH STATUS", callback_data=f"room:refresh:{room_id}")],
            [InlineKeyboardButton("🔗 HOW TO CONNECT", callback_data="link:how")],
            [InlineKeyboardButton("⬅️ BACK", callback_data="home")],
        ])

    @staticmethod
    def group_config(room_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("➕ ADD GROUP", callback_data=f"grp:how:{room_id}")],
            [InlineKeyboardButton("🔍 VERIFY GROUP", callback_data=f"grp:verify:{room_id}")],
            [InlineKeyboardButton("🔒 LOCK ROOM", callback_data=f"grp:lock:{room_id}"),
             InlineKeyboardButton("⬅️ BACK", callback_data=f"room:refresh:{room_id}")],
        ])

    @staticmethod
    def control_panel(room_id: str, active: bool, paused: bool) -> InlineKeyboardMarkup:
        rows = []
        if paused:
            rows.append([InlineKeyboardButton("▶️ RESUME", callback_data=f"ctl:resume:{room_id}")])
        else:
            rows.append([InlineKeyboardButton("▶️ START", callback_data=f"ctl:start:{room_id}")])
            if active:
                rows.append([InlineKeyboardButton("⏸️ PAUSE", callback_data=f"ctl:pause:{room_id}")])
        rows.append([InlineKeyboardButton("⏹️ STOP", callback_data=f"ctl:stop:{room_id}")])
        rows.append([InlineKeyboardButton("📊 STATISTICS", callback_data=f"room:statsroom:{room_id}"),
                     InlineKeyboardButton("⚙️ SETTINGS", callback_data=f"set:room:{room_id}")])
        rows.append([InlineKeyboardButton("⬅️ BACK", callback_data="home")])
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def creator_panel(room_id: str) -> InlineKeyboardMarkup:
        """Creator-only extras shown under the control panel."""
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🔓 UNLOCK ROOM", callback_data=f"grp:unlock:{room_id}")],
            [InlineKeyboardButton("⬅️ BACK", callback_data="home")],
        ])

    @staticmethod
    def settings_menu(room_id: Optional[str]) -> InlineKeyboardMarkup:
        p = f"set:room:{room_id}" if room_id else "set:global"
        prefix = f"setc:{room_id or 'g'}"
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 Topic", callback_data=f"{prefix}:topic"),
             InlineKeyboardButton("🎭 Personality", callback_data=f"{prefix}:persona")],
            [InlineKeyboardButton("⏱️ Interval", callback_data=f"{prefix}:interval"),
             InlineKeyboardButton("✉️ Max messages", callback_data=f"{prefix}:maxmsg")],
            [InlineKeyboardButton("⌛ Session length", callback_data=f"{prefix}:minutes"),
             InlineKeyboardButton("📏 Max length", callback_data=f"{prefix}:maxlen")],
            [InlineKeyboardButton("🧠 History length", callback_data=f"{prefix}:histlen"),
             InlineKeyboardButton("@️ Mention policy", callback_data=f"{prefix}:mention")],
            [InlineKeyboardButton("🚦 Daily limit", callback_data=f"{prefix}:daily")],
            [InlineKeyboardButton("♻️ Reset conversation", callback_data=f"{prefix}:resethist")],
            [InlineKeyboardButton("⬅️ BACK", callback_data=p)],
        ])

    @staticmethod
    def close_only() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("⬅️ BACK", callback_data="home")]])


# ----------------------------------------------------------------------------
# UI text helpers
# ----------------------------------------------------------------------------
def room_status_text(room: dict, settings: dict, usernames: dict[int, str]) -> str:
    creator = "@" + usernames[room["creator_id"]] if room["creator_id"] in usernames \
        else str(room["creator_id"])
    partner = ("@" + usernames[room["partner_id"]]) if room["partner_id"] and room["partner_id"] in usernames \
        else (str(room["partner_id"]) if room["partner_id"] else "Waiting")
    grp = "Configured" if room["group_id"] else "Not Configured"
    lock = "🔒 Locked" if room["locked"] else "🔓 Unlocked"
    return (
        f"🏠 ROOM `{room['room_id']}`\n\n"
        f"👤 Creator: {creator}\n"
        f"👥 Partner: {partner}\n"
        f"💬 Group: {grp}\n"
        f"🔒 Lock: {lock}\n"
        f"📌 Status: {STATUS_LABEL.get(room['status'], room['status'])}\n\n"
        f"💬 Topic: {settings.get('topic', '—')}\n"
        f"⏱️ Interval: {settings.get('interval', '-')}s · ✉️ Max: {settings.get('max_messages', '-')}\n"
        f"⌛ Session: {settings.get('session_minutes', '-')}min · @️ Mentions: {settings.get('mention_policy', '-')}"
    )


def control_panel_text(room: dict, settings: dict, usernames: dict[int, str]) -> str:
    creator = "@" + usernames[room["creator_id"]] if room["creator_id"] in usernames \
        else str(room["creator_id"])
    partner = ("@" + usernames[room["partner_id"]]) if room["partner_id"] and room["partner_id"] in usernames \
        else str(room["partner_id"])
    return (
        f"🤖 AI CONVERSATION ROOM\n\n"
        f"Room: `{room['room_id']}`\n"
        f"Status: {STATUS_LABEL.get(room['status'], room['status'])}\n\n"
        f"Participant A: {creator}\n"
        f"Participant B: {partner}\n"
        f"Target Group: {'Verified ✅' if room['group_id'] else 'Not set ❌'}\n\n"
        f"Language: English\n"
        f"Mode: {room.get('mode', 'Topic-Based').title()}\n"
        f"Topic: {settings.get('topic', '—')}\n\n"
        f"_Both participants can START / PAUSE / RESUME / STOP.\n"
        f"STOP then START continues the same discussion. Manual replies are "
        f"picked up naturally by the AI._"
    )


# ----------------------------------------------------------------------------
# Account connection (Telethon) — userbot manager + interactive linking
# ----------------------------------------------------------------------------
class UserbotManager:
    """Runs one Telethon client per connected (authorized) user account."""

    def __init__(self, db: Database):
        self.db = db
        self.clients: dict[int, TelegramClient] = {}
        self._group_msg_lock = asyncio.Lock()

    async def start_all(self) -> None:
        if not CFG.fernet:
            log.warning("SESSION_ENCRYPTION_KEY missing — stored sessions cannot be decrypted.")
            return
        rows = await self.db.fetchall("SELECT user_id FROM sessions")
        for row in rows:
            uid = row["user_id"]
            try:
                await self.start_client(uid)
            except InvalidToken:
                log.error("Cannot decrypt session for user %s (wrong key).", uid)
            except Exception as e:
                log.error("Failed to start userbot for %s: %s", uid, type(e).__name__)

    async def start_client(self, user_id: int) -> TelegramClient:
        if user_id in self.clients and self.clients[user_id].is_connected():
            return self.clients[user_id]
        enc = await self.db.get_session_enc(user_id)
        if not enc:
            raise RuntimeError("no session")
        raw = CFG.fernet.decrypt(enc.encode()).decode()
        client = TelegramClient(
            StringSession(raw), CFG.API_ID, CFG.API_HASH,
            device_model="AIConvRooms", system_version="1.0", auto_reconnect=True,
        )
        await client.connect()
        if not await client.is_user_authorized():
            raise AuthKeyUnregisteredError(None)
        client.add_event_handler(self._on_group_message, events.NewMessage(chats=None))
        self.clients[user_id] = client
        log.info("Userbot started for user %s", user_id)
        return client

    async def stop_all(self) -> None:
        for uid, client in list(self.clients.items()):
            with contextlib.suppress(Exception):
                await client.disconnect()
            self.clients.pop(uid, None)

    async def disconnect_user(self, user_id: int) -> None:
        client = self.clients.pop(user_id, None)
        if client:
            with contextlib.suppress(Exception):
                await client.disconnect()

    async def send_as_user(self, user_id: int, group_id: int, text: str) -> int:
        """Send a message through the participant's authorized account. Returns message id."""
        client = self.clients.get(user_id) or await self.start_client(user_id)
        try:
            msg = await client.send_message(int(group_id), text)
            return msg.id
        except FloodWaitError as e:
            log.warning("FloodWait %ss sending as %s; sleeping", e.seconds, user_id)
            await asyncio.sleep(e.seconds + 1)
            msg = await client.send_message(int(group_id), text)
            return msg.id

    async def _on_group_message(self, event: events.NewMessage.Event) -> None:
        """Ingest participant messages from configured target groups into room history."""
        if event.out:  # never treat our own outgoing message as incoming
            return
        chat_id = event.chat_id
        sender_id = event.sender_id
        if sender_id is None or not event.message.raw_text:
            return
        async with self._group_msg_lock:
            for room in await self.db.all_active_rooms():
                if room["group_id"] != chat_id:
                    continue  # ignore unrelated groups / cross-room leakage
                if sender_id not in (room["creator_id"], room["partner_id"] or 0):
                    continue  # ignore unauthorized users
                if await self.db.seen_before(room["room_id"], event.message.id):
                    continue  # duplicate processing guard
                await self.db.mark_seen(room["room_id"], event.message.id)
                role = "A" if sender_id == room["creator_id"] else "B"
                await self.db.add_history(room["room_id"], role, event.message.raw_text,
                                          event.message.id)


async def interactive_link(user_id: int, db: Database) -> None:
    """Terminal-only authorization. OTP/2FA are read via getpass and never stored/logged."""
    if not CFG.fernet:
        print("Set SESSION_ENCRYPTION_KEY (a Fernet key) before linking accounts.")
        print('Generate one with: python -c "from cryptography.fernet import Fernet; '
              'print(Fernet.generate_key().decode())"')
        return
    print(f"Linking Telegram account for user id {user_id}.")
    print("Use a session ONLY for the account that owns this Telegram user id.")
    phone = input("Phone number (international format): ").strip()
    client = TelegramClient(StringSession(), CFG.API_ID, CFG.API_HASH)
    await client.connect()
    try:
        sent = await client.send_code_request(phone)
        code = getpass.getpass("Telegram login code (hidden input): ").strip()
        try:
            await client.sign_in(phone, code, phone_code_hash=sent.phone_code_hash)
        except SessionPasswordNeededError:
            pwd = getpass.getpass("Two-step verification password (hidden input): ")
            await client.sign_in(password=pwd)
        session_str = client.session.save()
        encrypted = CFG.fernet.encrypt(session_str.encode()).decode()
        await db.save_session(user_id, encrypted)
        me = await client.get_me()
        await db.upsert_user(user_id, me.username)
        print(f"✅ Account linked and encrypted session stored for user id {user_id}.")
        if me.id != user_id:
            print(f"⚠️ WARNING: linked account id is {me.id}, not {user_id}. "
                  "Store the session under the correct user id or re-run with the right ID.")
    except (PhoneCodeInvalidError, PhoneCodeExpiredError):
        print("❌ Invalid or expired code. Nothing was stored.")
    finally:
        await client.disconnect()


# ----------------------------------------------------------------------------
# Conversation engine / task management
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class RunningTask:
    task: asyncio.Task
    pause_event: asyncio.Event     # set = running, cleared = paused
    stop_flag: bool = False


class ConversationManager:
    def __init__(self, db: Database, ai: AIEngine, bots: UserbotManager,
                 notify: "Handlers"):
        self.db = db
        self.ai = ai
        self.bots = bots
        self.notify = notify
        self.running: dict[str, RunningTask] = {}

    # ---- control (creator OR partner) ---------------------------------------
    async def start(self, room_id: str) -> tuple[bool, str]:
        if room_id in self.running and not self.running[room_id].task.done():
            return False, "A conversation task is already running for this room."
        room = await self.db.get_room(room_id)
        if not room or not room["group_id"] or not room["locked"]:
            return False, "Room is not locked/verified. Configure and lock it first."
        for uid in (room["creator_id"], room["partner_id"]):
            enc = await self.db.get_session_enc(uid or 0)
            if not enc:
                return False, "Both participants must connect their accounts first."

        # Fresh start vs continuation: existing history means CONTINUE from
        # where the conversation was stopped.
        row = await self.db.fetchone(
            "SELECT COUNT(*) n FROM history WHERE room_id=?", (room_id,))
        existing = row["n"] if row else 0

        if existing == 0:
            await self.db.execute("DELETE FROM seen_msgs WHERE room_id=?", (room_id,))
            await self.db.update_room(
                room_id, status="active",
                started_at=dt.datetime.utcnow().isoformat(),
                stopped_at=None, msg_count_a=0, msg_count_b=0)
            resumed = False
        else:
            await self.db.update_room(
                room_id, status="active",
                started_at=room.get("started_at") or dt.datetime.utcnow().isoformat(),
                stopped_at=None)
            resumed = True

        pause = asyncio.Event(); pause.set()
        task = asyncio.create_task(self._run(room_id))
        self.running[room_id] = RunningTask(task=task, pause_event=pause)
        return True, ("Continuing the conversation from where it stopped."
                      if resumed else "Conversation started.")

    async def pause(self, room_id: str) -> tuple[bool, str]:
        rt = self.running.get(room_id)
        room = await self.db.get_room(room_id)
        if room and room["status"] == "active":
            await self.db.update_room(room_id, status="paused")
        if not rt or rt.task.done():
            return False, "No active conversation task for this room."
        rt.pause_event.clear()
        return True, "Conversation paused. History preserved."

    async def resume(self, room_id: str) -> tuple[bool, str]:
        rt = self.running.get(room_id)
        if not rt or rt.task.done():
            return False, "No pausable conversation task for this room."
        room = await self.db.get_room(room_id)
        if not room or room["status"] != "paused":
            return False, "Room is not paused."
        await self.db.update_room(room_id, status="active")
        rt.pause_event.set()
        return True, "Conversation resumed."

    async def stop(self, room_id: str) -> tuple[bool, str]:
        rt = self.running.get(room_id)
        if rt:
            rt.stop_flag = True
            rt.pause_event.set()
            rt.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rt.task
            self.running.pop(room_id, None)
        await self.db.update_room(
            room_id, status="stopped", stopped_at=dt.datetime.utcnow().isoformat())
        return True, ("Conversation stopped. History preserved — "
                      "START will continue from where you left off.")

    # ---- loop --------------------------------------------------------------
    async def _run(self, room_id: str) -> None:
        rt = self.running[room_id]
        room = await self.db.get_room(room_id)
        settings = await self.db.get_settings(room_id)
        usernames = await self.notify.resolve_usernames([room["creator_id"], room["partner_id"]])
        username_b = usernames.get(room["partner_id"])
        deadline = time.time() + settings["session_minutes"] * 60
        sent_this_session = 0
        recent_outputs: list[str] = []

        while sent_this_session < settings["max_messages"] and time.time() < deadline:
            await rt.pause_event.wait()  # pause gate (history preserved)
            if rt.stop_flag or room_id not in self.running:
                return
            room = await self.db.get_room(room_id)
            if not room or room["status"] not in ("active", "paused"):
                return

            # --- adaptive turn order -----------------------------------------
            # Next speaker = whoever did NOT send the last message in history.
            # Works for both AI messages and manual human replies, so manual
            # participation never breaks the alternating flow.
            history = await self.db.get_history(room_id, settings["history_len"])
            if not history:
                role = "A"  # fresh conversation: A opens
            else:
                role = "B" if history[-1]["role"] == "A" else "A"

            sender_id = room["creator_id"] if role == "A" else room["partner_id"]
            if not sender_id:
                return
            if await self.db.usage_today(sender_id) >= settings["daily_limit"]:
                await self.notify.send_room_event(
                    room_id, "🚦 Daily message limit reached. Conversation paused.")
                await self.pause(room_id)
                return

            last_a = next((h["text"] for h in reversed(history) if h["role"] == "A"), None)
            last_b = next((h["text"] for h in reversed(history) if h["role"] == "B"), None)

            if role == "A":
                system = self.ai.PERSONALITY_A + f"\nDiscussion topic: {settings['topic']}"
                if not history:
                    prompt = ("Write the conversation opener. "
                              + (f"Naturally mention Participant B once as @{username_b}. "
                                 if settings["mention_policy"] != "never" and username_b
                                 else "Do not use @mentions. ")
                              + "Introduce the topic with a specific, thought-provoking angle "
                                "and end with one thoughtful question. English only.")
                else:
                    prompt = ("Continue the existing discussion from the conversation history. "
                              + (f"Respond to B's last message:\n\"{last_b or '...'}\"\n\n"
                                 if last_b else "\n")
                              + "Move the discussion forward with a new angle or deeper point. "
                                "Do not restart the topic from scratch."
                              + (f" You may naturally mention @{username_b} if it fits."
                                 if settings["mention_policy"] == "always" and username_b
                                 else " Do not use @mentions.") + " English only.")
            else:
                system = self.ai.PERSONALITY_B + f"\nDiscussion topic: {settings['topic']}"
                prompt = ("Reply to Participant A's last message:\n"
                          f"\"{last_a or '...'}\"\n\n"
                          "Give your own reasoned opinion with a concrete justification "
                          "(logic, example, or trade-off), and ask one relevant follow-up "
                          "question. Do not use @mentions. English only.")

            try:
                text = await self.ai.chat(system, history, prompt)
            except Exception as e:
                log.error("AI generation failed for room %s: %s", room_id, type(e).__name__)
                await self.notify.send_room_event(
                    room_id, "⚠️ AI provider error — retrying shortly.")
                await asyncio.sleep(15)
                continue

            # Language guard: reject/regenerate non-English output once.
            if not is_english(text):
                with contextlib.suppress(Exception):
                    text = await self.ai.chat(
                        system, history,
                        prompt + "\nSTRICT REMINDER: your previous attempt was NOT in pure "
                                 "English. Respond ONLY in clean English — no Hindi, no "
                                 "Hinglish, no other language.")
            if not is_english(text):
                log.warning("Non-English output suppressed in room %s", room_id)
                await asyncio.sleep(5)
                continue

            # anti-repetition: regenerate once if too similar to a recent output
            if any(_similar(text, prev) for prev in recent_outputs[-4:]):
                with contextlib.suppress(Exception):
                    text = await self.ai.chat(system, history, prompt +
                                              "\nIMPORTANT: avoid repeating earlier phrasing; "
                                              "take a genuinely fresh angle. English only.")
            recent_outputs.append(text)

            try:
                msg_id = await self.bots.send_as_user(sender_id, room["group_id"], text)
            except AuthKeyUnregisteredError:
                await self.notify.send_room_event(
                    room_id, "⚠️ A participant's session is no longer authorized. Conversation stopped.")
                await self.stop(room_id)
                return
            except FloodWaitError as e:
                await asyncio.sleep(min(e.seconds, 60))
                continue

            await self.db.add_history(room_id, role, text, msg_id)
            await self.db.incr_usage(sender_id)
            if role == "A":
                await self.db.update_room(room_id, msg_count_a=room["msg_count_a"] + 1)
            else:
                await self.db.update_room(room_id, msg_count_b=room["msg_count_b"] + 1)
            sent_this_session += 1
            await asyncio.sleep(settings["interval"])


def _similar(a: str, b: str) -> bool:
    a_w, b_w = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not a_w or not b_w:
        return False
    return len(a_w & b_w) / len(a_w | b_w) > 0.7


# ----------------------------------------------------------------------------
# Bot handlers (Kurigram)
# ----------------------------------------------------------------------------
class Handlers:
    def __init__(self, db: Database, ai: AIEngine, conv: Optional[ConversationManager],
                 bots: Optional[UserbotManager]):
        self.db = db
        self.ai = ai
        self.conv = conv
        self.bots = bots
        self.bot: Optional[Client] = None
        self.pending_group: dict[int, str] = {}  # creator_id -> room_id awaiting /setgroup

    # ---- utility -----------------------------------------------------------
    async def resolve_usernames(self, ids: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        for uid in filter(None, ids):
            try:
                u = await self.bot.get_users(uid)
                out[uid] = u.username or str(uid)
            except pyro_errors.RPCError:
                out[uid] = str(uid)
        return out

    async def send_room_event(self, room_id: str, text: str) -> None:
        room = await self.db.get_room(room_id)
        if not room:
            return
        for uid in filter(None, {room["creator_id"], room["partner_id"]}):
            with contextlib.suppress(pyro_errors.RPCError, FloodWaitError):
                await self.bot.send_message(uid, f"📢 {text}")

    async def is_room_participant(self, user_id: int, room_id: str) -> Optional[dict]:
        room = await self.db.get_room(room_id)
        if room and user_id in (room["creator_id"], room["partner_id"] or 0):
            return room
        return None

    async def refresh_room_panel(self, room_id: str, chat_id: int) -> None:
        room = await self.db.get_room(room_id)
        settings = await self.db.get_settings(room_id)
        usernames = await self.resolve_usernames([room["creator_id"], room["partner_id"]])
        if room["status"] in ("ready", "active", "paused"):
            text = control_panel_text(room, settings, usernames)
            kb = K.control_panel(room_id, room["status"] == "active", room["status"] == "paused")
            await self.bot.send_message(chat_id, text, reply_markup=kb)
            # creator-only extras (unlock) as a separate small panel
            if room["creator_id"] == chat_id:
                await self.bot.send_message(chat_id, "👑 Creator controls:",
                                            reply_markup=K.creator_panel(room_id))
        else:
            text = room_status_text(room, settings, usernames)
            kb = {"waiting": K.room_waiting(room_id), "linking": K.linking(room_id),
                  "config": K.group_config(room_id)}.get(room["status"], K.close_only())
            await self.bot.send_message(chat_id, text, reply_markup=kb)

    # ---- /start ------------------------------------------------------------
    async def cmd_start(self, client: Client, message: Message) -> None:
        self.bot = client
        payload = (message.command[1] if len(message.command) > 1 else "").strip()
        await self.db.upsert_user(message.from_user.id, message.from_user.username)

        if payload.startswith("join_"):
            await self._accept_invite(client, message, payload[5:])
            return

        await message.reply_text(
            "🤖 **AI CONVERSATION ROOMS**\n\n"
            "Create a room, invite your partner, connect your accounts, and "
            "configure an AI conversation between your two authorized accounts.\n\n"
            "➤ Use the buttons below to navigate.",
            reply_markup=K.main_menu(),
        )

    async def _accept_invite(self, client: Client, message: Message, code: str) -> None:
        inv = await self.db.get_valid_invite(code)
        if not inv:
            await message.reply_text("❌ Invalid or expired invitation code.",
                                     reply_markup=K.close_only())
            return
        room = await self.db.get_room(inv["room_id"])
        if not room or room["status"] != "waiting":
            await message.reply_text("❌ This room is no longer accepting a partner.",
                                     reply_markup=K.close_only())
            return
        if message.from_user.id == room["creator_id"]:
            await message.reply_text("❌ You cannot join your own room as a partner.",
                                     reply_markup=K.close_only())
            return
        if room["partner_id"]:
            await message.reply_text("❌ This room already has two participants.",
                                     reply_markup=K.close_only())
            return

        await self.db.consume_invite(code, message.from_user.id)
        await self.db.upsert_user(message.from_user.id, message.from_user.username)
        await self.db.update_room(inv["room_id"], partner_id=message.from_user.id,
                                  status="linking")
        await message.reply_text(
            f"✅ You joined room `{inv['room_id']}` as **Participant B**.\n\n"
            "Next step: both participants must connect their Telegram accounts.\n"
            "Press HOW TO CONNECT for secure authorization instructions.",
            reply_markup=K.linking(inv["room_id"]))
        await self.send_room_event(
            inv["room_id"],
            f"👥 @{message.from_user.username or message.from_user.id} joined your room "
            f"`{inv['room_id']}` as Participant B.")
        await self._maybe_ready_for_config(inv["room_id"])

    async def _maybe_ready_for_config(self, room_id: str) -> None:
        """Move to 'config' when both participants have authorized sessions."""
        room = await self.db.get_room(room_id)
        if not room or not room["partner_id"]:
            return
        for uid in (room["creator_id"], room["partner_id"]):
            if not await self.db.get_session_enc(uid):
                return
        await self.db.update_room(room_id, status="config")
        await self.send_room_event(
            room_id, "🔗 Both accounts connected! Creator: configure the target group.")
        await self.refresh_room_panel(room_id, room["creator_id"])

    # ---- group command: /setgroup (creator only) -----------------------------
    async def cmd_setgroup(self, client: Client, message: Message) -> None:
        uid = message.from_user.id if message.from_user else 0
        room_id = self.pending_group.get(uid)
        if not room_id:
            return
        room = await self.db.get_room(room_id)
        if not room or room["creator_id"] != uid or room["status"] != "config":
            return
        await self._verify_group(client, message.chat.id, room_id)

    async def _verify_group(self, client: Client, chat_id: int, room_id: str) -> None:
        room = await self.db.get_room(room_id)
        if not room:
            return
        try:
            member = await client.get_chat_member(chat_id, "me")
            can_post = getattr(member, "can_send_messages", False) or \
                member.status == enums.ChatMemberStatus.ADMINISTRATOR or \
                member.status == enums.ChatMemberStatus.OWNER
            if not can_post:
                raise pyro_errors.UserRestriction(None)
            test = await client.send_message(chat_id, "✅ Group verified for this AI room.")
            await test.delete()
        except pyro_errors.FloodWait as e:
            await asyncio.sleep(e.value)
            await self.send_room_event(room_id, "⏳ Rate limited while verifying. Try again.")
            return
        except pyro_errors.RPCError as e:
            await self.send_room_event(
                room_id, f"❌ Group verification failed: missing messaging permissions "
                         f"({type(e).__name__}). Add the bot and check its rights.")
            return
        await self.db.update_room(room_id, group_id=chat_id)
        self.pending_group.pop(room["creator_id"], None)
        await self.send_room_event(
            room_id, f"🔍 Group `{chat_id}` verified and configured. You may now lock the room.")
        await self.refresh_room_panel(room_id, room["creator_id"])

    # ---- callback router ----------------------------------------------------
    async def on_callback(self, client: Client, query: CallbackQuery) -> None:
        self.bot = client
        uid = query.from_user.id
        try:
            await self._route(client, query, uid, query.data)
        except pyro_errors.FloodWait as e:
            await query.answer(f"⏳ Rate limited. Retry in {e.value}s.", show_alert=True)
        except aiosqlite.Error:
            log.exception("db error")
            with contextlib.suppress(Exception):
                await query.answer("❌ Database error. Try again.", show_alert=True)
        except Exception:
            log.exception("callback error")
            with contextlib.suppress(Exception):
                await query.answer("❌ Unexpected error.", show_alert=True)

    async def _route(self, client: Client, query: CallbackQuery, uid: int, data: str) -> None:
        await self.db.upsert_user(uid, query.from_user.username)
        parts = data.split(":")
        kind, arg = parts[0], parts[1] if len(parts) > 1 else ""
        arg2 = parts[2] if len(parts) > 2 else ""

        if data == "home":
            await query.answer()
            await query.message.edit_text(
                "🤖 **AI CONVERSATION ROOMS**\n\nMain menu.",
                reply_markup=K.main_menu())
            return
        if data == "help":
            await query.answer()
            await self._show_help(query)
            return
        if data == "link:how":
            await query.answer()
            await query.message.reply_text(
                _LINK_HELP, reply_markup=K.close_only(), parse_mode=enums.ParseMode.MARKDOWN)
            return

        if kind == "room":
            await self._route_room(client, query, uid, arg, arg2)
        elif kind == "grp":
            await self._route_group(client, query, uid, arg, arg2)
        elif kind == "ctl":
            await self._route_control(client, query, uid, arg, arg2)
        elif kind == "set":
            await self._route_settings_view(query, uid, arg)
        elif kind == "setc":
            await self._route_settings_cycle(query, uid, arg, arg2)

    # ---- room routes ---------------------------------------------------------
    async def _route_room(self, client: Client, query: CallbackQuery, uid: int,
                          arg: str, room_id: str) -> None:
        await query.answer()
        if arg == "create":
            existing = await self.db.get_room_by_creator(uid)
            if existing:
                await self.refresh_room_panel(existing["room_id"], query.message.chat.id)
                return
            room_id_new = gen_room_id()
            while await self.db.get_room(room_id_new):
                room_id_new = gen_room_id()
            await self.db.create_room(room_id_new, uid)
            await self.refresh_room_panel(room_id_new, query.message.chat.id)
            return
        if arg == "join":
            await query.message.edit_text(
                "🤝 **JOIN ROOM**\n\nAsk the room creator for an invitation link, or send "
                "the invitation code as a message here (e.g. `XK29QW83`).",
                reply_markup=K.close_only())
            return
        if arg == "mine":
            rows = await self.db.fetchall(
                "SELECT * FROM rooms WHERE creator_id=? OR partner_id=? ORDER BY created_at DESC LIMIT 10",
                (uid, uid))
            if not rows:
                await query.message.edit_text("👤 You have no rooms yet.",
                                              reply_markup=K.main_menu())
                return
            lines = "\n".join(
                f"• `{r['room_id']}` — {STATUS_LABEL.get(r['status'], r['status'])}"
                + (" (creator)" if r["creator_id"] == uid else " (partner)")
                for r in rows)
            await query.message.edit_text(f"👤 **MY ROOMS**\n\n{lines}",
                                          reply_markup=K.main_menu())
            return
        if arg == "stats":
            total = await self.db.fetchone("SELECT COUNT(*) n FROM rooms")
            msgs = await self.db.fetchone("SELECT COUNT(*) n FROM history")
            mine = await self.db.fetchone(
                "SELECT COUNT(*) n FROM rooms WHERE creator_id=? OR partner_id=?", (uid, uid))
            await query.message.edit_text(
                "📊 **STATISTICS**\n\n"
                f"Total rooms: {total['n']}\n"
                f"Your rooms: {mine['n']}\n"
                f"Messages exchanged: {msgs['n']}",
                reply_markup=K.main_menu())
            return
        if arg == "statsroom":
            room = await self.is_room_participant(uid, room_id)
            if not room:
                await query.answer("⛔ Not your room.", show_alert=True)
                return
            st = await self.db.room_stats(room_id)
            dur = ""
            if room["started_at"]:
                start = dt.datetime.fromisoformat(room["started_at"])
                end = dt.datetime.fromisoformat(room["stopped_at"]) if room["stopped_at"] \
                    else dt.datetime.utcnow()
                dur = f"\nSession duration: {int((end-start).total_seconds()//60)} min"
            await query.message.reply_text(
                f"📊 **Room {room_id}**\n\nMessages: {st['messages']}\n"
                f"Participant A: {st['a']}\nParticipant B: {st['b']}{dur}",
                reply_markup=K.close_only())
            return
        if arg == "invite":
            room = await self.db.get_room(room_id)
            if not room or room["creator_id"] != uid:
                await query.answer("⛔ Only the creator may invite a partner.", show_alert=True)
                return
            code = gen_invite_code()
            await self.db.create_invite(code, room_id, CFG.INVITE_TTL_MINUTES)
            me = await client.get_me()
            link = f"https://t.me/{me.username}?start=join_{code}"
            await query.message.reply_text(
                f"📨 **Invitation created**\n\nRoom: `{room_id}`\nCode: `{code}`\n"
                f"Expires in {CFG.INVITE_TTL_MINUTES} minutes.\n\n"
                f"Share this link:\n{link}\n\n"
                "Only one partner can accept it.",
                reply_markup=K.room_waiting(room_id))
            return
        if arg == "cancel":
            room = await self.db.get_room(room_id)
            if not room or room["creator_id"] != uid:
                await query.answer("⛔ Only the creator may cancel the room.", show_alert=True)
                return
            await self.conv.stop(room_id)  # stops task if running
            await self.db.update_room(room_id, status="cancelled")
            await query.message.edit_text("❌ Room cancelled.", reply_markup=K.main_menu())
            return
        if arg == "refresh":
            room = await self.is_room_participant(uid, room_id)
            if not room:
                await query.answer("⛔ Not your room.", show_alert=True)
                return
            await self.refresh_room_panel(room_id, query.message.chat.id)
            return

    # ---- group routes (creator only) ------------------------------------------
    async def _route_group(self, client: Client, query: CallbackQuery, uid: int,
                           arg: str, room_id: str) -> None:
        room = await self.db.get_room(room_id)
        if not room:
            await query.answer("Room not found.", show_alert=True)
            return
        if room["creator_id"] != uid:
            await query.answer("⛔ Only the creator may configure/lock/unlock the room.",
                               show_alert=True)
            return
        if arg == "how":
            await query.answer()
            self.pending_group[uid] = room_id
            await query.message.reply_text(
                "➕ **ADD GROUP**\n\n"
                "1. Add this bot to your target group (with send permission).\n"
                "2. In that group, the creator sends: `/setgroup`\n"
                "3. The bot verifies access and messaging permissions.",
                reply_markup=K.group_config(room_id))
            return
        if arg == "verify":
            await query.answer("Verifying…")
            if not room["group_id"]:
                await query.message.reply_text(
                    "❌ No group configured yet. Use ADD GROUP first.",
                    reply_markup=K.group_config(room_id))
                return
            await self._verify_group(client, room["group_id"], room_id)
            return
        if arg == "lock":
            if not room["group_id"]:
                await query.answer("Verify a group before locking.", show_alert=True)
                return
            for p in (room["creator_id"], room["partner_id"]):
                if not await self.db.get_session_enc(p or 0):
                    await query.answer("Both participants must connect accounts first.",
                                       show_alert=True)
                    return
            await self.db.update_room(room_id, locked=1, status="ready")
            await query.answer("Room locked.")
            await self.refresh_room_panel(room_id, query.message.chat.id)
            return
        if arg == "unlock":
            if room["status"] == "active":
                await query.answer("Stop the conversation first.", show_alert=True)
                return
            await self.db.update_room(room_id, locked=0, status="config")
            await query.answer("Room unlocked.")
            await self.refresh_room_panel(room_id, query.message.chat.id)
            return

    # ---- control routes (creator OR partner) -----------------------------------
    async def _route_control(self, client: Client, query: CallbackQuery, uid: int,
                             action: str, room_id: str) -> None:
        room = await self.is_room_participant(uid, room_id)
        if not room:
            await query.answer("⛔ Only room participants may control the conversation.",
                               show_alert=True)
            return

        if action == "start":
            if room["status"] == "active":
                await query.answer("Already running (duplicate START prevented).", show_alert=True)
                return
            ok, msg = await self.conv.start(room_id)
            await query.answer(msg, show_alert=not ok)
            if ok:
                await self.send_room_event(room_id, f"▶️ {msg}")
                await self.refresh_room_panel(room_id, query.message.chat.id)
            return
        if action == "pause":
            ok, msg = await self.conv.pause(room_id)
            await query.answer(msg, show_alert=not ok)
            if ok:
                await self.send_room_event(room_id, "⏸️ Conversation paused.")
            await self.refresh_room_panel(room_id, query.message.chat.id)
            return
        if action == "resume":
            ok, msg = await self.conv.resume(room_id)
            await query.answer(msg, show_alert=not ok)
            if ok:
                await self.send_room_event(room_id, "▶️ Conversation resumed.")
                await self.refresh_room_panel(room_id, query.message.chat.id)
            return
        if action == "stop":
            ok, msg = await self.conv.stop(room_id)
            await query.answer(msg, show_alert=not ok)
            if ok:
                await self.send_room_event(room_id, "⏹️ Conversation stopped. "
                                                    "START will continue from where you left off.")
            await self.refresh_room_panel(room_id, query.message.chat.id)
            return

    # ---- settings -----------------------------------------------------------
    async def _route_settings_view(self, query: CallbackQuery, uid: int, arg: str) -> None:
        await query.answer()
        if arg == "global":
            await query.message.edit_text(
                "⚙️ **SETTINGS**\n\nPer-room settings are managed inside each room. "
                "Open MY ROOMS → select a room → ROOM SETTINGS.",
                reply_markup=K.main_menu())
            return
        room = await self.is_room_participant(uid, arg)
        if not room:
            await query.answer("⛔ Not your room.", show_alert=True)
            return
        s = await self.db.get_settings(arg)
        await query.message.edit_text(
            f"⚙️ **ROOM SETTINGS — {arg}**\n\n"
            f"Topic: {s['topic']}\nInterval: {s['interval']}s · Max msgs: {s['max_messages']}\n"
            f"Session: {s['session_minutes']}min · Max length: {s['max_len']} chars\n"
            f"History: {s['history_len']} · Mentions: {s['mention_policy']}\n"
            f"Daily limit: {s['daily_limit']}\n"
            f"Language: English (locked)\n"
            f"Personality A: {s['personality_a'][:60]}…\n"
            f"Personality B: {s['personality_b'][:60]}…",
            reply_markup=K.settings_menu(arg))
        return

    CYCLES = {
        "interval": ("interval", [30, 45, 60, 90, 120, 180]),
        "maxmsg": ("max_messages", [20, 40, 60, 80, 120]),
        "minutes": ("session_minutes", [30, 45, 60, 90, 120]),
        "maxlen": ("max_len", [280, 400, 600, 800]),
        "histlen": ("history_len", [10, 20, 30, 50]),
        "mention": ("mention_policy", ["first_only", "always", "never"]),
        "daily": ("daily_limit", [60, 120, 200, 400]),
    }
    PERSONAS = {
        "default": (AIEngine.PERSONALITY_A, AIEngine.PERSONALITY_B),
        "curious": (AIEngine.PERSONALITY_A.replace("opinionated but open-minded",
                                                   "deeply curious and Socratic"),
                    AIEngine.PERSONALITY_B.replace("independent, well-reasoned opinions",
                                                   "playful, skeptical, well-reasoned opinions")),
        "debate": (AIEngine.PERSONALITY_A.replace("opinionated but open-minded",
                                                  "a sharp debate-style thinker who argues with evidence"),
                   AIEngine.PERSONALITY_B.replace("Disagree respectfully",
                                                  "Regularly but respectfully disagree with clear reasons")),
    }

    async def _route_settings_cycle(self, query: CallbackQuery, uid: int,
                                    room_id: str, key: str) -> None:
        if room_id == "g":
            await query.answer("Global defaults are set via environment variables.")
            return
        room = await self.is_room_participant(uid, room_id)
        if not room:
            await query.answer("⛔ Not your room.", show_alert=True)
            return
        s = await self.db.get_settings(room_id)
        await query.answer()
        if key == "resethist":
            await self.db.clear_history(room_id)
            await self.db.execute("DELETE FROM seen_msgs WHERE room_id=?", (room_id,))
            await self._route_settings_view(query, uid, room_id)
            return
        if key == "topic":
            topics = ["the future of technology and society",
                      "space exploration and humanity",
                      "modern education and AI",
                      "remote work culture",
                      "personal productivity and habits",
                      "artificial intelligence and creativity",
                      "social media and mental health"]
            nxt = topics[(topics.index(s["topic"]) + 1) % len(topics)] \
                if s["topic"] in topics else topics[0]
            await self.db.update_settings(room_id, topic=nxt)
            await self.db.update_room(room_id, topic=nxt)
            await self._route_settings_view(query, uid, room_id)
            return
        if key == "persona":
            cur = next((k for k, v in self.PERSONAS.items() if v[0] == s["personality_a"]), "default")
            order = list(self.PERSONAS)
            nxt = order[(order.index(cur) + 1) % len(order)]
            a, b = self.PERSONAS[nxt]
            await self.db.update_settings(room_id, personality_a=a, personality_b=b)
            await self._route_settings_view(query, uid, room_id)
            return
        field, values = self.CYCLES[key]
        cur = s[field]
        nxt = values[(values.index(cur) + 1) % len(values)] if cur in values else values[0]
        await self.db.update_settings(room_id, **{field: nxt})
        await self._route_settings_view(query, uid, room_id)

    # ---- incoming text (invitation codes) --------------------------------------
    async def on_text(self, client: Client, message: Message) -> None:
        if not message.text or message.chat.type != enums.ChatType.PRIVATE:
            return
        code = message.text.strip().upper()
        if re.fullmatch(r"[A-Z2-9]{8}", code):
            await self._accept_invite(client, message, code)

    async def cmd_disconnect(self, client: Client, message: Message) -> None:
        await self.db.delete_session(message.from_user.id)
        await self.bots.disconnect_user(message.from_user.id)
        await message.reply_text(
            "🔌 Your account session was removed. You can re-link anytime with /link.")
        room = await self.db.get_room_of_participant(message.from_user.id)
        if room:
            await self.send_room_event(
                room["room_id"],
                f"🔌 A participant disconnected their account in `{room['room_id']}`.")

    async def cmd_link(self, client: Client, message: Message) -> None:
        await message.reply_text(_LINK_HELP, parse_mode=enums.ParseMode.MARKDOWN)

    async def _show_help(self, query: CallbackQuery) -> None:
        await query.message.reply_text(
            "❓ **HELP**\n\n"
            "• CREATE ROOM — become Participant A and get an invite.\n"
            "• JOIN ROOM — accept an invitation code/link as Participant B.\n"
            "• Account connection — /link for secure terminal authorization.\n"
            "• Group config — add bot to the target group, creator sends /setgroup there.\n"
            "• START/PAUSE/RESUME/STOP — **both participants** can use these.\n"
            "• STOP → START continues the same discussion (history kept).\n"
            "• Manual replies in the group are picked up naturally by the AI.\n"
            "• INVITE / CANCEL / LOCK / UNLOCK — **creator only**.\n\n"
            "AI speaks **English only**. Sessions are Fernet-encrypted at rest.",
            reply_markup=K.close_only())


_LINK_HELP = (
    "🔗 **ACCOUNT CONNECTION**\n\n"
    "Telegram phone authorization (OTP + optional 2FA) cannot be done safely inside a bot "
    "chat, so linking happens in a secure terminal session on the machine running this bot:\n\n"
    "```\npython main.py --link <YOUR_TELEGRAM_USER_ID>\n```\n"
    "Your login code and 2FA password are entered via hidden input (getpass), never sent to "
    "the bot, never logged, and never stored in plaintext. Only a Fernet-encrypted session "
    "string is saved. After both participants link, press REFRESH STATUS in the room.\n\n"
    "To unlink: `/disconnect` in this chat."
)

# ----------------------------------------------------------------------------
# Application bootstrap
# ----------------------------------------------------------------------------
class App:
    def __init__(self):
        self.db = Database(CFG.DATABASE_URL)
        self.ai = AIEngine()
        self.bots: Optional[UserbotManager] = None
        self.conv: Optional[ConversationManager] = None
        self.handlers: Optional[Handlers] = None
        self.bot: Optional[Client] = None

    async def run(self) -> None:
        await self.db.connect()
        await self.ai.start()
        self.bots = UserbotManager(self.db)
        self.handlers = Handlers(self.db, self.ai, None, self.bots)
        self.conv = ConversationManager(self.db, self.ai, self.bots, self.handlers)
        self.handlers.conv = self.conv

        self.bot = Client("ai_rooms_bot", api_id=CFG.API_ID or None,
                          api_hash=CFG.API_HASH or None, bot_token=CFG.BOT_TOKEN,
                          in_memory=True)
        h = self.handlers

        @self.bot.on_message(filters.command("start") & filters.private)
        async def _start(c, m): await h.cmd_start(c, m)
        @self.bot.on_message(filters.command("link") & filters.private)
        async def _link(c, m): await h.cmd_link(c, m)
        @self.bot.on_message(filters.command("disconnect") & filters.private)
        async def _disc(c, m): await h.cmd_disconnect(c, m)
        @self.bot.on_message(filters.group & filters.command("setgroup"))
        async def _setgroup(c, m): await h.cmd_setgroup(c, m)
        @self.bot.on_callback_query()
        async def _cb(c, q): await h.on_callback(c, q)
        @self.bot.on_message(filters.private & filters.text &
                             ~filters.command(["start", "link", "disconnect"]))
        async def _txt(c, m): await h.on_text(c, m)

        stop_evt = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop_evt.set)

        await self.bot.start()
        log.info("Bot started as @%s", (await self.bot.get_me()).username)
        await self.bots.start_all()

        # resume any room that was ACTIVE before shutdown as paused
        for room in await self.db.all_active_rooms():
            if room["status"] == "active":
                await self.db.update_room(room["room_id"], status="paused")
                await self.handlers.send_room_event(
                    room["room_id"],
                    "⚠️ Application restarted — conversation paused. "
                    "Press RESUME (or START) to continue from where it stopped.")

        bot_task = asyncio.create_task(self._idle())
        try:
            await stop_evt.wait()
        finally:
            log.info("Shutting down…")
            for room_id in list(self.conv.running):
                await self.conv.pause(room_id)  # preserve state on shutdown
            await self.bots.stop_all()
            bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bot_task
            await self.bot.stop()
            await self.ai.close()
            await self.db.close()

    async def _idle(self) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", type=int, metavar="USER_ID",
                        help="Interactively authorize a Telegram account (terminal-only).")
    args = parser.parse_args()

    missing = [k for k in ("BOT_TOKEN", "API_ID", "API_HASH") if not os.getenv(k)]
    if args.link:
        if not os.getenv("API_ID") or not os.getenv("API_HASH"):
            print("Linking requires API_ID and API_HASH in the environment.")
            sys.exit(1)
    else:
        if missing or not CFG.AI_API_KEY:
            print("Missing required env vars:", ", ".join(missing + ["AI_API_KEY"]))
            sys.exit(1)

    async def runner():
        db = Database(CFG.DATABASE_URL)
        await db.connect()
        try:
            if args.link:
                await interactive_link(args.link, db)
            else:
                await App().run()
        finally:
            await db.close()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
