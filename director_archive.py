"""Local conversation and synthesis records for the director WebUI."""

import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class DirectorArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.attachments = self.directory / "attachments"
        self.attachments.mkdir(exist_ok=True)
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, updated TEXT NOT NULL, history TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, chat_id TEXT, created TEXT NOT NULL, kind TEXT NOT NULL,
                    audio TEXT, ja TEXT NOT NULL, zh TEXT NOT NULL, detail TEXT NOT NULL
                );
            """)

    @contextmanager
    def _database(self):
        db = sqlite3.connect(self.directory / "director.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="microseconds")

    def new_chat(self):
        return uuid4().hex

    def list_chats(self):
        with self._database() as db:
            rows = db.execute("SELECT id, title, updated FROM chats ORDER BY updated DESC LIMIT 250").fetchall()
        return [(f"{row['updated'][5:16].replace('T', ' ')} · {row['title']}", row["id"]) for row in rows]

    def load_chat(self, chat_id):
        with self._database() as db:
            row = db.execute("SELECT history FROM chats WHERE id=?", (chat_id,)).fetchone()
        return json.loads(row["history"]) if row else []

    def save_chat(self, chat_id, history):
        if not history:
            return []
        title = "新对话"
        saved = []
        for turn in history:
            role, content = turn.get("role"), turn.get("content")
            if role not in ("user", "assistant"):
                continue
            if isinstance(content, str):
                if role == "user" and title == "新对话" and content.strip():
                    title = " ".join(content.split())[:50]
            elif isinstance(content, (tuple, list)) and content:
                file = Path(content[0]).resolve()
                if not file.is_file():
                    content = f"附件已不可用：{file.name}"
                elif not file.is_relative_to(self.attachments.resolve()):
                    hasher = hashlib.sha256()
                    with file.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    digest = hasher.hexdigest()
                    destination = self.attachments / digest / file.name
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_name(uuid4().hex + ".tmp")
                        shutil.copyfile(file, temporary)
                        temporary.replace(destination)
                    content = [str(destination)]
            else:
                continue
            saved.append({"role": role, "content": content})
        with self._database() as db:
            db.execute("INSERT OR REPLACE INTO chats VALUES (?,?,?,?)",
                       (chat_id, title, self._now(), json.dumps(saved, ensure_ascii=False)))
        return saved

    def record(self, chat_id, kind, detail, *, audio=None, ja="", zh=""):
        with self._database() as db:
            db.execute("INSERT INTO events (chat_id,created,kind,audio,ja,zh,detail) VALUES (?,?,?,?,?,?,?)",
                       (chat_id, self._now(), kind, str(audio) if audio else None, ja, zh, detail))

    def list_events(self, *, audio_only=False, chat_id=None):
        conditions, args = [], []
        if audio_only:
            conditions.append("audio IS NOT NULL")
        if chat_id:
            conditions.append("chat_id=?")
            args.append(chat_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._database() as db:
            rows = db.execute(
                "SELECT id,created,kind,audio,ja FROM events" + where + " ORDER BY id DESC LIMIT 250", args,
            ).fetchall()
        return [(f"{row['created'][5:16].replace('T', ' ')} · {row['kind']} · "
                 f"{(row['ja'] or Path(row['audio']).name if row['audio'] else row['ja'] or '查看记录')[:45]}",
                 str(row["id"])) for row in rows]

    def audio_paths(self):
        with self._database() as db:
            return {row[0] for row in db.execute("SELECT DISTINCT audio FROM events WHERE audio IS NOT NULL")}

    def event(self, event_id):
        with self._database() as db:
            return db.execute(
                "SELECT events.*, chats.title FROM events LEFT JOIN chats ON chats.id=events.chat_id WHERE events.id=?",
                (event_id,),
            ).fetchone()
