"""Local conversation and synthesis records for the director WebUI."""

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _clean_one_line(text: str, limit: int = 36) -> str:
    if not text:
        return ""
    s = re.sub(r"<[a-z_]+>", "", text)
    s = " ".join(s.split())
    return (s[:limit] + "…") if len(s) > limit else s


def _kind_badge(kind: str) -> str:
    if "满意" in kind:
        return "✅ 满意样本"
    if "标记错误" in kind:
        return "❌ 错题反馈"
    if "主选" in kind:
        return "⭐ 主选"
    if "备选" in kind:
        return "🎵 备选"
    if "异常" in kind:
        return "⚠️ 试音异常"
    if "评审" in kind:
        return "🎯 评审"
    return f"📝 {kind}"


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

    def _query_events(self, *, audio_only=False, main_only=False, chat_id=None):
        conditions, args = [], []
        if main_only:
            conditions.append("audio IS NOT NULL AND kind='主选音频'")
        elif audio_only:
            conditions.append("audio IS NOT NULL")
        if chat_id:
            conditions.append("chat_id=?")
            args.append(chat_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._database() as db:
            return db.execute(
                "SELECT id,created,kind,audio,ja,zh FROM events" + where + " ORDER BY id DESC LIMIT 250", args,
            ).fetchall()

    def list_events(self, *, audio_only=False, main_only=False, chat_id=None):
        rows = self._query_events(audio_only=audio_only, main_only=main_only, chat_id=chat_id)
        out = []
        for row in rows:
            ts = row["created"][5:16].replace("T", " ")
            badge = _kind_badge(row["kind"])
            zh_s = _clean_one_line(row["zh"], 28)
            ja_s = _clean_one_line(row["ja"], 28)
            if zh_s and ja_s and zh_s != ja_s:
                summary = f"「{zh_s}」({ja_s})"
            elif zh_s or ja_s:
                summary = f"「{zh_s or ja_s}」"
            elif row["audio"]:
                summary = Path(row["audio"]).stem
            else:
                summary = "查看记录"
            out.append((f"{badge} · {summary} · [{ts}]", str(row["id"])))
        return out

    def list_table_rows(self, *, audio_only=False, main_only=False, chat_id=None):
        rows = self._query_events(audio_only=audio_only, main_only=main_only, chat_id=chat_id)
        table = []
        for row in rows:
            ts = row["created"][5:16].replace("T", " ")
            badge = _kind_badge(row["kind"])
            zh_s = _clean_one_line(row["zh"], 34) or "—"
            ja_s = _clean_one_line(row["ja"], 34) or "—"
            fname = Path(row["audio"]).name if row["audio"] else "—"
            table.append([str(row["id"]), ts, badge, zh_s, ja_s, fname])
        return table

    def audio_paths(self):
        with self._database() as db:
            return {row[0] for row in db.execute("SELECT DISTINCT audio FROM events WHERE audio IS NOT NULL")}

    def event(self, event_id):
        with self._database() as db:
            return db.execute(
                "SELECT events.*, chats.title FROM events LEFT JOIN chats ON chats.id=events.chat_id WHERE events.id=?",
                (event_id,),
            ).fetchone()

    def delete_event(self, event_id):
        with self._database() as db:
            db.execute("DELETE FROM events WHERE id=?", (event_id,))
