"""Durable request log for coordinator crash recovery (SQLite, WAL mode).

The coordinator writes three kinds of facts:

* a request was accepted (prompt, max_new_tokens, deadline);
* a token was produced for a step (one transaction per micro-batch, committed
  BEFORE the tokens are streamed to clients, so every token a client has ever
  seen is in the log);
* a request reached a terminal status.

After a crash, a new coordinator incarnation opens the same file, takes a new
coordinator epoch, and rebuilds every request that has no terminal status from
`prompt + logged tokens`. Finished requests are kept for `retention_s` so a
client whose stream broke can still fetch the final result with Resume.

Fencing: the current coordinator epoch is stored in the file. Every write
transaction first checks that it is still the caller's epoch, so a coordinator
that has been replaced (e.g. it was frozen, not dead) can no longer write.

All methods are synchronous and must be called from one thread (the
coordinator's event loop). With WAL + synchronous=NORMAL a commit does not wait
for fsync, which is enough to survive a *process* crash; use sync="full" to
also survive a power loss / kernel crash.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

RUNNING = 0  # pb.RUNNING; kept numeric so this module does not depend on the protobuf bindings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    client_id   TEXT PRIMARY KEY,
    wire_id     TEXT NOT NULL,      -- id used on the worker wire by the incarnation that last ran it
    prompt      TEXT NOT NULL,      -- JSON list of token ids
    max_new     INTEGER NOT NULL,
    deadline    REAL NOT NULL,      -- wall clock (time.time())
    status      INTEGER NOT NULL,   -- 0 = RUNNING, otherwise a terminal RequestStatus
    detail      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS requests_status ON requests(status, finished_at);
CREATE TABLE IF NOT EXISTS tokens (
    client_id TEXT NOT NULL,
    step      INTEGER NOT NULL,
    token     INTEGER NOT NULL,
    PRIMARY KEY (client_id, step)
) WITHOUT ROWID;
"""


class FencedError(Exception):
    """A newer coordinator incarnation owns the log; this one must stop."""


@dataclass
class LoggedRequest:
    client_id: str
    wire_id: str
    prompt: list[int]
    max_new: int
    deadline: float
    status: int
    detail: str
    tokens: list[int] = field(default_factory=list)


@dataclass
class LogStats:
    commits: int = 0
    commit_s: float = 0.0
    max_commit_s: float = 0.0
    tokens: int = 0

    def as_dict(self) -> dict:
        return {"commits": self.commits, "tokens": self.tokens,
                "mean_commit_ms": (self.commit_s / self.commits * 1e3) if self.commits else 0.0,
                "max_commit_ms": self.max_commit_s * 1e3}


class RequestLog:
    def __init__(self, path: str, sync: str = "normal", retention_s: float = 600.0):
        self.path = str(path)
        self.retention_s = retention_s
        self.epoch = 0
        self.stats = LogStats()
        # isolation_level=None: we issue BEGIN/COMMIT ourselves.
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
        self.db.execute("PRAGMA journal_mode=WAL")
        if sync == "full":
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA fullfsync=ON")   # macOS: really flush to the device
        elif sync == "off":
            self.db.execute("PRAGMA synchronous=OFF")
        else:
            self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        try:
            self.db.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------ epochs / fencing
    def take_over(self, min_epoch: int = 0) -> int:
        """Become the owner of the log: atomically raise the coordinator epoch above the stored one
        (and to at least `min_epoch`) and return it. Every later write by an older owner fails."""
        with self._txn(check=False):
            row = self.db.execute("SELECT value FROM meta WHERE key='coordinator_epoch'").fetchone()
            self.epoch = max((int(row[0]) if row else 0) + 1, min_epoch)
            self._set_meta("coordinator_epoch", self.epoch)
        return self.epoch

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value) -> None:
        with self._txn():
            self._set_meta(key, value)

    def _set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    class _Txn:
        def __init__(self, log: "RequestLog", check: bool):
            self.log, self.check = log, check

        def __enter__(self):
            self.t0 = time.perf_counter()
            self.log.db.execute("BEGIN IMMEDIATE")
            if self.check:
                row = self.log.db.execute("SELECT value FROM meta WHERE key='coordinator_epoch'").fetchone()
                current = int(row[0]) if row else 0
                if current != self.log.epoch:
                    self.log.db.execute("ROLLBACK")
                    raise FencedError(f"request log owned by coordinator epoch {current}, "
                                      f"this coordinator is epoch {self.log.epoch}")
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.log.db.execute("COMMIT")
                dt = time.perf_counter() - self.t0
                st = self.log.stats
                st.commits += 1
                st.commit_s += dt
                st.max_commit_s = max(st.max_commit_s, dt)
            elif self.log.db.in_transaction:
                self.log.db.execute("ROLLBACK")
            return False

    def _txn(self, check: bool = True) -> "RequestLog._Txn":
        return RequestLog._Txn(self, check)

    # ------------------------------------------------------------------ writes
    def start(self, client_id: str, wire_id: str, prompt: list[int], max_new: int, deadline: float) -> None:
        with self._txn():
            self.db.execute("DELETE FROM tokens WHERE client_id=?", (client_id,))  # client reused a finished id
            self.db.execute(
                "INSERT OR REPLACE INTO requests(client_id, wire_id, prompt, max_new, deadline, status, detail, "
                "created_at, finished_at) VALUES(?, ?, ?, ?, ?, ?, '', ?, NULL)",
                (client_id, wire_id, json.dumps(list(prompt)), max_new, deadline, RUNNING, time.time()))

    def record(self, tokens: list[tuple[str, int, int]], finished: list[tuple[str, int, str]] = ()) -> None:
        """One transaction: tokens [(client_id, step, token)] and terminal statuses
        [(client_id, status, detail)]. Must return before any of these tokens is streamed."""
        if not tokens and not finished:
            return
        with self._txn():
            if tokens:
                # OR IGNORE: a step is written once; a re-run of the same step after recovery produces the
                # same token (greedy decoding), so the existing row is already correct.
                self.db.executemany("INSERT OR IGNORE INTO tokens(client_id, step, token) VALUES(?, ?, ?)", tokens)
            if finished:
                now = time.time()
                self.db.executemany("UPDATE requests SET status=?, detail=?, finished_at=? WHERE client_id=?",
                                    [(st, det, now, cid) for cid, st, det in finished])
        self.stats.tokens += len(tokens)

    def finish(self, client_id: str, status: int, detail: str = "") -> None:
        self.record([], [(client_id, status, detail)])

    def set_wire_ids(self, pairs: list[tuple[str, str]]) -> None:
        """[(client_id, new_wire_id)] after a restarted coordinator re-ids the requests it restored."""
        with self._txn():
            self.db.executemany("UPDATE requests SET wire_id=? WHERE client_id=?",
                                [(w, c) for c, w in pairs])

    def gc(self, now: float | None = None) -> int:
        """Delete finished requests older than the retention period. Returns how many were removed."""
        cutoff = (time.time() if now is None else now) - self.retention_s
        with self._txn():
            ids = [r[0] for r in self.db.execute(
                "SELECT client_id FROM requests WHERE status != ? AND finished_at < ?", (RUNNING, cutoff))]
            if ids:
                self.db.executemany("DELETE FROM tokens WHERE client_id=?", [(i,) for i in ids])
                self.db.executemany("DELETE FROM requests WHERE client_id=?", [(i,) for i in ids])
        return len(ids)

    # ------------------------------------------------------------------ reads
    def _tokens(self, client_id: str) -> list[int]:
        rows = self.db.execute("SELECT step, token FROM tokens WHERE client_id=? ORDER BY step",
                               (client_id,)).fetchall()
        out = []
        for step, token in rows:
            if step != len(out):  # cannot happen with in-order commits; never resume from a gap
                break
            out.append(token)
        return out

    def _row(self, row) -> LoggedRequest:
        cid, wire, prompt, max_new, deadline, status, detail = row
        return LoggedRequest(cid, wire, json.loads(prompt), max_new, deadline, status, detail, self._tokens(cid))

    def unfinished(self) -> list[LoggedRequest]:
        rows = self.db.execute("SELECT client_id, wire_id, prompt, max_new, deadline, status, detail "
                               "FROM requests WHERE status = ? ORDER BY created_at", (RUNNING,)).fetchall()
        return [self._row(r) for r in rows]

    def lookup(self, client_id: str) -> LoggedRequest | None:
        row = self.db.execute("SELECT client_id, wire_id, prompt, max_new, deadline, status, detail "
                              "FROM requests WHERE client_id=?", (client_id,)).fetchone()
        return self._row(row) if row else None

    def counts(self) -> dict:
        running = self.db.execute("SELECT COUNT(*) FROM requests WHERE status=?", (RUNNING,)).fetchone()[0]
        total = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        return {"running": running, "finished": total - running}
