"""
Out-of-process Moss access for the voice agent — Zone A.

Why this exists: the moss SDK's native core holds Python's GIL while it
embeds and writes. Measured with a heartbeat on the agent's event loop, every
ingest froze the whole process for ~3s and preloading a patient froze it for
~1s — even though the SDK runs those calls in a worker thread. In a live call
that stalls the agent's audio and delays its reply after every answer.
Running Moss in child processes removes the contention: the agent's loop only
does cheap pipe I/O.

Two children, not one: a write takes 3-5s, and a lookup queued behind it in
the same process would wait that long. The reader holds the patient's loaded
index and answers lookups; the writer only appends.

Plain subprocesses speaking JSON lines, rather than multiprocessing, so this
works inside LiveKit's job processes regardless of how those were spawned.

  request  {"id": 7, "op": "ingest"|"query"|"preload"|"ping", "args": {...}}
  response {"id": 7, "ok": true, "result": ...}
           {"id": 7, "ok": false, "error": "..."}

In STUB_MODE (no Moss credentials) everything runs in-process: the stub store
has no native core to isolate, and splitting it across processes would hide
writes from lookups.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from db import moss_client
from db.contracts import MossQARecord

logger = logging.getLogger("agent.moss")

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --- child process ------------------------------------------------------------

async def _serve() -> None:
    # Protocol lines go to the real stdout; anything else printed (by the SDK
    # or a stray print) goes to stderr so it can't corrupt the stream.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    moss_client.warm_up_client()

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    async def handle(req: dict) -> None:
        rid = req.get("id")
        try:
            op, args = req["op"], req.get("args") or {}
            result: Any = None
            if op == "ingest":
                await moss_client.ingest_qa_record(MossQARecord(**args["record"]))
            elif op == "ingest_many":
                await moss_client.ingest_qa_records([MossQARecord(**r) for r in args["records"]])
            elif op == "query":
                records = await moss_client.query_patient_history(**args)
                result = [dataclasses.asdict(r) for r in records]
            elif op == "preload":
                await moss_client.preload_patient(args["patient_id"])
            elif op == "ping":
                result = "pong"
            else:
                raise ValueError(f"unknown op {op!r}")
            reply = {"id": rid, "ok": True, "result": result}
        except Exception as exc:  # report it; one bad request shouldn't end Moss access for the call
            reply = {"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        proto.write(json.dumps(reply) + "\n")

    in_flight: set[asyncio.Task] = set()
    while True:
        line = await reader.readline()
        if not line:  # parent closed stdin: finish what's running, then exit
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = asyncio.create_task(handle(req))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


# --- parent side ----------------------------------------------------------------

class _Child:
    """One Moss child process and the request/response plumbing to it."""

    def __init__(self, role: str):
        self.role = role
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._next_id = 0
        self._waiters: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._started = asyncio.Event()
        self._start_error: Optional[BaseException] = None

    async def start(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "agent.moss_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                cwd=str(_REPO_ROOT),
                env=os.environ.copy(),
            )
            self._reader_task = asyncio.create_task(self._read_replies())
        except BaseException as exc:
            self._start_error = exc
            raise
        finally:
            self._started.set()  # never leave requests waiting on a start that failed

    async def _read_replies(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            fut = self._waiters.pop(msg.get("id"), None)
            if fut is None or fut.done():
                continue
            if msg.get("ok"):
                fut.set_result(msg.get("result"))
            else:
                fut.set_exception(RuntimeError(f"moss {self.role}: {msg.get('error')}"))
        for fut in self._waiters.values():
            if not fut.done():
                fut.set_exception(RuntimeError(f"moss {self.role} process exited"))
        self._waiters.clear()

    async def request(self, op: str, args: dict) -> Any:
        await self._started.wait()
        if self._start_error is not None:
            raise RuntimeError(f"moss {self.role} failed to start") from self._start_error
        if self._proc is None or self._proc.returncode is not None or self._proc.stdin is None:
            raise RuntimeError(f"moss {self.role} process is not running")
        self._next_id += 1
        rid = self._next_id
        fut = asyncio.get_running_loop().create_future()
        self._waiters[rid] = fut
        self._proc.stdin.write((json.dumps({"id": rid, "op": op, "args": args}) + "\n").encode())
        await self._proc.stdin.drain()
        return await fut

    async def close(self, timeout: float = 10.0) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
            await asyncio.wait_for(self._proc.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning("moss %s did not exit in %.0fs; killing it", self.role, timeout)
            self._proc.kill()
            await self._proc.wait()
        if self._reader_task:
            await asyncio.gather(self._reader_task, return_exceptions=True)


class MossBridge:
    """What the agent uses for Moss: fire-and-forget saves, lookups, preload.

    isolated=True (the default whenever real Moss credentials are set) runs
    Moss in child processes; isolated=False calls db.moss_client directly.
    """

    def __init__(self, isolated: Optional[bool] = None,
                 batch_window_s: float = 0.25, batch_max: int = 25):
        self.isolated = (not moss_client.STUB_MODE) if isolated is None else isolated
        self._writer = _Child("writer") if self.isolated else None
        self._reader = _Child("reader") if self.isolated else None
        self._pending: set[asyncio.Task] = set()
        # Answers wait briefly here so several go out in one write. A Moss
        # round trip costs 3-5s whatever it carries, so writing one answer at
        # a time left a whole call's worth of writes unfinished at hang-up.
        self._outbox: list[MossQARecord] = []
        self._batch_window_s = batch_window_s
        self._batch_max = batch_max
        self._flusher: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.isolated:
            await asyncio.gather(self._writer.start(), self._reader.start())

    def save(self, record: MossQARecord) -> None:
        """Queue a write and return immediately; failures are logged."""
        self._outbox.append(record)
        self._ensure_flusher()

    def _ensure_flusher(self) -> None:
        if self._flusher is not None and not self._flusher.done():
            return
        task = asyncio.create_task(self._drain_outbox())
        self._flusher = task
        self._pending.add(task)

        def _done(t: asyncio.Task) -> None:
            self._pending.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.warning("Moss write batch failed: %s", t.exception())

        task.add_done_callback(_done)

    async def _drain_outbox(self) -> None:
        await asyncio.sleep(self._batch_window_s)  # let a few answers accumulate
        while self._outbox:
            batch = self._outbox[: self._batch_max]
            del self._outbox[: len(batch)]
            try:
                await self._ingest_many(batch)
            except Exception as exc:
                logger.warning("Moss write failed for %s: %s",
                               ", ".join(r.question_id for r in batch), exc)

    async def _ingest_many(self, records: list[MossQARecord]) -> None:
        if self.isolated:
            await self._writer.request(
                "ingest_many", {"records": [dataclasses.asdict(r) for r in records]}
            )
        else:
            await moss_client.ingest_qa_records(records)

    async def query(
        self,
        patient_id: int,
        query_text: str,
        question_topic: Optional[str] = None,
        top_k: int = 5,
    ) -> list[MossQARecord]:
        args = {"patient_id": patient_id, "query_text": query_text,
                "question_topic": question_topic, "top_k": top_k}
        if not self.isolated:
            return await moss_client.query_patient_history(**args)
        rows = await self._reader.request("query", args)
        return [MossQARecord(**row) for row in rows]

    async def preload(self, patient_id: int) -> None:
        if self.isolated:
            await self._reader.request("preload", {"patient_id": patient_id})
        else:
            await moss_client.preload_patient(patient_id)

    async def flush(self, timeout: float = 30.0) -> None:
        """Wait for queued writes so the last answers of a call aren't lost."""
        if self._outbox:
            self._ensure_flusher()
        if not self._pending:
            return
        _, still_pending = await asyncio.wait(set(self._pending), timeout=timeout)
        if still_pending:
            logger.warning("%d Moss write(s) still pending after %.0fs", len(still_pending), timeout)

    async def close(self) -> None:
        if self.isolated:
            await asyncio.gather(self._writer.close(), self._reader.close(), return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    asyncio.run(_serve())
