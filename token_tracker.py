import atexit
import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_usage_tokens(resp: Any) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return None, None

    inp = getattr(usage, "prompt_tokens", None)
    if inp is None:
        inp = getattr(usage, "input_tokens", None)

    out = getattr(usage, "completion_tokens", None)
    if out is None:
        out = getattr(usage, "output_tokens", None)

    return _safe_int(inp), _safe_int(out)


def _book_name_from_env() -> Optional[str]:
    novel_path = os.environ.get("NOVEL_PATH")
    if not novel_path:
        return None
    return os.path.splitext(os.path.basename(novel_path))[0].strip() or None


def _generate_run_id() -> str:
    return f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"


def _new_book_entry(book_name: str, timestamp: str) -> Dict[str, Any]:
    return {
        "book_name": book_name,
        "book_total_input": 0,
        "book_total_output": 0,
        "book_total_tokens": 0,
        "updated_at": timestamp,
        "runs": {},
    }


def _new_run_entry(run_id: str, started_at: str, updated_at: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "run_total_input": 0,
        "run_total_output": 0,
        "run_total_tokens": 0,
        "started_at": started_at,
        "updated_at": updated_at,
        "scripts": {},
    }


class TokenTracker:
    def __init__(
        self,
        script: str,
        book_name: Optional[str],
        out_path: str = "./results/token_usage.json",
        run_id: Optional[str] = None,
        autosave: bool = True,
    ) -> None:
        resolved_book_name = (book_name or "").strip() or _book_name_from_env()
        if not resolved_book_name:
            raise ValueError("book_name is required unless NOVEL_PATH can be resolved")

        self.script = script
        self.book_name = resolved_book_name
        self.out_path = out_path
        self.run_id = (run_id or os.environ.get("TOKEN_RUN_ID") or _generate_run_id()).strip()
        self.start = datetime.now().isoformat()
        self.autosave = bool(autosave)
        self._lock = threading.Lock()
        self._flushed = False

        self.totals: Dict[str, int] = {"input": 0, "output": 0}
        self.last_write: Optional[str] = None
        self.status: str = "running"
        self.reason: str = ""

        atexit.register(self._atexit_flush)

    def add(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            if input_tokens:
                self.totals["input"] += int(input_tokens)
            if output_tokens:
                self.totals["output"] += int(output_tokens)
            if self.autosave:
                self._write_locked()

    def record(self, resp: Any) -> None:
        inp, out = _extract_usage_tokens(resp)
        if inp is None and out is None:
            return
        self.add(inp or 0, out or 0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "book_name": self.book_name,
                "run_id": self.run_id,
                "script": self.script,
                "status": self.status,
                "reason": self.reason,
                "start": self.start,
                "timestamp": datetime.now().isoformat(),
                "input": int(self.totals.get("input", 0)),
                "output": int(self.totals.get("output", 0)),
                "total": int(self.totals.get("input", 0)) + int(self.totals.get("output", 0)),
                "last_write": self.last_write,
                "pid": os.getpid(),
            }

    def flush(self, status: str = "finished", reason: str = "") -> None:
        with self._lock:
            self.status = status or self.status
            self.reason = (reason or "")[:500]
            self._write_locked()
            self._flushed = True

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.out_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _lock_path(self) -> str:
        return f"{self.out_path}.lock"

    def _acquire_file_lock(self, timeout_s: float = 5.0, poll_s: float = 0.05, stale_after_s: float = 120.0) -> bool:
        deadline = time.time() + float(timeout_s)
        lock_path = self._lock_path()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()}\n{datetime.now().isoformat()}".encode("utf-8", errors="ignore"))
                finally:
                    os.close(fd)
                return True
            except FileExistsError:
                # 自清理：锁文件超龄（持有者进程已死/残留）则视为僵死锁，删除后重试
                if self._is_stale_lock(lock_path, stale_after_s):
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                    continue
                if time.time() >= deadline:
                    return False
                time.sleep(poll_s)
            except Exception:
                return False

    @staticmethod
    def _is_stale_lock(lock_path: str, stale_after_s: float) -> bool:
        try:
            with open(lock_path, "r", encoding="utf-8", errors="ignore") as _lf:
                _lines = _lf.read().splitlines()
        except FileNotFoundError:
            return False
        except Exception:
            return True
        if not _lines:
            return True
        if len(_lines) >= 2:
            try:
                _ts = datetime.fromisoformat(_lines[1].strip())
                if (datetime.now() - _ts).total_seconds() > float(stale_after_s):
                    return True
            except Exception:
                return True
        return len(_lines) < 2

    def _release_file_lock(self) -> None:
        try:
            os.remove(self._lock_path())
        except Exception:
            pass

    def _read_existing_aggregate(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "schema_version": 3,
            "updated_at": None,
            "books": {},
        }
        if not os.path.exists(self.out_path):
            return base
        try:
            with open(self.out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return base

        if not isinstance(data, dict):
            return base
        if data.get("schema_version") != 3:
            return base

        books = data.get("books")
        if not isinstance(books, dict):
            return base

        data.setdefault("updated_at", None)
        return data

    def _build_payload(self) -> Dict[str, Any]:
        return {
            "script": self.script,
            "status": self.status,
            "reason": self.reason,
            "start": self.start,
            "timestamp": datetime.now().isoformat(),
            "input": int(self.totals.get("input", 0)),
            "output": int(self.totals.get("output", 0)),
            "total": int(self.totals.get("input", 0)) + int(self.totals.get("output", 0)),
            "pid": os.getpid(),
        }

    def _recompute_run_totals(self, run_entry: Dict[str, Any], timestamp: str) -> None:
        total_input = 0
        total_output = 0
        for script_payload in run_entry.get("scripts", {}).values():
            total_input += int(script_payload.get("input", 0))
            total_output += int(script_payload.get("output", 0))
        run_entry["run_total_input"] = total_input
        run_entry["run_total_output"] = total_output
        run_entry["run_total_tokens"] = total_input + total_output
        run_entry["updated_at"] = timestamp

    def _recompute_book_totals(self, book_entry: Dict[str, Any], timestamp: str) -> None:
        total_input = 0
        total_output = 0
        for run_entry in book_entry.get("runs", {}).values():
            total_input += int(run_entry.get("run_total_input", 0))
            total_output += int(run_entry.get("run_total_output", 0))
        book_entry["book_total_input"] = total_input
        book_entry["book_total_output"] = total_output
        book_entry["book_total_tokens"] = total_input + total_output
        book_entry["updated_at"] = timestamp

    def _write_locked(self) -> None:
        self._ensure_dir()
        payload = self._build_payload()

        lock_acquired = self._acquire_file_lock()
        if not lock_acquired:
            raise TimeoutError(f"failed to acquire token usage lock: {self._lock_path()}")
        try:
            agg = self._read_existing_aggregate()
            books = agg.setdefault("books", {})
            book_entry = books.get(self.book_name)
            if not isinstance(book_entry, dict):
                book_entry = _new_book_entry(self.book_name, payload["timestamp"])
                books[self.book_name] = book_entry
            else:
                book_entry.setdefault("book_name", self.book_name)
                book_entry.setdefault("runs", {})

            runs = book_entry.setdefault("runs", {})
            run_entry = runs.get(self.run_id)
            if not isinstance(run_entry, dict):
                run_entry = _new_run_entry(self.run_id, self.start, payload["timestamp"])
                runs[self.run_id] = run_entry
            else:
                run_entry.setdefault("run_id", self.run_id)
                run_entry.setdefault("started_at", self.start)
                run_entry.setdefault("scripts", {})

            scripts = run_entry.setdefault("scripts", {})
            scripts[self.script] = payload

            self._recompute_run_totals(run_entry, payload["timestamp"])
            self._recompute_book_totals(book_entry, payload["timestamp"])
            agg["updated_at"] = payload["timestamp"]

            tmp_path = f"{self.out_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(agg, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.out_path)
            self.last_write = payload["timestamp"]
        finally:
            if lock_acquired:
                self._release_file_lock()

    def _atexit_flush(self) -> None:
        try:
            with self._lock:
                if self._flushed:
                    return
                self.status = "exit"
                self._write_locked()
                self._flushed = True
        except Exception:
            pass


def create_default_tracker(
    script: str,
    book_name: Optional[str],
    out_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> TokenTracker:
    if out_path is None:
        out_path = os.path.join(".", "results", "token_usage.json")
    return TokenTracker(
        script=script,
        book_name=book_name,
        out_path=out_path,
        run_id=run_id,
        autosave=True,
    )
