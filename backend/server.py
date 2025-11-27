#!/usr/bin/env python3
"""
server.py — non-blocking analyzer runner + status API.

Behavior:
- GET /api/analyze?username=... will:
    * return loveble JSON immediately if present
    * if not present and not running: start background job and return status started
    * if running: return status running
- GET /api/analyze_status?username=... will:
    * return job status, tail(stdout), tail(stderr), done flag, and loveble path if ready

This file intentionally keeps the extractor/transform calls unchanged and only wraps them.
"""
from __future__ import annotations
import asyncio
import json
import subprocess
import sys
import shlex
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------ Configuration ------------
BASE_DIR = Path(__file__).resolve().parent
ANALYZE_SCRIPT = BASE_DIR / "analyze_user.py"
OUTPUT_ROOT = BASE_DIR / "outputs"
DEFAULT_TIMEOUT_SECONDS = 600  # default extractor timeout
ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # restrict in production
]

# ------------ App & CORS ------------
app = FastAPI(title="GitHub Lens Backend (non-blocking)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Response Models ------------
class AnalyzeResponse(BaseModel):
    ok: bool
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    loveble: Optional[Dict[str, Any]] = None
    status: Optional[str] = None  # "started" | "running" | "done" | "failed"

class StatusResponse(BaseModel):
    username: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    timed_out: Optional[bool] = False
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    loveble_ready: Optional[bool] = False
    loveble_path: Optional[str] = None
    error: Optional[str] = None

# ------------ Job store (in-memory) ------------
# Structure:
# jobs[username] = {
#   "task": asyncio.Task,
#   "running": bool,
#   "started_at": datetime,
#   "finished_at": datetime|None,
#   "stdout": str,
#   "stderr": str,
#   "returncode": int|None,
#   "timed_out": bool,
#   "loveble_path": Path|None,
#   "error": str|None
# }
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = asyncio.Lock()  # protect jobs dict

# ------------ Utilities ------------
def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def tail_text(s: str, lines: int = 30) -> str:
    if not s:
        return ""
    arr = s.splitlines()
    return "\n".join(arr[-lines:])

async def run_subprocess_in_thread(cmd: list[str], cwd: Optional[str], timeout: int):
    """
    Run subprocess.run in a thread to avoid blocking event loop.
    Returns dict with keys: returncode, stdout, stderr, timed_out (bool)
    """
    def _run():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
            return {"returncode": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or "", "timed_out": False}
        except subprocess.TimeoutExpired as te:
            out = getattr(te, "output", "") or ""
            err = getattr(te, "stderr", "") or f"Timeout after {timeout}s"
            return {"returncode": None, "stdout": out, "stderr": err, "timed_out": True}
        except Exception as e:
            return {"returncode": None, "stdout": "", "stderr": f"Failed to run subprocess: {e}", "timed_out": False}
    return await asyncio.to_thread(_run)

async def background_analyze(username: str, top_k: int, use_cache: bool, skip_stats: bool, timeout: int):
    """
    The actual background job runner that updates jobs[username].
    """
    outdir = OUTPUT_ROOT / username
    outdir.mkdir(parents=True, exist_ok=True)
    loveble_path = outdir / "loveble_schema.json"

    cmd = [sys.executable, str(ANALYZE_SCRIPT), "--username", username, "--top-k", str(top_k), "--output-dir", str(outdir)]
    if use_cache:
        cmd.append("--cache")
    if skip_stats:
        cmd.append("--skip-stats")

    # record start
    async with jobs_lock:
        jobs.setdefault(username, {})
        jobs[username].update({
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "timed_out": False,
            "loveble_path": None,
            "error": None,
        })

    # run
    res = await run_subprocess_in_thread(cmd, cwd=str(BASE_DIR), timeout=timeout)

    # update job state
    async with jobs_lock:
        j = jobs.setdefault(username, {})
        j["running"] = False
        j["finished_at"] = datetime.now(timezone.utc).isoformat()
        j["returncode"] = res.get("returncode")
        j["timed_out"] = res.get("timed_out", False)
        j["stdout"] = (j.get("stdout","") + "\n" + (res.get("stdout") or "")).strip()
        j["stderr"] = (j.get("stderr","") + "\n" + (res.get("stderr") or "")).strip()

        # If loveble file exists and readable, attach path
        if loveble_path.exists():
            j["loveble_path"] = str(loveble_path)
        else:
            j["loveble_path"] = None

        # Save final diagnostics to disk for later inspection
        try:
            diag = {
                "ok": False if not j.get("loveble_path") else True,
                "returncode": j.get("returncode"),
                "timed_out": j.get("timed_out"),
                "stdout": j.get("stdout"),
                "stderr": j.get("stderr"),
                "loveble_path": j.get("loveble_path"),
            }
            (outdir / "server_job_diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        except Exception:
            pass

    return

# ------------ Routes ------------
@app.get("/api/health")
async def health():
    return {"ok": True, "message": "backend is up"}

@app.get("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze(
    username: str,
    timeout: int = Query(DEFAULT_TIMEOUT_SECONDS, description="Max seconds to wait for analyzer (background job uses this)"),
    top_k: int = Query(8, description="Top repos forwarded"),
    use_cache: bool = Query(True),
    skip_stats: bool = Query(False)
):
    """
    Non-blocking analyze starter.
    - If loveble exists -> return it immediately.
    - If running -> return status running.
    - Else start background job and return started.
    """
    if not ANALYZE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"analyze_user.py not found at {ANALYZE_SCRIPT}")

    outdir = OUTPUT_ROOT / username
    loveble_path = outdir / "loveble_schema.json"

    # If loveble already present, return it immediately
    if loveble_path.exists():
        loveble = safe_load_json(loveble_path)
        return AnalyzeResponse(ok=True, loveble=loveble, stdout=None, stderr=None, status="done")

    # Acquire lock and check / start job
    async with jobs_lock:
        j = jobs.get(username)
        if j and j.get("running"):
            # Already running
            return AnalyzeResponse(ok=False, error="Analysis already running", status="running", stdout=tail_text(j.get("stdout","")), stderr=tail_text(j.get("stderr","")))
        # Not running -> start background task
        task = asyncio.create_task(background_analyze(username, top_k, use_cache, skip_stats, timeout))
        jobs[username] = {
            "task": task,
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "timed_out": False,
            "loveble_path": None,
            "error": None,
        }
        return AnalyzeResponse(ok=False, error="Analysis started", status="started", stdout=None, stderr=None)

@app.get("/api/analyze_status", response_model=StatusResponse)
async def api_analyze_status(username: str):
    """
    Poll this endpoint from frontend to get status, tails, and loveble readiness.
    """
    async with jobs_lock:
        j = jobs.get(username)

        # If no job and no loveble, say idle
        outdir = OUTPUT_ROOT / username
        loveble_path = outdir / "loveble_schema.json"

        if not j:
            if loveble_path.exists():
                return StatusResponse(
                    username=username,
                    status="done",
                    started_at=None,
                    finished_at=None,
                    returncode=0,
                    stdout_tail="",
                    stderr_tail="",
                    loveble_ready=True,
                    loveble_path=str(loveble_path)
                )
            else:
                return StatusResponse(username=username, status="idle", stdout_tail="", stderr_tail="", loveble_ready=False)

        # build response from job
        status = "running" if j.get("running") else "done"
        stdout_tail = tail_text(j.get("stdout",""), lines=40)
        stderr_tail = tail_text(j.get("stderr",""), lines=40)
        loveble_ready = bool(j.get("loveble_path")) or (loveble_path.exists())
        loveble_p = j.get("loveble_path") or (str(loveble_path) if loveble_path.exists() else None)

        return StatusResponse(
            username=username,
            status=status,
            started_at=j.get("started_at"),
            finished_at=j.get("finished_at"),
            returncode=j.get("returncode"),
            timed_out=j.get("timed_out", False),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            loveble_ready=loveble_ready,
            loveble_path=loveble_p,
            error=j.get("error")
        )

# ------------ Main ------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
