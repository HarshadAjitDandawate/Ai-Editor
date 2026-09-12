r"""
Phase 6: API server.
Wraps the existing pipeline (analyze -> classify -> select -> edit plan
-> render) behind a real HTTP API, so a frontend (e.g. the Bolt.new app)
can upload a video, poll job status, and download the finished result -
instead of running scripts by hand in PowerShell.

Runs each job in a background thread (simple, no extra infrastructure
like Redis needed) with an in-memory job store. Fine for local dev and
a single-user MVP; swap for a real task queue (Celery+Redis) before
scaling to multiple concurrent users.

Usage:
    pip install fastapi uvicorn python-multipart --break-system-packages
    uvicorn api_server:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test UI.
"""

import uuid
import threading
import traceback
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analyze import analyze_video
from classify import run_classification
from select_highlights import run_selection
from generate_edit_plan import generate_edit_plan
from render import render_video
from classify_project import run as run_classify_project
from select_highlights_project import run as run_select_project
from generate_edit_plan_project import run as run_edit_plan_project
from render_project import render_project

app = FastAPI(title="AI Video Editor API")

# Dev-only: allows any frontend origin to call this API. Restrict this
# to your actual frontend's domain before this goes anywhere near
# production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("api_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job store: {job_id: {...}}. Lost on server restart - that's
# fine for local dev, not fine for production (use a real database then).
jobs = {}
jobs_lock = threading.Lock()


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, analyzing, classifying, selecting, planning, rendering, done, failed
    video_name: Optional[str] = None
    error: Optional[str] = None
    download_url: Optional[str] = None


def update_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


ACTIVE_STATUSES = {"queued", "analyzing", "classifying", "selecting", "planning", "rendering"}


def find_any_active_job():
    """Global concurrency limit: only ONE job at all, regardless of name,
    may run at a time. Confirmed necessary on real testing: two DIFFERENT
    jobs running concurrently (each loading Whisper + OpenCV) exhausted
    memory on Render's free 512MB single-core tier and silently crashed
    the whole container (OOM-kill - no error logged, just an abrupt
    restart with all in-memory job data wiped). This never showed up
    locally, where there's far more RAM/CPU headroom to run several jobs
    at once - it's specifically a constraint of this deployment tier."""
    with jobs_lock:
        for job in jobs.values():
            if job["status"] in ACTIVE_STATUSES:
                return job
    return None


def run_pipeline(job_id, video_path, target_format):
    video_name = Path(video_path).stem
    try:
        update_job(job_id, status="analyzing")
        analyze_video(video_path)

        update_job(job_id, status="classifying")
        run_classification(video_path, target_format)

        update_job(job_id, status="selecting")
        run_selection(video_path)

        update_job(job_id, status="planning")
        generate_edit_plan(video_path)

        update_job(job_id, status="rendering")
        output_path = render_video(video_path)

        update_job(
            job_id,
            status="done",
            download_url=f"/jobs/{job_id}/download"
        )
    except Exception as e:
        update_job(
            job_id,
            status="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        )


def run_project_pipeline(job_id, project_name, video_paths):
    try:
        update_job(job_id, status="analyzing")
        for vp in video_paths:
            analyze_video(vp)

        update_job(job_id, status="classifying")
        run_classify_project(project_name, video_paths)

        update_job(job_id, status="selecting")
        run_select_project(project_name)

        update_job(job_id, status="planning")
        run_edit_plan_project(project_name)

        update_job(job_id, status="rendering")
        render_project(project_name)

        update_job(
            job_id,
            status="done",
            download_url=f"/jobs/{job_id}/download"
        )
    except Exception as e:
        update_job(
            job_id,
            status="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        )


@app.post("/projects", response_model=JobStatus)
async def create_project_job(files: List[UploadFile] = File(...), project_name: str = Form(...)):
    """Upload multiple videos and start the combined multi-clip editing
    pipeline on them - the AI selects the best moments across ALL of
    them and edits them together into one video."""
    existing = find_any_active_job()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Another job is already running (job_id: {existing['job_id']}, "
                    f"video: '{existing['video_name']}', status: {existing['status']}). "
                    f"This deployment only supports one job at a time due to limited "
                    f"server resources - wait for it to finish before submitting another."
        )

    job_id = str(uuid.uuid4())
    saved_paths = []

    for f in files:
        video_path = UPLOAD_DIR / f.filename
        with open(video_path, "wb") as out:
            out.write(await f.read())
        saved_paths.append(str(video_path))

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "video_name": project_name,
            "error": None,
            "download_url": None
        }

    thread = threading.Thread(
        target=run_project_pipeline,
        args=(job_id, project_name, saved_paths),
        daemon=True
    )
    thread.start()

    return jobs[job_id]


@app.post("/jobs", response_model=JobStatus)
async def create_job(file: UploadFile = File(...), target_format: Optional[str] = None):
    """Upload a single video and start the full editing pipeline on it.
    target_format: optional 'shorts' or 'long', if the user already knows
    which they want (otherwise the AI decides in Phase 2)."""
    video_name = Path(file.filename).stem
    existing = find_any_active_job()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Another job is already running (job_id: {existing['job_id']}, "
                    f"video: '{existing['video_name']}', status: {existing['status']}). "
                    f"This deployment only supports one job at a time due to limited "
                    f"server resources - wait for it to finish before submitting another."
        )

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / file.filename

    with open(video_path, "wb") as f:
        f.write(await file.read())

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "video_name": Path(file.filename).stem,
            "error": None,
            "download_url": None
        }

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, str(video_path), target_format),
        daemon=True
    )
    thread.start()

    return jobs[job_id]


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/download")
async def download_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not finished (status: {job['status']})")

    output_path = Path("rendered_output") / f"{job['video_name']}_final.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered file not found")

    return FileResponse(output_path, media_type="video/mp4", filename=output_path.name)


@app.get("/health")
async def health():
    return {"status": "ok"}