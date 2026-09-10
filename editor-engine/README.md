# AI Video Editor API

Backend API for the AI video editing pipeline: upload raw footage, get
back an AI-analyzed, AI-edited final video.

## Endpoints
- `POST /jobs` - upload a single video, starts the full editing pipeline
- `POST /projects` - upload multiple videos, edits them together into one
- `GET /jobs/{job_id}` - check job status
- `GET /jobs/{job_id}/download` - download the finished video
- `GET /health` - health check

## Deployment notes (Render free tier)
- Set `GEMINI_API_KEY` as an environment variable in Render's dashboard
  (Settings → Environment). Do not commit an `.env` file with this key.
- `WHISPER_MODEL_SIZE` is set to `base` in the Dockerfile to fit within
  Render free tier's 512MB RAM limit. Locally (no env var set),
  `analyze.py` still defaults to `medium` for better accuracy.
- Render's free tier sleeps after ~15 minutes of no HTTP traffic. Since
  jobs run in a background thread, a long job with no incoming requests
  during that window risks being killed mid-processing. Consider an
  external uptime monitor (e.g. UptimeRobot, free) pinging `/health`
  every few minutes while actively using the app.
