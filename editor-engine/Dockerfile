# Deployment image for the AI video editing API on Render's free tier.
# Free tier: 512MB RAM, no GPU - local PyTorch/Whisper does not fit in
# this limit at ANY model size (confirmed by repeated OOM crashes).
# Transcription on this deployed build uses Groq's hosted Whisper API
# instead (TRANSCRIPTION_BACKEND=groq) - no torch/whisper installed here
# at all. Local development on your own PC is unaffected: analyze.py
# defaults to TRANSCRIPTION_BACKEND=local when that env var isn't set.

FROM python:3.11-slim

# ffmpeg: needed for audio extraction (for Groq) and our own rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    TRANSCRIPTION_BACKEND=groq

WORKDIR /home/appuser/app

COPY --chown=appuser:appuser requirements-deploy.txt .
RUN pip install --user --no-cache-dir -r requirements-deploy.txt

COPY --chown=appuser:appuser . .

# Render provides the port to bind via the $PORT environment variable -
# it is NOT fixed like Hugging Face Spaces' port 7860.
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}