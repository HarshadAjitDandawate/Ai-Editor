# Deployment image for the AI video editing API on Render's free tier.
# Free tier: 512MB RAM, no GPU - nowhere near enough for the 'medium'
# Whisper model (~1.5GB+). WHISPER_MODEL_SIZE is set to 'base' here to
# fit within that limit. analyze.py reads this env var automatically -
# locally (no env var set) it still defaults to 'medium' for full
# accuracy, since you have the RAM for that on your own PC.

FROM python:3.11-slim

# ffmpeg: needed by Whisper (audio extraction) and our own rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    WHISPER_MODEL_SIZE=tiny

WORKDIR /home/appuser/app

# Install PyTorch CPU-only build FIRST, before requirements.txt - pip
# would otherwise pull the default GPU build as a transitive dependency
# of openai-whisper, which is much larger and useless without a GPU.
RUN pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the (smaller, RAM-appropriate) Whisper model at BUILD
# time, not first request - Render's free tier sleeps after ~15 min of
# inactivity, so without this every wake-up would redownload the model
# before the first request could complete.
RUN python -c "import whisper; whisper.load_model('tiny')"

COPY --chown=appuser:appuser . .

# Render provides the port to bind via the $PORT environment variable -
# it is NOT fixed like Hugging Face Spaces' port 7860.
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}