r"""
Phase 2: Content classification.
Takes the JSON produced by analyze.py (Phase 1) plus the original video,
samples a few representative frames, and asks Gemini to classify the
content genre, mood/energy, and confirm the target output format.

Usage:
    python classify.py <video_path>
    python classify.py <video_path> --format shorts
    python classify.py <video_path> --format long

Assumes analyze.py has already been run on this video and produced
analysis_output\<name>_analysis.json in the same folder.
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import time

load_dotenv()

MAX_FRAMES = 6


def extract_frame_at(video_path, timestamp, output_path):
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3", str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def pick_sample_timestamps(scenes, max_frames=MAX_FRAMES):
    """Pick up to max_frames timestamps spread evenly across the full
    video duration, regardless of how many scenes were detected.
    This matters for clips with few/long scenes (e.g. one continuous
    take) - we still want multiple visual samples across time, not
    just one frame from the scene's midpoint."""
    if not scenes:
        return []

    total_start = scenes[0]["start"]
    total_end = scenes[-1]["end"]
    duration = total_end - total_start

    if duration <= 0:
        return [round(total_start, 2)]

    return [
        round(total_start + duration * (i + 0.5) / max_frames, 2)
        for i in range(max_frames)
    ]


def gather_sample_frames(video_path, scenes, tmp_dir):
    timestamps = pick_sample_timestamps(scenes)
    frame_paths = []
    for i, t in enumerate(timestamps):
        out_path = Path(tmp_dir) / f"sample_{i}.jpg"
        extract_frame_at(video_path, t, out_path)
        if out_path.exists():
            frame_paths.append(out_path)
    return frame_paths


def build_transcript_excerpt(transcript, max_chars=1500):
    full_text = " ".join(seg["text"] for seg in transcript)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "..."
    return full_text if full_text.strip() else "(no speech detected)"


def classify_content(video_path, analysis, user_target_format=None):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = gather_sample_frames(video_path, analysis["scenes"], tmp_dir)

        parts = []
        for fp in frame_paths:
            parts.append(types.Part.from_bytes(
                data=fp.read_bytes(),
                mime_type="image/jpeg"
            ))

        transcript_excerpt = build_transcript_excerpt(analysis["transcript"])
        num_scenes = len(analysis["scenes"])
        total_duration = analysis["scenes"][-1]["end"] if analysis["scenes"] else 0

        format_hint = (
            f"The user says they intend to publish this as: {user_target_format}."
            if user_target_format else
            "The user has not specified an intended output format."
        )

        prompt = f"""You are analyzing raw, unedited video footage to help an automatic
video editing tool decide how to cut it.

You are shown {len(frame_paths)} sample frames taken evenly across the footage
(in chronological order), plus a transcript excerpt of any speech.

Video stats:
- Total duration: {total_duration:.1f} seconds
- Number of detected shots/scenes: {num_scenes}

Transcript excerpt:
\"\"\"{transcript_excerpt}\"\"\"

{format_hint}

Based on the frames and transcript, respond with ONLY a JSON object (no markdown
fences, no extra text) with this exact structure:

{{
  "content_genre": "one of: vlog, talking-head, sports, gaming, anime, tutorial, event, product-demo, other",
  "genre_confidence": "high, medium, or low",
  "mood_energy": "one of: high-energy, calm, emotional, informative, comedic, dramatic",
  "recommended_output_format": "long-form or shorts",
  "format_reasoning": "one short sentence on why this format fits, considering duration and content",
  "editing_style_notes": "1-2 short sentences on what kind of pacing/cuts/style would suit this content"
}}"""

        parts.append(prompt)

        response = call_gemini_with_retry(client, parts)

        return json.loads(response.text)


def call_gemini_with_retry(client, parts, max_attempts=4):
    """Call Gemini with automatic retry on transient server errors
    (like 503 'high demand'). Uses exponential backoff: 5s, 10s, 20s..."""
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Sending {len(parts) - 1} frames to Gemini "
                  f"(attempt {attempt}/{max_attempts})...")
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                print(f"\nGemini's servers are still unavailable after "
                      f"{max_attempts} attempts. This is on Google's end - "
                      f"wait a few minutes and try again.")
                raise
            print(f"Server busy ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def run_classification(video_path, user_target_format=None):
    video_path = Path(video_path)
    video_name = video_path.stem
    analysis_path = Path("analysis_output") / f"{video_name}_analysis.json"

    if not analysis_path.exists():
        print(f"Error: {analysis_path} not found. Run analyze.py on this video first.")
        sys.exit(1)

    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    print(f"\n=== Classifying: {video_name} ===")
    print("Sampling frames and calling Gemini...")

    classification = classify_content(video_path, analysis, user_target_format)

    print("\nResult:")
    print(json.dumps(classification, indent=2))

    out_path = Path("analysis_output") / f"{video_name}_classification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(classification, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return classification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--format", choices=["shorts", "long"], default=None,
                         help="Your intended output format, if you already know it")
    args = parser.parse_args()

    run_classification(args.video_path, args.format)