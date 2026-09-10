r"""
Multi-clip Phase 2: Content classification for a whole project.
Combines evidence from multiple uploaded videos into ONE classification
(genre, mood, format) for the overall edit, instead of one video at a time.

Usage:
    python classify_project.py --project myproject clip_a.mp4 clip_b.mp4 clip_c.mp4

Assumes analyze.py has already been run on every listed video.
"""

import sys
import os
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv()

FRAMES_PER_VIDEO = 3


def extract_frame_at(video_path, timestamp, output_path):
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
           "-frames:v", "1", "-q:v", "3", str(output_path)]
    subprocess.run(cmd, capture_output=True, check=True)


def sample_timestamps(scenes, n):
    if not scenes:
        return []
    total_start = scenes[0]["start"]
    total_end = scenes[-1]["end"]
    duration = total_end - total_start
    if duration <= 0:
        return [total_start]
    return [round(total_start + duration * (i + 0.5) / n, 2) for i in range(n)]


def call_gemini_with_retry(client, parts, max_attempts=4):
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Sending frames to Gemini (attempt {attempt}/{max_attempts})...")
            return client.models.generate_content(
                model="gemini-3.6-flash", contents=parts,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                print("Gemini's servers are still unavailable. Try again shortly.")
                raise
            print(f"Server busy ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def classify_project(video_names):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    all_transcript_bits = []
    parts = []
    total_duration_all = 0.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for vname in video_names:
            analysis_path = Path("analysis_output") / f"{vname}_analysis.json"
            if not analysis_path.exists():
                print(f"Error: {analysis_path} not found. Run analyze.py on {vname} first.")
                sys.exit(1)
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis = json.load(f)

            scenes = analysis["scenes"]
            transcript = analysis["transcript"]
            total_duration_all += scenes[-1]["end"] if scenes else 0

            timestamps = sample_timestamps(scenes, FRAMES_PER_VIDEO)
            for i, t in enumerate(timestamps):
                out_path = Path(tmp_dir) / f"{vname}_{i}.jpg"
                extract_frame_at(analysis["source_path"], t, out_path)
                if out_path.exists():
                    parts.append(types.Part.from_bytes(
                        data=out_path.read_bytes(), mime_type="image/jpeg"))

            text = " ".join(seg["text"] for seg in transcript)[:800]
            all_transcript_bits.append(f"[{vname}]: {text if text.strip() else '(no speech)'}")

        transcript_summary = "\n".join(all_transcript_bits)

        prompt = f"""You are analyzing raw, unedited footage from {len(video_names)} separate video
files that a user uploaded together to build ONE combined automatic edit.

You are shown sample frames from across all {len(video_names)} videos, in
order (grouped by video), plus a transcript excerpt from each.

Combined total footage duration across all videos: {total_duration_all:.1f} seconds

Transcript excerpts by video:
{transcript_summary}

Based on all of this, respond with ONLY a JSON object (no markdown fences,
no extra text) describing the OVERALL content, since these clips will be
edited together into one video:

{{
  "content_genre": "one of: vlog, talking-head, sports, gaming, anime, tutorial, event, product-demo, other",
  "genre_confidence": "high, medium, or low",
  "mood_energy": "one of: high-energy, calm, emotional, informative, comedic, dramatic",
  "recommended_output_format": "long-form or shorts",
  "format_reasoning": "one short sentence on why this format fits",
  "editing_style_notes": "1-2 short sentences on what pacing/cuts/style would suit this combined content"
}}"""

        parts.append(prompt)
        response = call_gemini_with_retry(client, parts)
        result = json.loads(response.text)

    result["source_videos"] = video_names
    return result


def run(project_name, video_paths):
    video_names = [Path(v).stem for v in video_paths]
    print(f"\n=== Classifying project: {project_name} ({len(video_names)} videos) ===")

    result = classify_project(video_names)

    print("\nResult:")
    print(json.dumps(result, indent=2))

    out_path = Path("analysis_output") / f"{project_name}_classification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="Video files, e.g. clip_a.mp4 clip_b.mp4")
    parser.add_argument("--project", required=True, help="Project name")
    args = parser.parse_args()

    run(args.project, args.videos)
