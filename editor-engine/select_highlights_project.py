r"""
Multi-clip Phase 3: Highlight selection across multiple videos.

Usage:
    python select_highlights_project.py --project myproject

Assumes classify_project.py has already been run for this project (its
output records which source videos are included).
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv()

TARGET_RANGES = {
    "shorts": (25, 60),
    "long-form": (0.7, 0.95)
}


def build_scene_text_map(video_name, scenes, transcript):
    scene_texts = []
    for scene in scenes:
        overlapping = [
            seg for seg in transcript
            if seg["start"] < scene["end"] and seg["end"] > scene["start"]
        ]
        all_words = []
        for seg in overlapping:
            all_words.extend(seg.get("words", []))

        scene_texts.append({
            "source_video": video_name,
            "scene_id": scene["scene_id"],
            "start": scene["start"],
            "end": scene["end"],
            "duration": scene["duration"],
            "speech": " ".join(seg["text"] for seg in overlapping) if overlapping else None,
            "words": all_words if all_words else None
        })
    return scene_texts


def compute_target_duration(total_duration, output_format):
    if output_format not in TARGET_RANGES:
        output_format = "shorts"
    lo, hi = TARGET_RANGES[output_format]
    if output_format == "shorts":
        target_max = min(hi, total_duration)
        target_min = min(lo, target_max)
    else:
        target_min = round(total_duration * lo, 1)
        target_max = round(total_duration * hi, 1)
    return target_min, target_max


def call_gemini_with_retry(client, prompt, max_attempts=4):
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Asking Gemini to select highlights across all clips "
                  f"(attempt {attempt}/{max_attempts})...")
            return client.models.generate_content(
                model="gemini-3.6-flash", contents=[prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                print("Gemini's servers are still unavailable. Try again shortly.")
                raise
            print(f"Server busy ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def select_highlights_project(classification):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    video_names = classification["source_videos"]

    all_scene_maps = []
    total_duration = 0.0
    for vname in video_names:
        analysis_path = Path("analysis_output") / f"{vname}_analysis.json"
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        scenes = analysis["scenes"]
        transcript = analysis["transcript"]
        total_duration += scenes[-1]["end"] if scenes else 0
        all_scene_maps.extend(build_scene_text_map(vname, scenes, transcript))

    output_format = classification.get("recommended_output_format", "shorts")
    output_format = "shorts" if "short" in output_format.lower() else "long-form"
    target_min, target_max = compute_target_duration(total_duration, output_format)

    prompt = f"""You are selecting the best moments from raw footage across {len(video_names)}
separate video files (in this order: {video_names}) to build ONE combined
automatic video edit.

Content genre: {classification.get('content_genre')}
Mood/energy: {classification.get('mood_energy')}
Target output format: {output_format}
Editing style notes: {classification.get('editing_style_notes')}

Combined total source duration: {total_duration:.1f} seconds
Target final duration: between {target_min:.1f} and {target_max:.1f} seconds

Here are all detected scenes across all videos, each tagged with which
source video it came from, in chronological order within each video:

{json.dumps(all_scene_maps, indent=2)}

Select which scenes (or parts of scenes) should be kept in the final
edit, drawing from whichever videos have the best moments. Prioritize
meaningful speech, clear action, or visual interest; deprioritize dead
time, filler, or repetitive moments. IMPORTANT: keep all moments from
the same source video in their original chronological order relative to
each other, and keep video groups in the order given above
({video_names}) - do not interleave moments from different videos.

Where a scene includes a "words" list, your chosen "start" and "end" for
that moment MUST exactly match a word's "start" or "end" value from that
list - never an arbitrary time in between two words. Scenes with no
"words" (null) have no speech - trim those freely based on visual pacing.

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "selected_moments": [
    {{
      "source_video": "<string, matching one of {video_names}>",
      "scene_id": <int>,
      "start": <float, seconds>,
      "end": <float, seconds>,
      "reason": "<short phrase>"
    }}
  ],
  "total_selected_duration": <float>,
  "selection_reasoning": "<2-3 sentences>"
}}"""

    response = call_gemini_with_retry(client, prompt)
    return json.loads(response.text)


def run(project_name):
    classification_path = Path("analysis_output") / f"{project_name}_classification.json"
    if not classification_path.exists():
        print(f"Error: {classification_path} not found. Run classify_project.py first.")
        sys.exit(1)

    with open(classification_path, "r", encoding="utf-8") as f:
        classification = json.load(f)

    print(f"\n=== Selecting highlights for project: {project_name} ===")

    selection = select_highlights_project(classification)

    print("\nResult:")
    print(json.dumps(selection, indent=2))

    out_path = Path("analysis_output") / f"{project_name}_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    run(args.project)
