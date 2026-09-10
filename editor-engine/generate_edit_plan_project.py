r"""
Multi-clip Phase 4: Edit plan generation across multiple videos.

Usage:
    python generate_edit_plan_project.py --project myproject

Assumes classify_project.py and select_highlights_project.py have
already been run for this project.
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

TARGET_RESOLUTIONS = {
    "shorts": {"width": 1080, "height": 1920},
    "long-form": {"width": 1920, "height": 1080}
}


def build_timeline(selected_moments):
    clips = []
    cursor = 0.0
    for i, m in enumerate(selected_moments):
        duration = round(m["end"] - m["start"], 2)
        clips.append({
            "order": i,
            "source_video": m["source_video"],
            "scene_id": m.get("scene_id"),
            "source_start": m["start"],
            "source_end": m["end"],
            "duration": duration,
            "timeline_start": round(cursor, 2),
            "timeline_end": round(cursor + duration, 2),
            "reason": m.get("reason", "")
        })
        cursor += duration
    return clips


def build_captions(clips, transcripts_by_video):
    captions = []
    for clip in clips:
        transcript = transcripts_by_video[clip["source_video"]]
        offset = clip["timeline_start"] - clip["source_start"]
        for seg in transcript:
            for w in seg.get("words", []) or []:
                if w["start"] >= clip["source_start"] and w["end"] <= clip["source_end"]:
                    captions.append({
                        "text": w["word"],
                        "timeline_start": round(w["start"] + offset, 2),
                        "timeline_end": round(w["end"] + offset, 2)
                    })
    return captions


def call_gemini_with_retry(client, prompt, max_attempts=4):
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Asking Gemini for transition/style decisions "
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


def get_style_decisions(clips, classification, output_format):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    clip_summary = [
        {"order": c["order"], "source_video": c["source_video"],
         "duration": c["duration"], "reason": c["reason"]}
        for c in clips
    ]
    num_transitions = max(len(clips) - 1, 0)

    prompt = f"""You are making editing style decisions for an automatic video edit built
from multiple source videos.

Content genre: {classification.get('content_genre')}
Mood/energy: {classification.get('mood_energy')}
Output format: {output_format}
Editing style notes: {classification.get('editing_style_notes')}

There are {len(clips)} clips being stitched together in this order:
{json.dumps(clip_summary, indent=2)}

Decide the transition BETWEEN each consecutive pair of clips
({num_transitions} transitions needed). Options:
- "cut": instant cut, duration 0.
- "crossfade": a stylized dissolve, typical 0.3-0.6s. Consider using a
  crossfade specifically where "source_video" changes between two
  clips, since that can soften an otherwise jarring jump between
  different footage sources - but only where it fits the mood. When you
  choose "crossfade", also pick a "style" that fits the mood from this
  list (real, distinct transition effects, not just a plain fade):
  "fade", "fadeblack", "dissolve", "wipeleft", "wiperight", "wipeup",
  "wipedown", "slideleft", "slideright", "slideup", "slidedown",
  "circleopen", "circleclose", "zoomin", "smoothleft", "smoothright".
  Vary the style across the edit rather than repeating the same one.

Also decide:
- Whether to fade in from black at the start and fade out at the end.
- A short caption style description (karaoke-style word captions are
  standard for shorts/reels).

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "transitions": [
    {{"between_clip_order": <int>, "type": "cut" or "crossfade", "duration": <float>, "style": "<style name, only if crossfade>"}}
  ],
  "fade_in_at_start": <true or false>,
  "fade_out_at_end": <true or false>,
  "caption_style": "<short description>",
  "style_reasoning": "<1-2 sentences>"
}}"""

    response = call_gemini_with_retry(client, prompt)
    return json.loads(response.text)


def run(project_name):
    classification_path = Path("analysis_output") / f"{project_name}_classification.json"
    selection_path = Path("analysis_output") / f"{project_name}_selection.json"

    for p, phase in [(classification_path, "classify_project.py"),
                      (selection_path, "select_highlights_project.py")]:
        if not p.exists():
            print(f"Error: {p} not found. Run {phase} first.")
            sys.exit(1)

    with open(classification_path, "r", encoding="utf-8") as f:
        classification = json.load(f)
    with open(selection_path, "r", encoding="utf-8") as f:
        selection = json.load(f)

    video_names = classification["source_videos"]
    transcripts_by_video = {}
    source_paths_by_video = {}
    for vname in video_names:
        analysis_path = Path("analysis_output") / f"{vname}_analysis.json"
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        transcripts_by_video[vname] = analysis["transcript"]
        source_paths_by_video[vname] = analysis["source_path"]

    print(f"\n=== Generating edit plan for project: {project_name} ===")

    output_format = classification.get("recommended_output_format", "shorts")
    output_format = "shorts" if "short" in output_format.lower() else "long-form"
    resolution = TARGET_RESOLUTIONS[output_format]

    clips = build_timeline(selection["selected_moments"])
    captions = build_captions(clips, transcripts_by_video)
    style = get_style_decisions(clips, classification, output_format)

    total_duration = clips[-1]["timeline_end"] if clips else 0.0

    edit_plan = {
        "project_name": project_name,
        "source_videos": video_names,
        "source_paths": source_paths_by_video,
        "output_format": output_format,
        "output_resolution": resolution,
        "total_duration": total_duration,
        "clips": clips,
        "transitions": style.get("transitions", []),
        "fade_in_at_start": style.get("fade_in_at_start", False),
        "fade_out_at_end": style.get("fade_out_at_end", False),
        "caption_style": style.get("caption_style", ""),
        "style_reasoning": style.get("style_reasoning", ""),
        "captions": captions
    }

    print(f"\n{len(clips)} clips from {len(video_names)} videos, "
          f"total duration {total_duration}s")
    summary = {k: v for k, v in edit_plan.items() if k != "captions"}
    print(json.dumps(summary, indent=2))
    print(f"(+{len(captions)} caption words, omitted from printout for brevity)")

    out_path = Path("analysis_output") / f"{project_name}_edit_plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(edit_plan, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return edit_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    run(args.project)