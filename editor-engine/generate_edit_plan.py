r"""
Phase 4: Edit plan generation.
Takes the selected moments (Phase 3) plus the classification (Phase 2)
and word-level transcript (Phase 1), and produces a precise, renderable
"edit decision list": exact timeline placement for each clip, transition
types between them, and word-by-word caption timing.

All numeric/timeline math is done deterministically in Python (never by
the LLM, which is error-prone at arithmetic) - Gemini is only asked for
the creative judgment calls: which transition type fits where, and
overall style notes.

Usage:
    python generate_edit_plan.py <video_path>

Assumes analyze.py, classify.py, and select_highlights.py have already
been run on this video.
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
    """Deterministically place each selected moment back-to-back on the
    output timeline. This is plain arithmetic - no LLM involved, so the
    numbers are guaranteed correct.

    NOTE: this assumes zero-gap hard cuts. It's intentionally naive at
    this stage - it's only used to give get_style_decisions() relative
    clip order/duration before transitions are even decided. Once
    transitions are known, call apply_transition_overlaps() to correct
    these positions before building captions from them."""
    clips = []
    timeline_cursor = 0.0
    for i, m in enumerate(selected_moments):
        duration = round(m["end"] - m["start"], 2)
        clips.append({
            "order": i,
            "scene_id": m.get("scene_id"),
            "source_start": m["start"],
            "source_end": m["end"],
            "duration": duration,
            "timeline_start": round(timeline_cursor, 2),
            "timeline_end": round(timeline_cursor + duration, 2),
            "reason": m.get("reason", "")
        })
        timeline_cursor += duration
    return clips


def apply_transition_overlaps(clips, transitions):
    """Recompute each clip's timeline position accounting for crossfade
    overlaps. A crossfade of duration d between two clips makes the
    combined result d seconds SHORTER than simply summing their
    durations, since the clips play simultaneously during the overlap.

    Confirmed necessary on real testing: without this, captions were
    computed against the naive back-to-back timeline while the actual
    render (which does apply crossfade overlaps) ran progressively
    faster than that - causing captions to drift later and later behind
    the real audio as crossfades accumulated through the edit."""
    if not clips:
        return clips

    transition_map = {t["between_clip_order"]: t for t in transitions}
    clips[0]["timeline_start"] = 0.0
    clips[0]["timeline_end"] = round(clips[0]["duration"], 2)
    cursor = clips[0]["timeline_end"]

    for i in range(1, len(clips)):
        t = transition_map.get(i - 1, {"type": "cut", "duration": 0})
        overlap = t["duration"] if t.get("type") == "crossfade" else 0.0
        clips[i]["timeline_start"] = round(cursor - overlap, 2)
        clips[i]["timeline_end"] = round(clips[i]["timeline_start"] + clips[i]["duration"], 2)
        cursor = clips[i]["timeline_end"]

    return clips


def build_captions(clips, transcript):
    """Map every word that falls inside a selected clip to its position
    on the OUTPUT timeline (not the source timeline) - words in trimmed-out
    sections are simply skipped since their clip never made the cut."""
    captions = []
    for clip in clips:
        offset = clip["timeline_start"] - clip["source_start"]
        for seg in transcript:
            for w in seg.get("words", []) or []:
                if w["start"] >= clip["source_start"] and w["end"] <= clip["source_end"]:
                    captions.append({
                        "text": w["word"],
                        "timeline_start": round(w["start"] + offset, 2),
                        "timeline_end": round(w["end"] + offset, 2)
                    })
    return merge_repeated_captions(captions)


def merge_repeated_captions(captions):
    """Collapse consecutive identical words (e.g. Whisper repeat-loop
    hallucinations, or genuine repeated speech) into one held caption
    instead of rapidly re-flashing the same word - reads as broken
    either way, regardless of whether the repetition was real speech."""
    if not captions:
        return captions

    merged = [dict(captions[0])]
    for cap in captions[1:]:
        if cap["text"].lower() == merged[-1]["text"].lower():
            merged[-1]["timeline_end"] = cap["timeline_end"]
        else:
            merged.append(dict(cap))
    return merged


def call_gemini_with_retry(client, prompt, max_attempts=4):
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Asking Gemini for transition/style decisions "
                  f"(attempt {attempt}/{max_attempts})...")
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                print(f"\nGemini's servers are still unavailable after "
                      f"{max_attempts} attempts. Try again shortly.")
                raise
            print(f"Server busy ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def get_style_decisions(clips, classification, output_format):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    clip_summary = [
        {
            "order": c["order"],
            "duration": c["duration"],
            "reason": c["reason"]
        }
        for c in clips
    ]

    num_transitions = max(len(clips) - 1, 0)

    prompt = f"""You are making editing style decisions for an automatic video edit.

Content genre: {classification.get('content_genre')}
Mood/energy: {classification.get('mood_energy')}
Output format: {output_format}
Editing style notes: {classification.get('editing_style_notes')}

There are {len(clips)} clips being stitched together in this order:
{json.dumps(clip_summary, indent=2)}

Decide the transition BETWEEN each consecutive pair of clips (there are
{num_transitions} transitions needed - between clip 0 and 1, clip 1 and
2, etc). Options:
- "cut": instant cut, duration 0. Best for high-energy, fast-paced content.
- "crossfade": a stylized dissolve between the two clips. Best for
  calmer, emotional, reflective content, or to soften a jump between
  very different-feeling moments. Typical duration: 0.3-0.6 seconds.
  When you choose "crossfade", also pick a "style" that fits the mood
  from this list (these are real, distinct visual transition effects,
  not just a plain fade):
  - "fade": classic smooth dissolve. Safe, versatile default.
  - "fadeblack": fades through black. Good for a somber/dramatic beat.
  - "dissolve": pixel-scatter dissolve. Slightly grittier than fade.
  - "wipeleft" / "wiperight" / "wipeup" / "wipedown": directional wipe.
    Good for energetic or purposeful movement between scenes.
  - "slideleft" / "slideright" / "slideup" / "slidedown": one clip
    slides the other off-frame. Punchy, works well for upbeat content.
  - "circleopen" / "circleclose": circular reveal. Distinctive,
    good for a focus-pulling or reveal moment.
  - "zoomin": zooms into the transition. Good for building intensity.
  - "smoothleft" / "smoothright": smooth directional blend, softer
    than a hard wipe.
  Vary the style across the edit rather than repeating the same one
  every time, and pick styles that match each specific moment's mood -
  a somber character beat and a high-energy action beat shouldn't
  necessarily use the same transition style.

Also decide:
- Whether to fade in from black at the very start and fade out to black
  at the very end (usually yes for a polished feel, especially shorts).
- A short caption style description (e.g. "bold white text, bottom
  center, one word highlighted at a time" - karaoke-style word captions
  are standard for shorts/reels).

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "transitions": [
    {{"between_clip_order": <int, the earlier clip's order number>, "type": "cut" or "crossfade", "duration": <float, seconds, 0 for cut>, "style": "<one of the style names above, only if type is crossfade>"}}
  ],
  "fade_in_at_start": <true or false>,
  "fade_out_at_end": <true or false>,
  "caption_style": "<short description>",
  "style_reasoning": "<1-2 sentences on the overall approach>"
}}"""

    response = call_gemini_with_retry(client, prompt)
    return json.loads(response.text)


def generate_edit_plan(video_path):
    video_path = Path(video_path)
    video_name = video_path.stem

    analysis_path = Path("analysis_output") / f"{video_name}_analysis.json"
    classification_path = Path("analysis_output") / f"{video_name}_classification.json"
    selection_path = Path("analysis_output") / f"{video_name}_selection.json"

    for p, phase in [(analysis_path, "analyze.py"), (classification_path, "classify.py"),
                      (selection_path, "select_highlights.py")]:
        if not p.exists():
            print(f"Error: {p} not found. Run {phase} first.")
            sys.exit(1)

    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    with open(classification_path, "r", encoding="utf-8") as f:
        classification = json.load(f)
    with open(selection_path, "r", encoding="utf-8") as f:
        selection = json.load(f)

    print(f"\n=== Generating edit plan: {video_name} ===")

    output_format = classification.get("recommended_output_format", "shorts")
    output_format = "shorts" if "short" in output_format.lower() else "long-form"
    resolution = TARGET_RESOLUTIONS[output_format]

    clips = build_timeline(selection["selected_moments"])
    style = get_style_decisions(clips, classification, output_format)
    clips = apply_transition_overlaps(clips, style.get("transitions", []))
    captions = build_captions(clips, analysis["transcript"])

    total_duration = clips[-1]["timeline_end"] if clips else 0.0

    edit_plan = {
        "video_file": video_name,
        "source_path": analysis["source_path"],
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

    print("\nResult:")
    print(json.dumps(edit_plan, indent=2))

    out_path = Path("analysis_output") / f"{video_name}_edit_plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(edit_plan, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return edit_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()

    generate_edit_plan(args.video_path)