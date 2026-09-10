r"""
Phase 3: Highlight / moment selection.
Given the scene + transcript data from analyze.py and the classification
from classify.py, ask Gemini to select which moments to keep for the
final edit, aiming for a duration that fits the recommended output format.

Usage:
    python select_highlights.py <video_path>

Assumes analyze.py and classify.py have already been run on this video.
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

# (min_seconds, max_seconds) for shorts; (min_fraction, max_fraction) of
# original duration for long-form. These are GUIDELINES, not hard caps -
# see the prompt below: never cut a genuinely good/necessary moment just
# to fit a number. The shorts max is set to match real platform limits
# (YouTube Shorts allows up to ~3 min) rather than an arbitrary low
# ceiling, so there's room to include real content when it's there.
TARGET_RANGES = {
    "shorts": (25, 180),
    "long-form": (0.7, 0.95)
}

FRAMES_PER_SILENT_SCENE = 4


def extract_frame_at(video_path, timestamp, output_path):
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
           "-frames:v", "1", "-q:v", "3", str(output_path)]
    subprocess.run(cmd, capture_output=True, check=True)


def build_scene_text_map(scenes, transcript):
    """Attach any transcript text (with word-level timestamps) that
    overlaps each scene's time range, so Gemini can see exactly what
    was said and choose cut points that land on real word boundaries."""
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
        target_min, target_max = lo, hi
        target_max = min(target_max, total_duration)
        target_min = min(target_min, target_max)
    else:
        target_min = round(total_duration * lo, 1)
        target_max = round(total_duration * hi, 1)

    return target_min, target_max


DIALOGUE_CRITICAL_GENRES = {"anime", "tutorial", "product-demo", "event", "talking-head", "vlog"}
VISUAL_PRIORITY_GENRES = {"sports", "gaming"}


def build_genre_guidance(content_genre):
    genre = (content_genre or "").lower()

    if genre in DIALOGUE_CRITICAL_GENRES:
        return f"""This content genre ({content_genre}) is narrative/dialogue-driven -
spoken lines are often ESSENTIAL for a viewer to understand what's
happening: who's talking, why an event matters, what's being explained.
Strongly prioritize scenes with meaningful speech, even over purely
visual/action moments, unless that speech is filler or repetitive.
A visually striking moment with NO context is much weaker than one
paired with the dialogue that explains it. If the duration budget is
tight, prefer keeping key expository or emotionally important lines
over silent visual beats. Do not build an edit that omits all spoken
dialogue if meaningful dialogue exists in the footage."""

    if genre in VISUAL_PRIORITY_GENRES:
        return f"""This content genre ({content_genre}) is primarily visual/action-driven -
dialogue or commentary is a bonus, not a requirement. Prioritize the
clearest, most exciting visual/action moments; only include speech if
it doesn't come at the cost of losing better action content."""

    return """Balance speech and visual interest based on what best represents
this content - include meaningful dialogue when it adds context, but
don't force it in if the visual moments are clearly stronger."""


def call_gemini_and_parse(client, parts, label="Gemini", max_attempts=4):
    """Calls Gemini and returns parsed JSON, retrying on BOTH transient
    server errors (503s) AND malformed/truncated JSON in the response
    itself - confirmed necessary on real testing: a long, complex prompt
    (frames + transcript + several instruction blocks) occasionally
    produces a response that gets cut off mid-structure, which used to
    crash immediately with no retry since only server errors were
    handled before. Also normalizes Gemini's occasional habit of
    wrapping the response in a list ([{...}] instead of {...})."""
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"{label} (attempt {attempt}/{max_attempts})...")
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            if isinstance(result, list):
                result = result[0]
            return result
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                print(f"\n{label}: Gemini's servers are still unavailable after "
                      f"{max_attempts} attempts. Try again shortly.")
                raise
            print(f"Server busy ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
        except json.JSONDecodeError as e:
            if attempt == max_attempts:
                print(f"\n{label}: Gemini returned malformed JSON "
                      f"{max_attempts} times in a row. Giving up.")
                raise
            print(f"Received malformed JSON ({e}). Retrying...")
            time.sleep(2)


def gather_scene_frames(video_path, scenes, tmp_dir):
    """Sample frames for any scene with no speech to anchor on, so the
    selector can actually SEE what's happening instead of guessing blind
    from duration alone. Scenes with real transcript text skip this -
    word timestamps already give a reliable anchor there."""
    frame_parts = []
    for scene in scenes:
        if scene.get("_has_words"):
            continue
        start, end = scene["start"], scene["end"]
        duration = max(end - start, 0.1)
        n = FRAMES_PER_SILENT_SCENE
        timestamps = [round(start + duration * (i + 0.5) / n, 2) for i in range(n)]
        for i, t in enumerate(timestamps):
            out_path = Path(tmp_dir) / f"scene{scene['scene_id']}_{i}.jpg"
            extract_frame_at(video_path, t, out_path)
            if out_path.exists():
                frame_parts.append(types.Part.from_bytes(
                    data=out_path.read_bytes(), mime_type="image/jpeg"))
                frame_parts.append(
                    f"^ Frame from scene_id={scene['scene_id']}, timestamp={t}s"
                )
    return frame_parts


def select_highlights(analysis, classification, video_path):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    scenes = analysis["scenes"]
    transcript = analysis["transcript"]
    total_duration = scenes[-1]["end"] if scenes else 0

    output_format = classification.get("recommended_output_format", "shorts")
    output_format = "shorts" if "short" in output_format.lower() else "long-form"

    target_min, target_max = compute_target_duration(total_duration, output_format)
    scene_text_map = build_scene_text_map(scenes, transcript)
    for s, meta in zip(scene_text_map, scenes):
        meta["_has_words"] = s["words"] is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_parts = gather_scene_frames(video_path, scenes, tmp_dir)

        genre_guidance = build_genre_guidance(classification.get('content_genre'))

        prompt = f"""You are selecting the best moments from raw footage to build an
automatic video edit.

Content genre: {classification.get('content_genre')}
Mood/energy: {classification.get('mood_energy')}
Target output format: {output_format}
Editing style notes: {classification.get('editing_style_notes')}

{genre_guidance}

Total source duration: {total_duration:.1f} seconds
Target final duration: roughly {target_min:.1f} to {target_max:.1f} seconds -
this is a GUIDELINE, not a hard requirement. If the source footage has
more genuinely good, necessary moments than fit in this range, INCLUDE
THEM ANYWAY rather than cutting something worthwhile just to hit a
number - a longer edit that tells the story properly beats a shorter
one that omits something important. Only trim toward the lower end of
the range if the source doesn't actually have enough strong content to
fill it. Never sacrifice a causally-important or otherwise essential
moment purely to stay under this target.

Here are all detected scenes in chronological order, each with its
timing and any speech spoken during it (null means no speech detected):

{json.dumps(scene_text_map, indent=2)}

For scenes with no speech, you are also shown sample frames from
across that scene (labeled with their scene_id and timestamp) so you
can see what's actually happening before choosing where to cut -
never guess a sub-range blindly from duration alone.

IMPORTANT for on-screen event text (kill confirmations, score
notifications, achievement popups, etc.): this text always appears
AFTER the actual action already happened. If you see such text in a
frame, the real highlight moment is the 1-3 seconds BEFORE that
frame's timestamp, not the frame itself - make sure your selected
window includes that lead-up, not just the confirmation text. Prefer
windows of at least 2-3 seconds so the actual action is visible, not
just its aftermath.

Just as important: do not cut away the INSTANT an action happens
(e.g. right as a hit-marker flashes). A satisfying highlight shows the
action AND its payoff - briefly hold on the confirmation (kill feed
text, counter updating, etc.) before the window ends, even if that
means extending a fraction of a second into the next scene. Cutting
away before the payoff is visible feels unsatisfying even at a fast,
high-energy pace - the goal is fast cuts BETWEEN complete moments, not
cutting off each moment before its reward is shown.

Also: after a confirmed hit/kill/impact moment (a hit-marker flash,
ragdoll, confirmation text, etc.), extend the window roughly 0.5-1
second PAST that moment before it ends, rather than cutting away the
instant it registers. Even fast-paced montages give a brief beat for
a kill to land before cutting - an instant zero-pause cut right at the
moment of impact reads as chaotic rather than satisfying, since the
viewer never gets to register the payoff.

Avoid redundancy: when several candidate scenes are just near-duplicate
reactions to the SAME single event (e.g. four different close-ups of
one character's shock/collapse/fall), include only the clearest one or
two rather than all of them. Use the duration freed up by cutting
redundant reaction shots to include causally-distinct beats instead -
especially any scene showing WHY something is happening (a cause), not
just more angles on THAT it happened (the effect). An edit that shows
only aftermath with no cause is confusing, even if each individual shot
looks good on its own.

SELF-CHECK before finalizing your answer: for every major consequence
or dramatic reaction you've included (a death, collapse, shock,
discovery, victory, etc.), verify you've also included the specific
moment that CAUSES it, if that causal moment exists in the footage and
was identifiable as significant. A compelling reaction shot should
never crowd out the action that explains it. If you notice you've shown
an effect without its cause, add the causal scene back in - even if it
means removing a lower-priority moment to make room for it.

IMPORTANT: a highlight window is NOT required to stay within a single
scene's boundaries. scene_id is just a reference point for where the
window begins - if the ideal window (e.g. a few seconds before and
after a kill) spans across a scene cut into the next scene, your
"start"/"end" should cross that boundary rather than stopping short at
the scene edge. Prioritize capturing the actual moment of action
correctly over respecting scene boundaries.

Select which scenes (or parts of scenes) should be kept in the final
edit. Prioritize scenes with meaningful speech, clear action, or visual
interest; deprioritize dead time, setup, filler words, or repetitive
moments. Keep selected moments in chronological order (do not reorder).
You may trim a scene's start/end rather than using the whole thing if
only part of it is worth keeping.

Where a scene includes a "words" list, your chosen "start" and "end"
for that moment MUST exactly match a word's "start" or "end" value from
that list - never an arbitrary time in between two words.

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "selected_moments": [
    {{
      "scene_id": <int, matching the scene_id from the input>,
      "start": <float, seconds - can be later than the scene's original start if trimmed>,
      "end": <float, seconds - can be earlier than the scene's original end if trimmed>,
      "reason": "<short phrase: why this moment made the cut>"
    }}
  ],
  "total_selected_duration": <float, sum of (end - start) across all selected moments>,
  "selection_reasoning": "<2-3 sentences on the overall selection strategy used>"
}}"""

        parts = frame_parts + [prompt]
        result = call_gemini_and_parse(client, parts, label="Selecting highlights")

        result = verify_causal_completeness(client, result, scene_text_map)
        return result


def snap_to_word_boundaries(scene):
    """When programmatically inserting a scene (not via the LLM, which
    is separately instructed to respect word boundaries), make sure we
    don't cut mid-word if the scene has speech - confirmed necessary on
    real testing: a scene added by the causal-completeness check cut off
    audio mid-word ("informat-") because it used the scene's raw visual-
    cut boundary directly instead of checking word timing."""
    words = scene.get("words")
    if not words:
        return scene["start"], scene["end"]

    start = scene["start"]
    end = scene["end"]

    # Trim start forward to the first word that starts at/after it.
    starts_after = [w["start"] for w in words if w["start"] >= start]
    if starts_after:
        start = min(starts_after)

    # Trim end backward to the last word that ends at/before it.
    ends_before = [w["end"] for w in words if w["end"] <= end]
    if ends_before:
        end = max(ends_before)

    if end <= start:
        return scene["start"], scene["end"]
    return start, end


def verify_causal_completeness(client, selection, scene_text_map):
    """Second, separate pass: check the selection for causal gaps and
    patch them in programmatically. Confirmed necessary on real testing -
    three separate attempts at wording the main prompt to prevent a
    known-critical causal scene from being dropped all failed at least
    once; asking one pass to both select AND self-verify wasn't reliable.
    Splitting verification into its own focused pass is a more robust
    pattern for exactly this kind of multi-part instruction."""
    selected_ids = {m["scene_id"] for m in selection.get("selected_moments", [])}

    verify_prompt = f"""You are reviewing an already-made highlight selection for ONE thing only:
causal completeness. Do not judge pacing, redundancy, or anything else -
only check whether any SELECTED consequence (a death, collapse, shock,
victory, revelation, etc.) is missing the specific scene that CAUSES it.

Currently selected scenes (by scene_id) and why each was picked:
{json.dumps([{"scene_id": m["scene_id"], "reason": m.get("reason", "")} for m in selection.get("selected_moments", [])], indent=2)}

Full list of ALL available scenes, for reference (to find any missing
causal scene among them):
{json.dumps(scene_text_map, indent=2)}

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "missing_causal_scenes": [
    {{"scene_id": <int, from the full scene list, NOT already selected>, "reason": "<why this scene is the necessary cause of something already selected>"}}
  ]
}}

If nothing is missing, return an empty list for missing_causal_scenes."""

    try:
        verify_result = call_gemini_and_parse(client, [verify_prompt], label="Causal-completeness check")
    except Exception as e:
        print(f"Causal-completeness check failed ({e}) - proceeding with original selection.")
        return selection

    missing = verify_result.get("missing_causal_scenes", [])
    missing = [m for m in missing if m.get("scene_id") not in selected_ids]

    if not missing:
        print("Causal-completeness check: no gaps found.")
        return selection

    print(f"Causal-completeness check: adding {len(missing)} missing scene(s) back in.")

    scenes_by_id = {s["scene_id"]: s for s in scene_text_map}
    for m in missing:
        scene = scenes_by_id.get(m["scene_id"])
        if not scene:
            continue
        start, end = snap_to_word_boundaries(scene)
        selection["selected_moments"].append({
            "scene_id": scene["scene_id"],
            "start": start,
            "end": end,
            "reason": f"[added by causal-completeness check] {m.get('reason', '')}"
        })

    selection["selected_moments"].sort(key=lambda m: m["start"])
    selection["total_selected_duration"] = round(
        sum(m["end"] - m["start"] for m in selection["selected_moments"]), 2
    )
    return selection


def load_overrides(video_name):
    """Load must-include overrides if a {video_name}_overrides.json file
    exists in analysis_output. Fully optional - if the file doesn't
    exist, returns an empty override set and behavior is identical to
    not having this feature at all.

    File format:
    {
      "must_include": [
        {"start": 47.03, "end": 48.8, "reason": "why this must be kept"}
      ]
    }

    Timestamps refer to the SOURCE video, not the output timeline - and
    stay valid even if scene-detection tuning ever changes scene_id
    numbering in the future, since they don't depend on it."""
    path = Path("analysis_output") / f"{video_name}_overrides.json"
    if not path.exists():
        return {"must_include": []}
    with open(path, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    return {"must_include": overrides.get("must_include", [])}


def apply_must_include_overrides(selection, overrides, scenes):
    """Final, fully deterministic step - guarantees inclusion regardless
    of what any AI judgment call (main selection or causal-completeness
    check) decided. Runs last, after all AI-driven steps."""
    must_include = overrides.get("must_include", [])
    if not must_include:
        return selection

    selected = selection["selected_moments"]

    for override in must_include:
        o_start, o_end = override["start"], override["end"]

        already_covered = any(
            m["start"] < o_end and m["end"] > o_start for m in selected
        )
        if already_covered:
            print(f"Override {o_start}-{o_end}s already covered by an "
                  f"existing selection - skipping.")
            continue

        # Snap to word boundaries if this range falls within a scene
        # that has speech, same safety as the causal-completeness patch.
        start, end = o_start, o_end
        containing_scene = next(
            (s for s in scenes if s["start"] <= o_start and s["end"] >= o_end),
            None
        )
        if containing_scene:
            start, end = snap_to_word_boundaries({
                "start": o_start, "end": o_end,
                "words": containing_scene.get("words")
            })

        print(f"Forcing in override: {start}-{end}s "
              f"({override.get('reason', 'no reason given')})")
        selected.append({
            "scene_id": containing_scene["scene_id"] if containing_scene else None,
            "start": start,
            "end": end,
            "reason": f"[must-include override] {override.get('reason', '')}"
        })

    selected.sort(key=lambda m: m["start"])
    selection["total_selected_duration"] = round(
        sum(m["end"] - m["start"] for m in selected), 2
    )
    return selection


def resolve_natural_language_overrides(client, video_path, overrides, scene_text_map):
    """Convert any override with a natural-language 'description' instead
    of explicit start/end into real timestamps, by showing Gemini sample
    frames from every speech-less scene plus each scene's speech text
    (reusing the same evidence the main selection pass gets) and asking
    which scene_id the description refers to.

    Entries that already have explicit start/end pass through untouched -
    this only fires for description-based entries, so hand-typed
    timestamps keep working exactly as before."""
    needs_resolution = [
        o for o in overrides.get("must_include", [])
        if "start" not in o and "description" in o
    ]
    if not needs_resolution:
        return overrides

    print(f"Resolving {len(needs_resolution)} natural-language override(s)...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_parts = []
        for scene in scene_text_map:
            if scene.get("words"):
                continue  # scenes with speech are identifiable by text alone
            mid = round((scene["start"] + scene["end"]) / 2, 2)
            out_path = Path(tmp_dir) / f"resolve_scene{scene['scene_id']}.jpg"
            extract_frame_at(video_path, mid, out_path)
            if out_path.exists():
                frame_parts.append(types.Part.from_bytes(
                    data=out_path.read_bytes(), mime_type="image/jpeg"))
                frame_parts.append(
                    f"^ Frame from scene_id={scene['scene_id']} "
                    f"(timestamp {mid}s, duration {scene['duration']}s)"
                )

        scene_summary = [
            {
                "scene_id": s["scene_id"], "start": s["start"], "end": s["end"],
                "speech": s["speech"]
            }
            for s in scene_text_map
        ]

        resolved_entries = []
        for override in needs_resolution:
            prompt = f"""A user wants a specific moment guaranteed to be included in a video
edit, described in their own words: "{override['description']}"

Here is every scene in the source video (by scene_id, timing, and any
speech), plus sample frames for scenes with no speech (shown above,
each labeled with its scene_id):

{json.dumps(scene_summary, indent=2)}

Identify which SINGLE scene_id this description most likely refers to.

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "scene_id": <int>,
  "confidence": "high", "medium", or "low",
  "matched_content": "<brief description of what's actually in that scene, so the user can verify this is correct>"
}}"""

            result = call_gemini_and_parse(
                client, frame_parts + [prompt],
                label=f"Resolving '{override['description'][:40]}...'"
            )

            scene = next((s for s in scene_text_map if s["scene_id"] == result["scene_id"]), None)
            if not scene:
                print(f"Could not resolve '{override['description']}' - scene_id "
                      f"{result.get('scene_id')} not found. Skipping.")
                continue

            print(f"Matched '{override['description']}' -> scene_id {scene['scene_id']} "
                  f"({scene['start']}-{scene['end']}s), confidence: {result['confidence']}. "
                  f"Gemini sees: {result['matched_content']}")
            if result["confidence"] == "low":
                print("  WARNING: low confidence match - verify this is actually correct.")

            resolved_entries.append({
                "start": scene["start"],
                "end": scene["end"],
                "reason": override.get("reason", override["description"])
            })

    already_explicit = [
        o for o in overrides.get("must_include", [])
        if "start" in o
    ]
    overrides["must_include"] = already_explicit + resolved_entries
    return overrides


def run_selection(video_path):
    video_path = Path(video_path)
    video_name = video_path.stem

    analysis_path = Path("analysis_output") / f"{video_name}_analysis.json"
    classification_path = Path("analysis_output") / f"{video_name}_classification.json"

    if not analysis_path.exists():
        print(f"Error: {analysis_path} not found. Run analyze.py first.")
        sys.exit(1)
    if not classification_path.exists():
        print(f"Error: {classification_path} not found. Run classify.py first.")
        sys.exit(1)

    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    with open(classification_path, "r", encoding="utf-8") as f:
        classification = json.load(f)

    print(f"\n=== Selecting highlights: {video_name} ===")

    selection = select_highlights(analysis, classification, video_path)

    overrides = load_overrides(video_name)
    if overrides["must_include"]:
        scenes_with_words = build_scene_text_map(analysis["scenes"], analysis["transcript"])
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        overrides = resolve_natural_language_overrides(client, video_path, overrides, scenes_with_words)
        print(f"\nApplying {len(overrides['must_include'])} must-include override(s)...")
        selection = apply_must_include_overrides(selection, overrides, scenes_with_words)

    print("\nResult:")
    print(json.dumps(selection, indent=2))

    out_path = Path("analysis_output") / f"{video_name}_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()

    run_selection(args.video_path)