r"""
Phase 1: Footage analysis.
Takes a video file, detects scenes and transcribes speech,
and saves the result as a structured JSON file.

Usage:
    python analyze.py path\to\your\video.mp4
"""

import sys
import json
import os
import subprocess
from pathlib import Path

from scenedetect import detect, ContentDetector
import whisper


def get_video_duration(video_path):
    """Get total duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def merge_short_scenes(scenes, min_duration=1.5):
    """
    Merge any scene shorter than min_duration into its neighbor.
    Real cuts rarely produce fragments this tiny - short fragments are
    almost always false positives from motion blur, rack focus, or
    something briefly passing in front of the lens.
    """
    if len(scenes) <= 1:
        return scenes

    scenes = [dict(s) for s in scenes]
    changed = True
    while changed and len(scenes) > 1:
        changed = False
        for i, s in enumerate(scenes):
            if s["duration"] < min_duration:
                if i == 0:
                    nxt = scenes[i + 1]
                    nxt["start"] = s["start"]
                    nxt["duration"] = round(nxt["end"] - nxt["start"], 2)
                else:
                    prev = scenes[i - 1]
                    prev["end"] = s["end"]
                    prev["duration"] = round(prev["end"] - prev["start"], 2)
                del scenes[i]
                changed = True
                break

    for i, s in enumerate(scenes):
        s["scene_id"] = i
    return scenes


def detect_scenes(video_path, threshold=35.0, min_scene_duration=1.5):
    """Split the video into shots and return their timestamps.
    If no cuts are found, treat the whole video as a single scene.

    threshold: ContentDetector's sensitivity (default ~27).
    min_scene_duration: fragments shorter than this (in seconds) get
    merged into a neighboring scene - see merge_short_scenes above."""
    scene_list = detect(video_path, ContentDetector(threshold=threshold))

    if not scene_list:
        duration = get_video_duration(video_path)
        return [{
            "scene_id": 0,
            "start": 0.0,
            "end": round(duration, 2),
            "duration": round(duration, 2)
        }]

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "scene_id": i,
            "start": start.seconds,
            "end": end.seconds,
            "duration": round(end.seconds - start.seconds, 2)
        })

    return merge_short_scenes(scenes, min_duration=min_scene_duration)


def transcribe_audio(video_path, model_size=None, language="en"):
    """
    Transcribe speech with word-level timestamps using local Whisper.
    model_size options: tiny, base, small, medium, large (bigger = more accurate, slower).

    If model_size isn't passed explicitly, reads WHISPER_MODEL_SIZE from
    the environment (defaulting to 'medium' if unset) - this lets the
    SAME codebase run 'medium' locally (where you have the RAM) and a
    smaller model on a resource-limited host, just by setting one
    environment variable there, instead of maintaining separate code.

    'medium' catches dialogue 'small' misses - confirmed on a real case
    where 'small' completely missed a line overlapping with alarm/sound
    effect audio - at the cost of roughly 2-3x 'small''s processing time
    on CPU. Drop to 'small' or 'base' if RAM/speed is constrained (e.g.
    a free-tier host with limited memory); step up to 'large' if
    accuracy still isn't good enough and you have the resources.

    language: force the expected language instead of relying on Whisper's
    auto-detection, which isn't reliable and doesn't get more accurate
    with bigger models - confirmed on a real case where 'medium' mis-
    detected clearly English audio as Hindi and transcribed everything
    as meaningless phonetic Hindi script. Set to None to auto-detect if
    you're processing genuinely multi-language content.
    """
    if model_size is None:
        model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")
    print(f"Loading Whisper '{model_size}' model (downloads once, then cached)...")
    model = whisper.load_model(model_size)

    result = model.transcribe(video_path, verbose=False, word_timestamps=True,
                               language=language)

    segments = []
    for seg in result["segments"]:
        # Whisper flags segments it's not actually confident contain real
        # speech via no_speech_prob - segments with a high score here are
        # usually hallucinated text from music/game sound effects/noise,
        # not genuine dialogue (this is exactly what produced "ael" and a
        # stray Chinese character from PUBG gunfire/UI sounds).
        if seg.get("no_speech_prob", 0) > 0.6:
            continue

        words = [
            {
                "word": w["word"].strip(),
                "start": round(w["start"], 2),
                "end": round(w["end"], 2)
            }
            for w in seg.get("words", [])
        ]

        # A single spoken word lasting several seconds is physically
        # implausible and reliably indicates a broken/hallucinated
        # timestamp, regardless of what caused it - reject the whole
        # segment rather than trust partially-broken timing.
        if any((w["end"] - w["start"]) > 4.0 for w in words):
            continue

        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "words": words
        })

    # Whisper sometimes hallucinates empty/zero-duration segments when
    # there's little or no real speech (common on gaming/ambient audio) -
    # filter those out rather than let them clutter downstream steps.
    segments = [s for s in segments if s["text"] and s["end"] > s["start"]]
    return segments


def analyze_video(video_path, output_dir="analysis_output"):
    video_path = str(video_path)
    video_name = Path(video_path).stem

    print(f"\n=== Analyzing: {video_name} ===")

    print("\n[1/2] Detecting scenes...")
    scenes = detect_scenes(video_path)
    print(f"Found {len(scenes)} scenes.")

    print("\n[2/2] Transcribing audio (this can take a few minutes on CPU)...")
    transcript = transcribe_audio(video_path)
    print(f"Found {len(transcript)} transcript segments.")

    analysis = {
        "video_file": video_name,
        "source_path": video_path,
        "scenes": scenes,
        "transcript": transcript
    }

    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"{video_name}_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nSaved analysis to: {output_path}")
    return analysis


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path_to_video>")
        sys.exit(1)

    video_file = sys.argv[1]
    analyze_video(video_file)