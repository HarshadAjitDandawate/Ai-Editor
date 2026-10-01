r"""
Phase 5: Render the final video.
Takes the edit plan (Phase 4) and the original source video, and produces
the actual finished MP4 using FFmpeg: trims + concatenates the selected
clips, applies transitions and fades, scales to the target resolution,
and burns in word-by-word captions.

Usage:
    python render.py <video_path>
    python render.py <video_path> --font "C:\path\to\font.ttf"

Assumes generate_edit_plan.py has already been run on this video.
"""

import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

DEFAULT_FONT = r"C:\Windows\Fonts\arialbd.ttf"

# Whitelist matching what generate_edit_plan.py offers the AI - guards
# against passing an invalid/unexpected style straight into the FFmpeg
# command if the model ever returns something outside this list.
VALID_XFADE_STYLES = {
    "fade", "fadeblack", "dissolve",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circleopen", "circleclose", "zoomin",
    "smoothleft", "smoothright"
}


def escape_drawtext(text):
    """Escape special characters for ffmpeg's drawtext filter text value.
    Backslash must be escaped; apostrophes are swapped for a unicode
    quote instead of escaped, since escaping a single quote inside an
    already single-quoted ffmpeg filter value is notoriously fragile."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")
    return text


def escape_path_for_filter(path):
    """Convert a filesystem path to a form safe to embed inside an
    ffmpeg filter string (forward slashes avoid backslash-escaping
    issues entirely, including with the drive-letter colon)."""
    return str(path).replace("\\", "/")


def build_trim_filters(clips):
    parts = []
    for c in clips:
        i = c["order"]
        parts.append(
            f"[0:v]trim=start={c['source_start']}:end={c['source_end']},"
            f"setpts=PTS-STARTPTS[v{i}];"
        )
        parts.append(
            f"[0:a]atrim=start={c['source_start']}:end={c['source_end']},"
            f"asetpts=PTS-STARTPTS[a{i}];"
        )
    return parts


def get_video_dimensions(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height",
           "-of", "csv=s=x:p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def build_scale_filters(clips, resolution, source_width, source_height):
    """Two strategies depending on how different the source and target
    shapes are:
    - Same orientation (both portrait or both landscape): scale to cover
      + center-crop. Works well since little content is lost.
    - Orientation mismatch (e.g. landscape gaming footage -> vertical
      shorts): a hard center-crop would cut off UI elements near the
      edges (minimap, kill counter, etc. - confirmed by testing on real
      gaming footage). Instead, use a blurred letterbox/pillarbox: the
      full original frame is preserved, centered on a softly blurred
      copy of itself filling the rest of the frame."""
    w, h = resolution["width"], resolution["height"]
    source_aspect = source_width / source_height
    target_aspect = w / h
    orientation_mismatch = (source_aspect >= 1) != (target_aspect >= 1)

    parts = []
    for c in clips:
        i = c["order"]
        if orientation_mismatch:
            parts.append(
                f"[v{i}]split=2[main{i}][bg{i}];"
                f"[bg{i}]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},gblur=sigma=20[bgblur{i}];"
                f"[main{i}]scale={w}:{h}:force_original_aspect_ratio=decrease[fg{i}];"
                f"[bgblur{i}][fg{i}]overlay=(W-w)/2:(H-h)/2[vs{i}];"
            )
        else:
            parts.append(
                f"[v{i}]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}:(iw-{w})/2:(ih-{h})/2[vs{i}];"
            )
    return parts


def get_video_fps(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def build_punchzoom_filters(clips, resolution, fps, total_zoom_fraction=0.08):
    """Add a subtle continuous push-in zoom over each clip's duration,
    using zoompan - the filter actually built for progressive per-frame
    zoom (crop's w/h are fixed at init and can't do this). Applied after
    scaling, on the already-correctly-sized clip.
    total_zoom_fraction: total zoom by the clip's last frame (0.08 = 8%)."""
    w, h = resolution["width"], resolution["height"]
    parts = []
    for c in clips:
        i = c["order"]
        duration = max(c["duration"], 0.1)
        total_frames = max(int(duration * fps), 1)
        increment = total_zoom_fraction / total_frames
        max_zoom = 1 + total_zoom_fraction
        parts.append(
            f"[vs{i}]zoompan=z='min(zoom+{increment},{max_zoom})':"
            f"d=1:s={w}x{h}:fps={fps}[vz{i}];"
        )
    return parts


def build_concat_chain(clips, transitions):
    """Chain clips together with cuts and crossfades.

    Crossfades are resolved FIRST, pairwise, directly between two fresh
    individually-scaled clips (never against a long accumulated concat
    chain) - xfade needs to know the exact duration of its first input,
    and after many chained concats, float rounding can drift enough to
    break it ("Failed to configure output pad"). Keeping every xfade
    shallow avoids that entirely. The resulting units (single clips or
    crossfade-merged pairs) are then joined with simple concat, which
    has no such fragility.

    Note: this doesn't support two crossfades sharing a clip (e.g.
    crossfade between 1-2 AND 2-3) - not needed by anything generated
    so far, but worth knowing if that ever comes up."""
    transition_map = {t["between_clip_order"]: t for t in transitions}
    parts = []

    units = []
    i = 0
    while i < len(clips):
        t = transition_map.get(i, {"type": "cut", "duration": 0}) if i < len(clips) - 1 else None

        if t and t["type"] == "crossfade" and t["duration"] > 0:
            d = t["duration"]
            v_a, a_a = f"vz{i}", f"a{i}"
            v_b, a_b = f"vz{i + 1}", f"a{i + 1}"
            offset = max(clips[i]["duration"] - d, 0)
            out_v, out_a = f"xfv{i}", f"xfa{i}"
            xfade_style = t.get("style") if t.get("style") in VALID_XFADE_STYLES else "fade"
            parts.append(
                f"[{v_a}][{v_b}]xfade=transition={xfade_style}:duration={d}:"
                f"offset={offset}[{out_v}];"
            )
            parts.append(f"[{a_a}][{a_b}]acrossfade=d={d}[{out_a}];")
            merged_duration = clips[i]["duration"] + clips[i + 1]["duration"] - d
            units.append({"video": out_v, "audio": out_a, "duration": merged_duration})
            i += 2
        else:
            units.append({
                "video": f"vz{i}", "audio": f"a{i}",
                "duration": clips[i]["duration"]
            })
            i += 1

    current_v, current_a = units[0]["video"], units[0]["audio"]
    cumulative_duration = units[0]["duration"]

    for idx, u in enumerate(units[1:]):
        out_v, out_a = f"cout{idx}v", f"cout{idx}a"
        parts.append(
            f"[{current_v}][{current_a}][{u['video']}][{u['audio']}]"
            f"concat=n=2:v=1:a=1[{out_v}][{out_a}];"
        )
        cumulative_duration += u["duration"]
        current_v, current_a = out_v, out_a

    return parts, current_v, current_a, cumulative_duration


def build_fade_filters(video_label, audio_label, total_duration, fade_in, fade_out):
    parts = []
    v, a = video_label, audio_label
    fade_duration = 0.5

    if fade_in:
        parts.append(f"[{v}]fade=t=in:st=0:d={fade_duration}[vfi];")
        v = "vfi"
    if fade_out:
        start = max(total_duration - fade_duration, 0)
        parts.append(f"[{v}]fade=t=out:st={start}:d={fade_duration}[vfo];")
        v = "vfo"
    if fade_in:
        parts.append(f"[{a}]afade=t=in:st=0:d={fade_duration}[afi];")
        a = "afi"
    if fade_out:
        start = max(total_duration - fade_duration, 0)
        parts.append(f"[{a}]afade=t=out:st={start}:d={fade_duration}[afo];")
        a = "afo"

    return parts, v, a


def format_ass_time(t):
    """Convert seconds to ASS subtitle timestamp format: H:MM:SS.cc"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass_subtitle_file(video_name, captions, resolution):
    """Write a .ass subtitle file with one word per line, timed exactly
    to the caption cues. This is far more robust than chaining dozens of
    drawtext filters on the command line - it sidesteps Windows command-
    line encoding issues entirely, since Python writes the file directly
    as UTF-8."""
    w, h = resolution["width"], resolution["height"]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,2,2,40,40,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    for cap in captions:
        text = cap["text"].replace("{", "(").replace("}", ")")
        start = format_ass_time(cap["timeline_start"])
        end = format_ass_time(cap["timeline_end"])
        # Pop-in animation: word appears at 130% size and shrinks to
        # 100% over its first 150ms, giving each word a punchy "pop"
        # instead of just appearing statically.
        pop_tag = r"{\fscx130\fscy130\t(0,150,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pop_tag}{text}\n")

    ass_path = Path(f"{video_name}_captions.ass")
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.writelines(lines)

    return ass_path


def render_video(video_path, font_path=DEFAULT_FONT):
    video_path = Path(video_path)
    video_name = video_path.stem
    plan_path = Path("analysis_output") / f"{video_name}_edit_plan.json"

    if not plan_path.exists():
        print(f"Error: {plan_path} not found. Run generate_edit_plan.py first.")
        sys.exit(1)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    clips = plan["clips"]
    transitions = plan["transitions"]
    resolution = plan["output_resolution"]

    print(f"\n=== Rendering: {video_name} ===")
    print(f"{len(clips)} clips -> {resolution['width']}x{resolution['height']}, "
          f"{len(plan['captions'])} caption words")

    fps = get_video_fps(video_path)
    source_width, source_height = get_video_dimensions(video_path)

    filter_parts = []
    filter_parts += build_trim_filters(clips)
    filter_parts += build_scale_filters(clips, resolution, source_width, source_height)
    filter_parts += build_punchzoom_filters(clips, resolution, fps)

    concat_parts, v_label, a_label, total_duration = build_concat_chain(clips, transitions)
    filter_parts += concat_parts

    fade_parts, v_label, a_label = build_fade_filters(
        v_label, a_label, total_duration,
        plan.get("fade_in_at_start", False), plan.get("fade_out_at_end", False)
    )
    filter_parts += fade_parts

    caption_parts = []
    if plan["captions"]:
        ass_path = build_ass_subtitle_file(video_name, plan["captions"], resolution)
        ass_filename = escape_path_for_filter(str(ass_path))
        caption_parts.append(f"[{v_label}]subtitles='{ass_filename}'[vcap];")
        v_label = "vcap"
    filter_parts += caption_parts

    filter_complex = " ".join(filter_parts)

    output_dir = Path("rendered_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{video_name}_final.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", f"[{v_label}]",
        "-map", f"[{a_label}]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-threads", "1",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ]

    print("\nRunning FFmpeg (this may take a minute or two)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_log_path = Path("ffmpeg_error.log")
        error_log_path.write_text(result.stderr, encoding="utf-8")

        lines = result.stderr.splitlines()
        error_lines = [
            l for l in lines
            if any(kw in l for kw in ("rror", "nvalid", "ould not", "ailed", "o such"))
            and "embedded font" not in l
        ]

        print(f"\nFFmpeg failed. Full log saved to {error_log_path}\n")
        if error_lines:
            print("Likely relevant line(s):")
            for l in error_lines[:10]:
                print(l)
        else:
            print("Last part of output:")
            print(result.stderr[-1500:])
        sys.exit(1)

    print(f"\nDone! Final video saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--font", default=DEFAULT_FONT,
                         help=r"Path to a .ttf font file (default: Windows Arial Bold)")
    args = parser.parse_args()

    render_video(args.video_path, args.font)