r"""
Multi-clip Phase 5: Render the final video from a multi-video edit plan.
Unlike render.py (single source file), this opens one FFmpeg input per
source video and routes each clip's trim filter to the correct input
index based on its "source_video" field.

Usage:
    python render_project.py --project myproject

Assumes generate_edit_plan_project.py has already been run for this project.
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

# Whitelist matching what generate_edit_plan_project.py offers the AI -
# see render.py for the full explanation.
VALID_XFADE_STYLES = {
    "fade", "fadeblack", "dissolve",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circleopen", "circleclose", "zoomin",
    "smoothleft", "smoothright"
}


def escape_path_for_filter(path):
    return str(path).replace("\\", "/")


def build_trim_filters(clips, input_index_by_video):
    parts = []
    for c in clips:
        i = c["order"]
        idx = input_index_by_video[c["source_video"]]
        parts.append(
            f"[{idx}:v]trim=start={c['source_start']}:end={c['source_end']},"
            f"setpts=PTS-STARTPTS[v{i}];"
        )
        parts.append(
            f"[{idx}:a]atrim=start={c['source_start']}:end={c['source_end']},"
            f"asetpts=PTS-STARTPTS[a{i}];"
        )
    return parts


def build_scale_filters(clips, resolution):
    w, h = resolution["width"], resolution["height"]
    parts = []
    for c in clips:
        i = c["order"]
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


def build_punchzoom_filters(clips, resolution, fps_by_video, total_zoom_fraction=0.08):
    """Subtle continuous push-in zoom per clip, using each clip's own
    source video's fps (different source videos in a project can have
    different frame rates)."""
    w, h = resolution["width"], resolution["height"]
    parts = []
    for c in clips:
        i = c["order"]
        fps = fps_by_video[c["source_video"]]
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
    """See render.py's build_concat_chain for the full explanation:
    crossfades are resolved first, pairwise, directly between two fresh
    clips (never against a long accumulated chain), then all resulting
    units are joined with simple concat."""
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
    fd = 0.5
    if fade_in:
        parts.append(f"[{v}]fade=t=in:st=0:d={fd}[vfi];")
        v = "vfi"
    if fade_out:
        start = max(total_duration - fd, 0)
        parts.append(f"[{v}]fade=t=out:st={start}:d={fd}[vfo];")
        v = "vfo"
    if fade_in:
        parts.append(f"[{a}]afade=t=in:st=0:d={fd}[afi];")
        a = "afi"
    if fade_out:
        start = max(total_duration - fd, 0)
        parts.append(f"[{a}]afade=t=out:st={start}:d={fd}[afo];")
        a = "afo"
    return parts, v, a


def format_ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass_subtitle_file(project_name, captions, resolution):
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
        pop_tag = r"{\fscx130\fscy130\t(0,150,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pop_tag}{text}\n")

    ass_path = Path(f"{project_name}_captions.ass")
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.writelines(lines)
    return ass_path


def render_project(project_name):
    plan_path = Path("analysis_output") / f"{project_name}_edit_plan.json"
    if not plan_path.exists():
        print(f"Error: {plan_path} not found. Run generate_edit_plan_project.py first.")
        sys.exit(1)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    clips = plan["clips"]
    transitions = plan["transitions"]
    resolution = plan["output_resolution"]
    source_paths = plan["source_paths"]
    video_names = plan["source_videos"]

    input_index_by_video = {vname: i for i, vname in enumerate(video_names)}
    fps_by_video = {vname: get_video_fps(source_paths[vname]) for vname in video_names}

    print(f"\n=== Rendering project: {project_name} ===")
    print(f"{len(clips)} clips from {len(video_names)} source videos -> "
          f"{resolution['width']}x{resolution['height']}, "
          f"{len(plan['captions'])} caption words")

    filter_parts = []
    filter_parts += build_trim_filters(clips, input_index_by_video)
    filter_parts += build_scale_filters(clips, resolution)
    filter_parts += build_punchzoom_filters(clips, resolution, fps_by_video)

    concat_parts, v_label, a_label, total_duration = build_concat_chain(clips, transitions)
    filter_parts += concat_parts

    fade_parts, v_label, a_label = build_fade_filters(
        v_label, a_label, total_duration,
        plan.get("fade_in_at_start", False), plan.get("fade_out_at_end", False)
    )
    filter_parts += fade_parts

    if plan["captions"]:
        ass_path = build_ass_subtitle_file(project_name, plan["captions"], resolution)
        ass_filename = escape_path_for_filter(str(ass_path))
        filter_parts.append(f"[{v_label}]subtitles='{ass_filename}'[vcap];")
        v_label = "vcap"

    filter_complex = " ".join(filter_parts)

    output_dir = Path("rendered_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{project_name}_final.mp4"

    cmd = ["ffmpeg", "-y"]
    for vname in video_names:
        cmd += ["-i", str(source_paths[vname])]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{v_label}]",
        "-map", f"[{a_label}]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
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
            and "embedded font" not in l and "memory font" not in l
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
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    render_project(args.project)