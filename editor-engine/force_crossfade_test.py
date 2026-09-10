r"""
One-off test utility: forces the first transition in an existing edit
plan to be a crossfade, so we can directly test render.py's xfade/
acrossfade code path without waiting for Gemini to happen to choose one.

Makes a backup of the original plan first so it's easy to restore.

Usage:
    python force_crossfade_test.py <video_name>

Example:
    python force_crossfade_test.py clip4
"""

import sys
import json
import shutil
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python force_crossfade_test.py <video_name>")
    sys.exit(1)

video_name = sys.argv[1]
plan_path = Path("analysis_output") / f"{video_name}_edit_plan.json"
backup_path = Path("analysis_output") / f"{video_name}_edit_plan_BACKUP.json"

if not plan_path.exists():
    print(f"Error: {plan_path} not found.")
    sys.exit(1)

if not backup_path.exists():
    shutil.copy(plan_path, backup_path)
    print(f"Backed up original plan to: {backup_path}")
else:
    print(f"Backup already exists at {backup_path} (not overwriting).")

with open(plan_path, "r", encoding="utf-8") as f:
    plan = json.load(f)

if len(plan["clips"]) < 2:
    print("Error: this video only has 1 clip - no transition to force. "
          "Try this on a video with multiple selected moments.")
    sys.exit(1)

if not plan["transitions"]:
    plan["transitions"] = [{"between_clip_order": 0, "type": "cut", "duration": 0}]

plan["transitions"][0] = {
    "between_clip_order": 0,
    "type": "crossfade",
    "duration": 0.5
}

with open(plan_path, "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)

print(f"Forced a 0.5s crossfade between clip 0 and clip 1 in {plan_path}")
print(f"Now run: python render.py {video_name}.mp4")
print(f"To restore the original plan afterward: "
      f"copy {backup_path} over {plan_path}")