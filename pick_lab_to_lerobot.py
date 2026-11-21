"""Convert Pick Lab episodes (artifact db export + wrist-camera WebM assets) into a LeRobot dataset.

Input layout (what Claude downloads with ArtifactData out_dir + Artifact asset reads):
    raw/episodes/<id>.json   one episode per file ({"data": {...}} wrapper or the bare doc), or a .jsonl export
    raw/videos/<asset>.webm  wrist-camera video named by the episode's cam.asset id

Usage:
    .venv/bin/python pick_lab_to_lerobot.py --raw raw --out dataset --repo-id local/pick-lab [--only-success] [--push]

Episodes without a wrist video (v2 and earlier, or a failed upload) are skipped: every frame needs an image.
"""
import argparse
import glob
import json
import os
import shutil

import av
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # lerobot < 0.3
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FPS = 10
CAM = 128
MAX_ROWS = 2000  # stages cap at 75 s * 10 Hz; anything longer is not from the game
JOINTS = ["j1_base", "j2_shoulder", "j3_elbow", "j4_wrist_pitch", "j5_wrist_roll", "gripper"]
TASKS = {
    "lift": "Pick up the red brick and lift it at least 15 cm.",
    "place": "Pick up the red brick and place it inside the green circle.",
    "precise": "Place the red brick precisely inside the small green circle.",
    "wall": "Move the red brick over the barricade into the green circle.",
    "stack": "Stack the red brick and the blue block on top of each other.",
}
POSE = ["x", "y", "z", "qw", "qx", "qy", "qz"]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    "observation.images.wrist": {"dtype": "video", "shape": (CAM, CAM, 3), "names": ["height", "width", "channels"]},
    "observation.ee_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    # cube (7) + cube2 (7, zeros if unused) + wall (x, y, yaw; zeros if none) in the MuJoCo world frame
    "observation.environment_state": {"dtype": "float32", "shape": (17,),
        "names": [f"cube_{p}" for p in POSE] + [f"cube2_{p}" for p in POSE] + ["wall_x", "wall_y", "wall_yaw"]},
    # goal zone center, tolerance, and whether the stage has a zone
    "observation.goal": {"dtype": "float32", "shape": (4,), "names": ["zone_x", "zone_y", "zone_tol", "has_zone"]},
    # raw human keyboard input (gripper-mode: vx, vy, vz, vroll, 0, grip; joint-mode: dq1..dq5, grip) and mode (0 gripper, 1 joint)
    "teleop.keys": {"dtype": "float32", "shape": (6,), "names": [f"key{i}" for i in range(6)]},
    "teleop.mode": {"dtype": "float32", "shape": (1,), "names": ["mode"]},
    "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    "next.done": {"dtype": "bool", "shape": (1,), "names": None},
}


def load_episodes(raw):
    docs = []
    for path in sorted(glob.glob(os.path.join(raw, "episodes", "*.json"))):
        with open(path) as f:
            docs.append(json.load(f))
    for path in sorted(glob.glob(os.path.join(raw, "*.jsonl"))):
        with open(path) as f:
            docs += [json.loads(line) for line in f if line.strip()]
    return [d.get("data", d) for d in docs]


def columns(ep):
    """Validate an untrusted episode doc and return float columns, or raise ValueError."""
    cols, fields = ep.get("cols"), ep.get("fields")
    if not isinstance(cols, dict) or not isinstance(fields, list) or not fields:
        raise ValueError("missing cols/fields")
    n = len(cols.get("t", []))
    if not 1 < n <= MAX_ROWS:
        raise ValueError(f"bad length {n}")
    scale = float(ep.get("scale", 1))
    out = {}
    for name in fields:
        arr = np.asarray(cols.get(name), dtype=np.float64)
        if arr.shape != (n,) or not np.isfinite(arr).all():
            raise ValueError(f"bad column {name}")
        out[name] = (arr / scale).astype(np.float32)
    return out, n


def wrist_frames(path, n):
    """Decode the WebM and pick one frame per row (the browser recorder can be off by a frame or two)."""
    with av.open(path) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    if not frames:
        raise ValueError("video has no frames")
    m = len(frames)
    idx = np.round(np.linspace(0, m - 1, n)).astype(int)
    return [frames[i] for i in idx], m


def stack(cols, names, n):
    return np.stack([cols[k] if k in cols else np.zeros(n, np.float32) for k in names], axis=1)


def convert(args):
    if os.path.exists(args.out):
        shutil.rmtree(args.out)  # the output is always rebuilt from raw/
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=FEATURES, root=args.out,
                               robot_type="pick-lab-5dof", use_videos=True)
    kept, skipped = [], []
    for ep in sorted(load_episodes(args.raw), key=lambda e: e.get("created", 0)):
        eid = ep.get("id", "?")
        task = "lift" if ep.get("task") == "pick-cube" else ep.get("task")
        cam = ep.get("cam") or {}
        video = os.path.join(args.raw, "videos", f"{cam.get('asset', '')}.webm")
        if task not in TASKS:
            skipped.append((eid, f"unknown task {task!r}")); continue
        if args.only_success and not ep.get("success"):
            skipped.append((eid, "failed episode")); continue
        if not cam.get("asset") or not os.path.exists(video):
            skipped.append((eid, "no wrist video")); continue
        try:
            cols, n = columns(ep)
            frames, m = wrist_frames(video, n)
        except (ValueError, av.error.FFmpegError) as e:
            skipped.append((eid, str(e))); continue

        state = stack(cols, ["q1", "q2", "q3", "q4", "q5", "grip"], n)
        action = stack(cols, ["ctrl1", "ctrl2", "ctrl3", "ctrl4", "ctrl5", "ctrl_grip"], n)
        ee = stack(cols, ["tcp_x", "tcp_y", "tcp_z"], n)
        env = stack(cols, [f"cube_{p}" for p in ["x", "y", "z", "qw", "qx", "qy", "qz"]]
                    + [f"cube2_{p}" for p in ["x", "y", "z", "qw", "qx", "qy", "qz"]], n)
        goal = ep.get("goal") or {}
        wall = goal.get("wall") or [0, 0, 0]
        env = np.concatenate([env, np.tile(np.asarray(wall, np.float32), (n, 1))], axis=1)
        zone = goal.get("zone")
        goal_vec = np.asarray([*(zone or [0, 0]), goal.get("tol") or 0, 1 if zone else 0], np.float32)
        key_names = [f"key{i}" for i in range(6)] if "key0" in cols else [f"a{i}" for i in range(6)]
        keys = stack(cols, key_names, n)
        mode = stack(cols, ["mode"], n)
        success = bool(ep.get("success"))

        for i in range(n):
            last = i == n - 1
            ds.add_frame({
                "observation.state": state[i], "action": action[i],
                "observation.images.wrist": frames[i],
                "observation.ee_pos": ee[i], "observation.environment_state": env[i], "observation.goal": goal_vec,
                "teleop.keys": keys[i], "teleop.mode": mode[i],
                "next.success": np.array([last and success]), "next.done": np.array([last]),
                "task": TASKS[task],
            })
        ds.save_episode()
        kept.append((eid, task, n, m, success))
    if hasattr(ds, "finalize"):
        ds.finalize()
    return ds, kept, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="raw")
    p.add_argument("--out", default="dataset")
    p.add_argument("--repo-id", default="local/pick-lab")
    p.add_argument("--only-success", action="store_true", help="keep successful episodes only (imitation learning)")
    p.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub as --repo-id")
    args = p.parse_args()

    _, kept, skipped = convert(args)
    for eid, task, n, m, ok in kept:
        print(f"kept    {eid}  {task:<8} {n:>4} frames (video had {m})  {'success' if ok else 'fail'}")
    for eid, why in skipped:
        print(f"skipped {eid}  {why}")
    if not kept:
        raise SystemExit("no episodes converted")

    # self-check: reload from disk and read one frame back
    ds = LeRobotDataset(args.repo_id, root=args.out, video_backend="pyav")  # torchcodec needs a system FFmpeg
    assert ds.num_episodes == len(kept), (ds.num_episodes, len(kept))
    assert ds.num_frames == sum(k[2] for k in kept), (ds.num_frames, sum(k[2] for k in kept))
    item = ds[ds.num_frames - 1]
    assert tuple(item["observation.images.wrist"].shape) == (3, CAM, CAM), item["observation.images.wrist"].shape
    assert tuple(item["action"].shape) == (6,)
    print(f"OK  {ds.num_episodes} episodes, {ds.num_frames} frames -> {os.path.abspath(args.out)}")
    if args.push:
        ds.push_to_hub()


if __name__ == "__main__":
    main()
