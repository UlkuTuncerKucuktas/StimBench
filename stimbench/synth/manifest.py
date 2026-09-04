import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

FIELDS = [
    "file", "clip_id", "cls", "label", "index", "seed",
    "severity", "topography_id", "posture", "goal_directed", "trigger", "pace",
    "requested_hz", "slow_factor",
    "gender", "gender_label", "age", "build", "hair", "skin", "clothing", "detail",
    "environment_id", "setting", "setting_family", "clutter",
    "light", "extra", "people_visible", "pose", "shot",
    "camera_id", "camera_motion", "aspect", "aesthetic_id",
    "model", "repo", "width", "height", "gen_frames", "gen_fps", "gen_duration_s",
    "steps", "guidance", "guidance_2", "flow_shift", "speed_mode", "plan_hash",
    "out_frames", "out_fps", "out_duration_s", "retimed", "gen_seconds",
    "generated_at",
    "topography", "severity_text", "secondary", "environment", "camera",
    "aesthetic", "prompt",
]

METADATA_FIELDS = ["file_name", "label", "split", "type", "source_dataset",
                   "group_id", "url", "youtube_id", "clip_start", "clip_end",
                   "clip_duration", "gender",
                   "severity", "setting", "topography_id", "camera_motion",
                   "aspect", "seed"]


def read_records(path: Path) -> List[dict]:
    # a kill -9 mid-append leaves one truncated line; the last record per file wins
    by_file = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_file[rec["file"]] = rec
    return list(by_file.values())


def _repair_truncated_tail(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


class ManifestWriter:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "manifest.jsonl"
        _repair_truncated_tail(self.jsonl_path)
        self.records = {r["file"]: r for r in read_records(self.jsonl_path)}
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")

    def add(self, rec: dict):
        self.records[rec["file"]] = rec
        self._jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def close(self):
        self._jsonl.close()


def write_manifest_csv(root: Path, records: Iterable[dict]):
    write_plan_csv(Path(root) / "manifest.csv", sorted(records, key=lambda r: r["file"]))


def write_plan_csv(path: Path, records: Iterable[dict]):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def write_metadata_csv(root: Path, records: Iterable[dict], split: str = "train"):
    root = Path(root)
    rows = []
    for r in records:
        rows.append({
            "file_name": r["file"], "label": r["label"], "split": split,
            "type": "synthetic", "source_dataset": "StimBench-Syn",
            "group_id": r["clip_id"], "url": "", "youtube_id": "",
            "clip_start": 0.0, "clip_end": r.get("out_duration_s", ""),
            "clip_duration": r.get("out_duration_s", ""),
            "gender": r["gender_label"], "severity": r["severity"],
            "setting": r["setting"], "topography_id": r["topography_id"],
            "camera_motion": r["camera_motion"], "aspect": r["aspect"],
            "seed": r["seed"],
        })
    rows.sort(key=lambda x: x["file_name"])
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METADATA_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_run_config(root: Path, cfg: dict, plan_settings: dict):
    (Path(root) / "run_config.json").write_text(
        json.dumps({"config": cfg, "plan": plan_settings}, indent=2, ensure_ascii=False))
