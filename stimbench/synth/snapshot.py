import json
from pathlib import Path

from . import vocab as V

"""A vocabulary snapshot lets a paired screening render one condition with the wording of
an earlier commit while the scene, seed and everything else come from the current plan."""


def dump(path: Path):
    data = {
        "topography": {t.id: t.text for ts in V.TOPOGRAPHIES.values() for t in ts},
        "severity": {f"{cls}|{posture}": texts for (cls, posture), texts in V.SEVERITY.items()},
        "severity_by_topography": V.SEVERITY_BY_TOPOGRAPHY,
        "secondary": V.SECONDARY,
        "pose": V.POSE,
        "pose_by_class": getattr(V, "POSE_BY_CLASS", {}),
        "negative": V.NEGATIVE,
        "negative_by_class": V.NEGATIVE_BY_CLASS,
        "negative_drop_by_class": {k: list(v) for k, v in getattr(V, "NEGATIVE_DROP_BY_CLASS", {}).items()},
    }
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def severity_text(snap: dict, cls: str, topography_id: str, posture: str, severity: str):
    by_topo = snap["severity_by_topography"].get(topography_id)
    table = by_topo or snap["severity"].get(f"{cls}|{posture}", {})
    return table.get(severity)


def negative(snap: dict, cls: str) -> str:
    text = snap["negative"]
    for phrase in snap["negative_drop_by_class"].get(cls, []):
        text = text.replace(phrase, "")
    return text + snap["negative_by_class"].get(cls, "")
