import hashlib
import math
import random
from dataclasses import dataclass, asdict, field, replace
from typing import Dict, List, Optional

from . import vocab as V


def largest_remainder(weights: Dict[str, float], n: int) -> Dict[str, int]:
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    short = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:short]:
        counts[k] += 1
    return counts


def dealt(items: List, n: int, rng: random.Random) -> List:
    out = []
    while len(out) < n:
        deck = list(items)
        rng.shuffle(deck)
        out.extend(deck)
    return out[:n]


def clip_seed(seed: int, cls: str, index: int) -> int:
    # hash() is salted per process; sha256 is not
    h = hashlib.sha256(f"{seed}|{cls}|{index}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


def uniform_slow_factor(classes, requested: float, duration: float,
                        min_cycles: int) -> float:
    # the tightest class bound is applied to every class, so retiming is uniform
    if requested <= 1.0:
        return 1.0
    bound = requested
    for cls in classes:
        hz = V.TARGET_HZ.get(cls)
        if hz:
            bound = min(bound, hz * duration / min_cycles)
    return max(1.0, bound)


@dataclass
class ClipSpec:
    cls: str
    index: int
    gender: str
    severity: str
    topography_id: str
    topography: str
    posture: str
    goal_directed: bool
    environment_id: str
    environment: str
    setting: str
    setting_family: str
    camera_id: str
    camera: str
    camera_motion: str
    aspect: str
    age: str
    build: str
    hair: str
    skin: str
    clothing: str
    detail: str
    clutter: str
    light: str
    extra: str
    people_visible: bool
    pose: str
    shot: str
    secondary: str
    aesthetic_id: int
    aesthetic: str
    severity_text: str
    trigger: str
    direction: str
    pace: str
    requested_hz: object
    slow_factor: float
    prompt: str = ""

    @property
    def label(self):
        return V.LABELS[self.cls]

    @property
    def gender_label(self):
        return V.GENDER_LABEL[self.gender]

    def record(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        d["gender_label"] = self.gender_label
        return d


@dataclass
class Plan:
    clips: List[ClipSpec]
    settings: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.clips)

    def by_class(self):
        out = {}
        for c in self.clips:
            out.setdefault(c.cls, []).append(c)
        return out


def pace_clause(cls: str, slow: float, duration: float, min_cycles: int) -> str:
    # only a count is stated for flapping, which tracks it; head banging ignores
    # counts, so it gets the beat words; every class ends with the same continuity
    tail = "the motion continuing steadily from the first frame to the last"
    if cls == "Normal":
        return f"{{normal_pace}}, {tail}"
    if cls == "HeadBanging":
        return f"at a fast steady beat, several knocks per second, {tail}"
    hz = V.TARGET_HZ[cls] * V.COUNT_SCALE.get(cls, 1.0) / slow
    reps = min(12, max(min_cycles, int(hz * duration)))
    unit = "full turn" if cls == "Spinning" else "flap"
    return f"at a steady easy beat, about {V.WORD[reps]} {unit}s across the clip, {tail}"


def _years(age_text: str) -> int:
    return int("".join(ch for ch in age_text if ch.isdigit()))


def _short_hair_fix(sev_text: str, secondary: str, hair: str):
    if not any(h in hair for h in V.SHORT_HAIR):
        return sev_text, secondary
    sev_text = (sev_text
                .replace("hair and loose clothing carried outward",
                         "loose clothing carried outward")
                .replace(", hair carried along with it", ""))
    secondary = (secondary
                 .replace("brushing hair back from the face mid turn",
                          "wiping a hand across the face mid turn")
                 .replace("brushing hair back from the face", "rubbing at one eye")
                 .replace("letting the hair and clothing trail outward with the "
                          "turn", "letting loose clothing trail outward with the "
                          "turn"))
    return sev_text, secondary


def render_prompt(c: ClipSpec) -> str:
    qualifier = "" if c.cls == "Normal" else ", " + V.STEREOTYPY_QUALIFIER[c.cls]
    # the behaviour comes before the room: a long scene description ahead of
    # it dilutes the motion tokens the text encoder attends to
    return (
        f"{c.aesthetic}. {c.age} {c.gender}, {c.build}, with {c.hair} and "
        f"{c.skin}, wearing {c.clothing}{c.detail}. "
        f"The child is {c.topography.replace('{dir}', c.direction)}{c.trigger}, "
        f"{c.severity_text}, {c.pace}{qualifier}. "
        f"At the same time the child is {c.secondary}, {c.pose}, the body "
        f"loose and natural. "
        f"The scene is {c.environment}, {c.clutter}{c.extra}. "
        f"{c.shot}, {c.camera}, {c.light}. "
        f"One continuous unbroken shot, the room and furniture staying in place."
    )


def _compatible(cls: str, env: V.Environment) -> List[V.Topography]:
    return [t for t in V.TOPOGRAPHIES[cls] if t.needs <= env.tags]


def topography_quota(cls: str, n: int) -> dict:
    tops = V.TOPOGRAPHIES[cls]
    fixed = V.TOPOGRAPHY_WEIGHTS.get(cls, {})
    rest = [t.id for t in tops if t.id not in fixed]
    remainder = max(0.0, 1.0 - sum(fixed.values()))
    quota = {tid: n * w for tid, w in fixed.items()}
    quota.update({tid: n * remainder / max(len(rest), 1) for tid in rest})
    return quota


def make_plan(classes=V.CLASSES, n_per_class: int = 130, seed: int = 0,
              slow_factor: float = 2.0, min_cycles: int = 2,
              duration: float = 81 / 16, ab: Optional[dict] = None) -> Plan:
    rng = random.Random(seed)
    slow = uniform_slow_factor(classes, slow_factor, duration, min_cycles)
    clips: List[ClipSpec] = []
    if ab:
        return ab_plan(ab, rng, slow, duration, min_cycles, seed, slow_factor)

    for cls in classes:
        n = n_per_class
        genders = [V.GENDER[i % 2] for i in range(n)]

        if n < 2 * len(V.SEVERITY_LEVELS):
            # too few clips for a per-gender mix; walk the levels so a smoke set
            # still spans the axis instead of collapsing onto the first level
            deck = dealt(list(V.SEVERITY_LEVELS), n, rng)
            sev_iter = {g: iter(deck[i % 2::2]) for i, g in enumerate(V.GENDER)}
        else:
            sev_iter = {}
            weights = V.SEVERITY_WEIGHTS_BY_CLASS.get(cls, V.SEVERITY_WEIGHTS)
            for g in V.GENDER:
                counts = largest_remainder(weights, genders.count(g))
                deck = [lvl for lvl in V.SEVERITY_LEVELS for _ in range(counts[lvl])]
                rng.shuffle(deck)
                sev_iter[g] = iter(deck)

        envs = dealt(V.ENVIRONMENTS, n, rng)
        cams = dealt(V.CAMERA, n, rng)
        shots = dealt(V.SHOT, n, rng)
        aesthetics = dealt(range(len(V.AESTHETIC)), n, rng)
        builds = dealt(V.BUILD, n, rng)
        skins = dealt(V.SKIN, n, rng)
        topo_quota = topography_quota(cls, n)
        pace = pace_clause(cls, slow, duration, min_cycles)

        for i in range(n):
            g = genders[i]
            severity = next(sev_iter[g])
            env = envs[i]
            # rooms are dealt exactly; each room takes the compatible variant with
            # the most quota left, so variants are as even as the rooms allow
            options = _compatible(cls, env)
            topo = max(options, key=lambda t: (topo_quota[t.id], rng.random()))
            topo_quota[topo.id] -= 1

            outdoor = "outdoor" in env.tags
            io = "outdoor" if outdoor else "indoor"
            clutter_key = ("outdoor" if outdoor
                           else "institutional" if "institutional" in env.tags
                           else "domestic")

            age = rng.choice(V.AGE)
            years = _years(age)
            build = builds[i]
            hair = rng.choice(V.HAIR[g])
            skin = skins[i]

            def clothing_ok(tag):
                if tag == "school":
                    return "school" in env.tags and years >= 5
                if tag == "home":
                    return "home" in env.tags
                return True

            clothing = rng.choice([c for c, t in V.CLOTHING[g] if clothing_ok(t)])
            legwear = not any(w in clothing for w in ("dress", "skirt", "pinafore"))
            detail = rng.choice([d for d, t in V.DETAIL
                                 if t in ("any", io) or (t == "trousers" and legwear)])
            clutter = rng.choice(V.CLUTTER[clutter_key])
            light = rng.choice(V.LIGHT[io])
            extra = rng.choice(V.EXTRA[io])
            pose = rng.choice([p for p in V.POSE[topo.posture]
                               if p not in V.POSE_CLASS_BLOCK.get(cls, ())])
            shot = shots[i]
            cam = cams[i]

            sev_text = V.SEVERITY_BY_TOPOGRAPHY.get(topo.id, V.SEVERITY[(cls, topo.posture)])[severity]
            blocked = V.SECONDARY_POSTURE_BLOCK[topo.posture] | V.SECONDARY_TOPOGRAPHY_BLOCK.get(topo.id, set())
            secondary = rng.choice([s for s in V.SECONDARY[cls] if s not in blocked])
            sev_text, secondary = _short_hair_fix(sev_text, secondary, hair)
            aesthetic_id = aesthetics[i]
            trigger = rng.choice(V.AF_TRIGGER) if cls == "ArmFlapping" else ""
            direction = rng.choice(V.DIRECTION) if cls == "Spinning" else ""
            clip_pace = pace.replace("{normal_pace}", rng.choice(V.NORMAL_PACE)) if cls == "Normal" else pace

            spec = ClipSpec(
                cls=cls, index=i, gender=g, severity=severity,
                topography_id=topo.id, topography=topo.text,
                posture=topo.posture, goal_directed=topo.goal_directed,
                environment_id=env.id, environment=env.text, setting=io,
                setting_family=env.family,
                camera_id=cam.id, camera=cam.text, camera_motion=cam.motion,
                aspect=cam.aspect, age=age, build=build, hair=hair, skin=skin,
                clothing=clothing, detail=detail, clutter=clutter, light=light,
                extra=extra,
                people_visible=any(m in extra for m in V.PEOPLE_MARKERS),
                pose=pose, shot=shot, secondary=secondary,
                aesthetic_id=aesthetic_id, aesthetic=V.AESTHETIC[aesthetic_id],
                severity_text=sev_text, trigger=trigger, direction=direction, pace=clip_pace,
                requested_hz=V.TARGET_HZ.get(cls, ""), slow_factor=round(slow, 4),
            )
            spec.prompt = render_prompt(spec)
            clips.append(spec)

    rng.shuffle(clips)
    return Plan(clips, settings={
        "classes": list(classes), "n_per_class": n_per_class, "seed": seed,
        "slow_factor_requested": slow_factor, "slow_factor": round(slow, 4),
        "min_cycles": min_cycles, "duration_s": round(duration, 4),
        "negative_prompt": V.NEGATIVE,
    })


def ab_plan(ab: dict, rng, slow, duration, min_cycles, seed, requested) -> Plan:
    # one child, room, camera and severity drawn once; only the movement sentence
    # differs between clips, so a viewer compares wording and nothing else
    # base scene: clip `base_index` of an n=`base_n` plan, so a clip the user liked
    # in a smoke set can be reproduced exactly and varied one clause at a time
    cls = ab["cls"]
    if ab.get("base_manifest"):
        base = clipspec_from_record(_find_record(ab["base_manifest"], ab["base_file"]))
    else:
        base = make_plan([cls], int(ab.get("base_n", 1)), seed, requested, min_cycles,
                         duration).clips[int(ab.get("base_index", 0))]
    clips = []
    for i, m in enumerate(ab["movements"]):
        posture = m.get("posture", base.posture)
        severity = m.get("severity", ab.get("severity", base.severity))
        sev_text = m.get("severity_text") or V.SEVERITY_BY_TOPOGRAPHY.get(
            m["id"], V.SEVERITY[(cls, posture)])[severity]
        c = replace(base, index=i, topography_id=m["id"], topography=m.get("text", base.topography),
                    posture=posture, severity=severity, severity_text=sev_text,
                    trigger=m.get("trigger", ab.get("trigger", base.trigger)),
                    pace=m.get("pace", base.pace), secondary=m.get("secondary", base.secondary),
                    pose=m.get("pose", base.pose))
        c.prompt = render_prompt(c)
        clips.append(c)
    return Plan(clips, settings={
        "classes": [cls], "n_per_class": len(clips), "seed": seed,
        "slow_factor_requested": requested, "slow_factor": round(slow, 4),
        "min_cycles": min_cycles, "duration_s": round(duration, 4),
        "negative_prompt": V.NEGATIVE, "ab": ab,
    })


def _find_record(manifest_path, file):
    import json
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if rec.get("file") == file:
                    return rec
    raise KeyError(f"{file} not in {manifest_path}")


def clipspec_from_record(rec: dict) -> ClipSpec:
    # a generated clip's own slots, so a scene a viewer liked can be reused exactly
    import dataclasses
    names = [f.name for f in dataclasses.fields(ClipSpec)]
    defaults = {"trigger": "", "direction": "", "requested_hz": "", "prompt": ""}
    return ClipSpec(**{n: rec.get(n, defaults.get(n)) for n in names})
