from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Tuple

from . import vocab as V

FACTORS = ("gender", "severity", "setting", "setting_family", "aspect",
           "camera_motion", "environment_id", "topography_id", "posture",
           "age", "skin", "people_visible", "goal_directed", "aesthetic_id",
           "out_fps", "out_frames", "retimed", "speed_mode")

STOP_WORDS = ("stopping", "pausing", "then settling", "sitting down",
              "coming to rest", "standing still")


def composition(records: Iterable[dict]) -> Dict:
    records = list(records)
    classes = [c for c in V.CLASSES if any(r["cls"] == c for r in records)]
    out = OrderedDict()
    out["n"] = {c: sum(r["cls"] == c for r in records) for c in classes}
    for f in FACTORS:
        out[f] = {c: dict(Counter(str(r.get(f, "")) for r in records if r["cls"] == c))
                  for c in classes}
    words = [len(r["prompt"].split()) for r in records]
    out["prompt_words"] = {"min": min(words), "mean": round(sum(words) / len(words), 1),
                           "max": max(words)}
    out["slow_factor"] = sorted({float(r["slow_factor"]) for r in records})
    return out


MIN_CLIPS_FOR_BALANCE = 20   # a 1-per-class smoke test can only be 0% or 100% outdoor


def check(records: Iterable[dict], max_setting_spread: float = 0.05) -> List[Tuple[str, int]]:
    records = list(records)
    problems = []

    def add(name, hits):
        n = sum(hits)
        if n:
            problems.append((name, n))

    add("prompt without a pace instruction",
        ["slowly and deliberately" not in r["prompt"]
         and "natural pace" not in r["prompt"] for r in records])
    add("Normal behaviour that stops or pauses, against the continuity clause",
        [r["cls"] == "Normal" and any(w in r["topography"] for w in STOP_WORDS)
         for r in records])
    add("seated posture asked to lift heels",
        [r["posture"] == "seated" and "heels lift" in r["severity_text"]
         for r in records])
    add("seated posture given a standing severity text",
        [r["posture"] == "seated" and ("step" in r["severity_text"]
                                       or "heels" in r["severity_text"])
         for r in records])
    add("topography placed in an environment lacking what it needs",
        [_needs_unmet(r) for r in records])
    add("duplicate prompt", [n > 1 for n in Counter(r["prompt"] for r in records).values()])
    add("stereotypy qualifier on a Normal prompt",
        [r["cls"] == "Normal" and V.STEREOTYPY_QUALIFIER in r["prompt"] for r in records])
    add("stimming prompt without the stereotypy qualifier",
        [r["cls"] != "Normal" and V.STEREOTYPY_QUALIFIER not in r["prompt"]
         for r in records])

    for f in ("slow_factor", "out_fps", "out_frames", "out_duration_s", "retimed",
              "speed_mode", "steps", "plan_hash_config"):
        vals = {str(r[f]) for r in records if f in r}
        if len(vals) > 1:
            problems.append((f"{f} differs between clips", len(vals)))
    for aspect in ("portrait", "landscape"):
        sizes = {(r["width"], r["height"]) for r in records
                 if r["aspect"] == aspect and "width" in r}
        if len(sizes) > 1:
            problems.append((f"{aspect} clips have more than one frame size", len(sizes)))
    add("generated record missing probe data",
        ["out_frames" in r and r["out_frames"] == "" for r in records])
    add("prompt over the 512-token encoder limit",
        [r.get("tokens", 0) > 512 for r in records])

    by_cls = {}
    for r in records:
        by_cls.setdefault(r["cls"], []).append(r)
    if min(len(rs) for rs in by_cls.values()) < MIN_CLIPS_FOR_BALANCE:
        return problems
    genders = {c: Counter(r["gender"] for r in rs) for c, rs in by_cls.items()}
    for c, cnt in genders.items():
        if abs(cnt.get("boy", 0) - cnt.get("girl", 0)) > 1:
            problems.append((f"gender imbalance in {c}", abs(cnt["boy"] - cnt["girl"])))
    for c, rs in by_cls.items():
        share = Counter(r["topography_id"] for r in rs).most_common(1)[0]
        if share[1] > 0.5 * len(rs):
            problems.append((f"one topography holds over half of {c}", share[1]))
    outdoor = {c: sum(r["setting"] == "outdoor" for r in rs) / len(rs)
               for c, rs in by_cls.items()}
    if outdoor and max(outdoor.values()) - min(outdoor.values()) > max_setting_spread:
        problems.append(("outdoor share differs between classes by more than "
                         f"{max_setting_spread:.0%}", 1))
    return problems


def _needs_unmet(r: dict) -> bool:
    topo = next((t for t in V.TOPOGRAPHIES[r["cls"]] if t.id == r["topography_id"]), None)
    env = next((e for e in V.ENVIRONMENTS if e.id == r["environment_id"]), None)
    if topo is None or env is None:
        return True
    return not topo.needs <= env.tags


def token_lengths(prompts: List[str], repo: str, subfolder: str = "tokenizer"):
    # A diffusers repo has no root config.json, so AutoTokenizer needs the network;
    # offline, the tokenizer directory of a cached snapshot loads directly.
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        return None, f"transformers not installed ({e})"
    errors = []
    try:
        tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
        return [len(tok(p).input_ids) for p in prompts], f"{repo}/{subfolder}"
    except Exception as e:
        errors.append(str(e).splitlines()[0][:120])
    import glob, os
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME", "~/.cache/huggingface")
    pattern = os.path.expanduser(os.path.join(cache, "**", "models--" + repo.replace("/", "--"),
                                              "snapshots", "*", subfolder))
    for snap in sorted(glob.glob(pattern, recursive=True)):
        try:
            tok = AutoTokenizer.from_pretrained(snap)
            return [len(tok(p).input_ids) for p in prompts], snap
        except Exception as e:
            errors.append(str(e).splitlines()[0][:120])
    return None, "; ".join(errors) or "no cached snapshot found"


def render_markdown(comp: Dict, problems: List[Tuple[str, int]], title: str,
                    tokens=None, max_tokens: int = 512) -> str:
    classes = list(comp["n"])
    lines = [f"# {title}", ""]
    lines.append("Clips per class: " + ", ".join(f"{c} {comp['n'][c]}" for c in classes))
    lines.append(f"Slow factor: {comp['slow_factor']}   Prompt words: "
                 f"{comp['prompt_words']['min']}-{comp['prompt_words']['max']} "
                 f"(mean {comp['prompt_words']['mean']})")
    if tokens:
        lines.append(f"Prompt tokens: {min(tokens)}-{max(tokens)}, "
                     f"{sum(t > max_tokens for t in tokens)} over {max_tokens}")
    lines.append("")
    lines.append("## Checks")
    if problems:
        for name, n in problems:
            lines.append(f"- FAIL {name}: {n}")
    else:
        lines.append("- all checks passed")
    lines.append("")
    for f in FACTORS:
        keys = sorted({k for c in classes for k in comp[f][c]})
        if len(keys) > 40:
            continue
        lines.append(f"## {f}")
        lines.append("| value | " + " | ".join(classes) + " |")
        lines.append("|---|" + "---|" * len(classes))
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(str(comp[f][c].get(k, 0)) for c in classes) + " |")
        lines.append("")
    return "\n".join(lines)
