import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stimbench.synth import audit  # noqa: E402
from stimbench.synth import vocab as V  # noqa: E402
from stimbench.synth.generate import resolve, run, setup_logging, finish, previous_run_matches  # noqa: E402
from stimbench.synth.manifest import read_records, write_plan_csv  # noqa: E402
from stimbench.synth.motion import measure_set, summarise  # noqa: E402
from stimbench.synth.sampler import make_plan  # noqa: E402


def load_config(path):
    with open(path) as f:
        return resolve(yaml.safe_load(f))


def build_plan(cfg, args):
    s = cfg["sampling"]
    m = cfg["model"]
    return make_plan(
        classes=args.classes or s.get("classes", list(V.CLASSES)),
        n_per_class=args.n_per_class or s.get("n_per_class", 130),
        seed=cfg.get("experiment", {}).get("seed", 0),
        slow_factor=s.get("slow_factor", 2.0),
        min_cycles=s.get("min_cycles", 2),
        duration=m["frames"] / m["fps"],
    )


def cmd_plan(cfg, args, root):
    plan = build_plan(cfg, args)
    records = [c.record() for c in plan.clips]
    root.mkdir(parents=True, exist_ok=True)
    write_plan_csv(root / "plan.csv", records)
    comp = audit.composition(records)
    problems = audit.check(records)
    tokens = None
    if args.check_tokens:
        tokens, note = audit.token_lengths([r["prompt"] for r in records], cfg["model"]["repo"])
        if tokens is None:
            print(f"  token check skipped, no tokenizer could be loaded: {note}")
        else:
            print(f"  tokenizer: {note}")
            for r, t in zip(records, tokens):
                r["tokens"] = t
            problems = audit.check(records)
    md = audit.render_markdown(comp, problems, f"StimBench-Syn plan ({len(plan)} clips)", tokens)
    (root / "plan_composition.md").write_text(md)

    print(f"{'='*60}\nStimBench-Syn plan\n{'='*60}")
    print(f"  Config:   {args.config}")
    print(f"  Clips:    {len(plan)} ({plan.settings['n_per_class']} per class)")
    print(f"  Seed:     {plan.settings['seed']}   slow factor {plan.settings['slow_factor']}"
          f" (requested {plan.settings['slow_factor_requested']})")
    print(f"  Prompt words: {comp['prompt_words']}")
    if tokens:
        print(f"  Prompt tokens: {min(tokens)}-{max(tokens)}, {sum(t > 512 for t in tokens)} over 512")
    print(f"  Checks:   {'all passed' if not problems else ''}")
    for name, n in problems:
        print(f"    FAIL {name}: {n}")
    for f in ("gender", "severity", "setting", "aspect", "camera_motion"):
        print(f"  {f}:")
        for c, cnt in comp[f].items():
            print(f"    {c:<12} {dict(sorted(cnt.items()))}")
    for c in plan.clips[:args.show]:
        print(f"\n--- {c.cls} #{c.index} {c.gender} {c.severity} {c.environment_id} "
              f"{c.camera_id} ---\n{c.prompt}")
    print(f"\n  plan.csv and plan_composition.md written to {root}")
    return 1 if problems else 0


def cmd_generate(cfg, args, root):
    plan = build_plan(cfg, args)
    problems = audit.check([c.record() for c in plan.clips])
    log = setup_logging(root / "gen.log")
    log.info("=" * 62)
    log.info("model      %s (%s)", cfg["model"]["key"], cfg["model"]["repo"])
    log.info("size       %s  frames=%d  fps=%d  steps=%d  guidance=%s/%s  shift=%s",
             cfg["model"]["size"], cfg["model"]["frames"], cfg["model"]["fps"],
             cfg["model"]["steps"], cfg["model"]["guidance"], cfg["model"].get("guidance_2"),
             cfg["model"].get("flow_shift"))
    log.info("speed      %s", cfg["speed"])
    log.info("output     %s", root)
    if problems and not args.force:
        for name, n in problems:
            log.error("plan check FAILED: %s (%d)", name, n)
        log.error("refusing to generate from a plan that fails its checks; use --force to override")
        return 2
    if not previous_run_matches(root, cfg, plan) and not args.force:
        log.error("%s was started with a different model, speed or sampling config; "
                  "use a new --out or --force (stale clips are then regenerated)", root)
        return 2
    run(cfg, plan, root, log)
    return 0


def cmd_report(cfg, args, root):
    records = read_records(root / "manifest.jsonl")
    if not records:
        print(f"no manifest under {root}")
        return 1
    finish(root, setup_logging(root / "gen.log"))
    return 0


def cmd_motion(cfg, args, root):
    records = read_records(root / "manifest.jsonl")
    if not records:
        print(f"no manifest under {root}")
        return 1
    rows = measure_set(root, sorted(records, key=lambda r: r["file"]), root / "motion.csv")
    print(summarise(rows))
    print(f"\n  motion.csv written to {root} ({len(rows)} clips)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="StimBench-Syn generator. plan: render prompts, run the cue audit, write "
                    "plan.csv and composition.md (no GPU). generate: resumable generation; "
                    "rerun the same command to continue. report: rebuild metadata.csv and "
                    "composition.md from an existing manifest. motion: measure motion energy, freeze "
                    "fraction and achieved period per clip into motion.csv.")
    ap.add_argument("command", choices=["plan", "generate", "report", "motion"])
    ap.add_argument("--config", required=True, help="YAML config, see configs/synth/")
    ap.add_argument("--out", default=None, help="override output.root from the config")
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--classes", nargs="+", choices=list(V.CLASSES), default=None)
    ap.add_argument("--show", type=int, default=2, help="plan: prompts to print")
    ap.add_argument("--check-tokens", action="store_true",
                    help="plan: count prompt tokens with the generator's tokenizer")
    ap.add_argument("--force", action="store_true", help="generate even if checks fail")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.out or cfg["output"].get("root", "synth_out"))
    return {"plan": cmd_plan, "generate": cmd_generate, "report": cmd_report,
            "motion": cmd_motion}[args.command](cfg, args, root)


if __name__ == "__main__":
    sys.exit(main())
