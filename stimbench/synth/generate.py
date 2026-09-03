import logging
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import audit
from . import vocab as V
from .manifest import (ManifestWriter, write_metadata_csv, write_run_config,
                       read_records)
from .sampler import Plan, clip_seed
from .video import have_ffmpeg, probe, retime_cfr

DEFAULT_MODEL = {
    "key": "wan2.2-14b-480p",
    "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "size": [480, 832],
    "frames": 81,
    "fps": 16,
    "steps": 40,
    "guidance": 4.0,
    "guidance_2": 3.0,
    "flow_shift": 3.0,
}

DEFAULT_SPEED = {
    "offload": False,
    "attention_backend": None,
    "compile": False,
    "compile_mode": "default",
    "cache": "none",
    "lightning_lora": None,
}


def resolve(cfg: dict) -> dict:
    out = dict(cfg)
    out["model"] = {**DEFAULT_MODEL, **cfg.get("model", {})}
    out["speed"] = {**DEFAULT_SPEED, **cfg.get("speed", {})}
    out.setdefault("sampling", {})
    out.setdefault("output", {})
    lora = out["speed"].get("lightning_lora")
    if lora:
        out["model"].update({k: lora[k] for k in ("steps", "guidance", "guidance_2",
                                                    "flow_shift") if k in lora})
    return out


def speed_mode(cfg: dict) -> str:
    m, s = cfg["model"], cfg["speed"]
    parts = [f"steps{m['steps']}", f"cache={s['cache']}", f"compile={int(bool(s['compile']))}"]
    if s.get("attention_backend"):
        parts.append(f"attn={s['attention_backend']}")
    if s.get("lightning_lora"):
        parts.append("lightning")
    return "|".join(parts)


def setup_logging(path: Path) -> logging.Logger:
    log = logging.getLogger("stimbench.synth")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def fmt_dur(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def load_pipeline(cfg: dict, log: logging.Logger):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler

    m, sp = cfg["model"], cfg["speed"]
    # Wan documents its VAE in fp32; the transformer stays bf16
    vae = AutoencoderKLWan.from_pretrained(m["repo"], subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(m["repo"], vae=vae, torch_dtype=torch.bfloat16)
    if m.get("flow_shift"):
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=m["flow_shift"])

    lora = sp.get("lightning_lora")
    if lora:
        log.info("loading distilled LoRA %s", lora["repo"])
        pipe.load_lora_weights(lora["repo"], weight_name=lora["high"],
                               adapter_name="lightning_high")
        if getattr(pipe, "transformer_2", None) is not None and lora.get("low"):
            pipe.load_lora_weights(lora["repo"], weight_name=lora["low"],
                                   adapter_name="lightning_low",
                                   load_into_transformer_2=True)

    if sp.get("offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    # the VAE decode, not the transformer, is the peak allocator
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    experts = [t for t in (getattr(pipe, "transformer", None),
                           getattr(pipe, "transformer_2", None)) if t is not None]

    if sp.get("attention_backend"):
        for t in experts:
            t.set_attention_backend(sp["attention_backend"])
        log.info("attention backend %s", sp["attention_backend"])

    cache = (sp.get("cache") or "none").lower()
    if cache == "pab":
        from diffusers import PyramidAttentionBroadcastConfig
        c = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=2,
            spatial_attention_timestep_skip_range=(100, 800),
            current_timestep_callback=lambda: pipe.current_timestep)
        for t in experts:
            t.enable_cache(c)
        log.info("cache: pyramid attention broadcast")
    elif cache == "fastercache":
        from diffusers import FasterCacheConfig
        c = FasterCacheConfig(
            spatial_attention_block_skip_range=2,
            spatial_attention_timestep_skip_range=(-1, 681),
            current_timestep_callback=lambda: pipe.current_timestep,
            attention_weight_callback=lambda _: 0.3,
            unconditional_batch_skip_range=5,
            unconditional_batch_timestep_skip_range=(-1, 641),
            tensor_format="BCFHW")
        for t in experts:
            t.enable_cache(c)
        log.info("cache: FasterCache")
    elif cache != "none":
        raise ValueError(f"unknown cache mode {cache!r}")

    if sp.get("compile"):
        for t in experts:
            t.compile(mode=sp.get("compile_mode", "default"), dynamic=False)
        log.info("torch.compile enabled (%s); first clip per aspect is slow",
                 sp.get("compile_mode", "default"))

    if not sys.stdout.isatty() and hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def clip_relpath(cfg: dict, cls: str, index: int) -> str:
    return f"{cls}/{cfg['model']['key']}_{cls}_{index:04d}.mp4"


def frame_size(cfg: dict, aspect: str):
    h, w = cfg["model"]["size"]
    return (w, h) if aspect == "portrait" else (h, w)


def run(cfg: dict, plan: Plan, root: Path, log: logging.Logger) -> dict:
    import torch
    from diffusers.utils import export_to_video

    m = cfg["model"]
    root = Path(root)
    writer = ManifestWriter(root)
    write_run_config(root, cfg, plan.settings)
    todo = [c for c in plan.clips if clip_relpath(cfg, c.cls, c.index) not in writer.done]
    log.info("plan       %d clips, %d already done, %d to generate",
             len(plan), len(plan) - len(todo), len(todo))
    if not todo:
        writer.close()
        finish(root, log)
        return {"generated": 0, "failed": 0}
    if not have_ffmpeg():
        log.warning("ffmpeg/ffprobe not found: clips will stay at 1/%.2f tempo",
                    plan.settings["slow_factor"])

    t0 = time.time()
    pipe = load_pipeline(cfg, log)
    log.info("pipeline loaded in %s", fmt_dur(time.time() - t0))
    mode = speed_mode(cfg)
    slow = plan.settings["slow_factor"]

    times, failed, retime_failed = [], 0, 0
    t_start = time.time()
    for n, spec in enumerate(todo, 1):
        rel = clip_relpath(cfg, spec.cls, spec.index)
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        seed = clip_seed(plan.settings["seed"], spec.cls, spec.index)
        h, w = frame_size(cfg, spec.aspect)
        t_clip = time.time()
        try:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            kwargs = {}
            if m.get("guidance_2") is not None and getattr(pipe, "transformer_2", None) is not None:
                kwargs["guidance_scale_2"] = m["guidance_2"]
            frames = pipe(prompt=spec.prompt, negative_prompt=V.NEGATIVE,
                          height=h, width=w, num_frames=m["frames"],
                          num_inference_steps=m["steps"], guidance_scale=m["guidance"],
                          generator=gen, **kwargs).frames[0]
            export_to_video(frames, str(dst), fps=m["fps"])
        except KeyboardInterrupt:
            log.warning("interrupted after %d clip(s); rerun the same command to resume", n - 1)
            break
        except Exception as exc:
            failed += 1
            log.error("[%d/%d] FAILED %s: %s: %s", n, len(todo), rel, type(exc).__name__, exc)
            log.debug(traceback.format_exc())
            if dst.exists():
                dst.unlink()
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            continue

        retimed, err = False, ""
        if slow > 1.0 and have_ffmpeg():
            retimed, _, err = retime_cfr(dst, slow, m["fps"])
            if not retimed:
                retime_failed += 1
                log.error("retime failed for %s: %s", rel, err)
        info = probe(dst) or {}
        dt = time.time() - t_clip
        rec = {
            **spec.record(),
            "file": rel, "clip_id": Path(rel).stem, "seed": seed,
            "model": m["key"], "repo": m["repo"], "width": w, "height": h,
            "gen_frames": m["frames"], "gen_fps": m["fps"],
            "gen_duration_s": round(m["frames"] / m["fps"], 4),
            "steps": m["steps"], "guidance": m["guidance"],
            "guidance_2": m.get("guidance_2", ""), "flow_shift": m.get("flow_shift", ""),
            "speed_mode": mode, "retimed": retimed, "gen_seconds": round(dt, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "out_frames": info.get("out_frames", ""), "out_fps": info.get("out_fps", ""),
            "out_duration_s": info.get("out_duration_s", ""),
        }
        writer.add(rec)
        times.append(dt)
        eta = statistics.median(times) * (len(todo) - n)
        log.info("[%d/%d] %-46s %-9s %-10s %5.1fs  eta %s", n, len(todo), rel,
                 spec.aspect, spec.severity, dt, fmt_dur(eta))

    writer.close()
    log.info("finished   %d generated, %d failed, in %s", len(times), failed,
             fmt_dur(time.time() - t_start))
    if times:
        log.info("per clip   median %.1fs  min %.1fs  max %.1fs  (%s)",
                 statistics.median(times), min(times), max(times), mode)
    if retime_failed:
        log.warning("%d clip(s) failed to retime and are at 1/%.2f tempo", retime_failed, slow)
    if failed:
        log.warning("%d clip(s) failed; rerun the same command to fill the gaps", failed)
    finish(root, log)
    return {"generated": len(times), "failed": failed}


def finish(root: Path, log: logging.Logger):
    records = read_records(Path(root) / "manifest.jsonl")
    if not records:
        return
    n = write_metadata_csv(root, records)
    comp = audit.composition(records)
    problems = audit.check(records)
    (Path(root) / "composition.md").write_text(
        audit.render_markdown(comp, problems, f"StimBench-Syn composition ({n} clips)"))
    log.info("metadata   %s (%d clips)", Path(root) / "metadata.csv", n)
    log.info("report     %s%s", Path(root) / "composition.md",
             "" if not problems else f"  ({len(problems)} check(s) FAILED)")
