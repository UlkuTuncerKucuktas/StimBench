import hashlib
import json
import time
from pathlib import Path
from typing import List, Tuple

from . import vocab as V
from .generate import load_pipeline, fmt_dur
from .sampler import make_plan
from .video import probe, retime_cfr


def targets_from_config(cfg: dict, spec: dict) -> List[Tuple[str, dict]]:
    """(id, plan record) pairs whose prompts describe a new child and scene around the same movement."""
    if "from_plan" in spec:
        fp = spec["from_plan"]
        s, m = cfg["sampling"], cfg["model"]
        plan = make_plan(classes=[fp.get("cls", "ArmFlapping")], n_per_class=fp.get("pool", 40),
                         seed=fp.get("seed", 0), slow_factor=s.get("slow_factor", 2.0),
                         min_cycles=s.get("min_cycles", 2), duration=m["frames"] / m["fps"])
        clips = [c for c in plan.clips
                 if (not fp.get("topography") or c.topography_id == fp["topography"])
                 and (not fp.get("severity") or c.severity == fp["severity"])]
        return [(f"t{c.index:02d}", c.record()) for c in clips[:fp.get("n", 3)]]
    return [(t["id"], {"cls": t.get("cls", "ArmFlapping"), "prompt": t["prompt"]}) for t in spec]


def read_frames(path: Path, expected: int):
    import imageio.v2 as imageio
    from PIL import Image
    frames = [Image.fromarray(f[..., :3]) for f in imageio.mimread(str(path), memtest=False)]
    # imageio-ffmpeg resamples variable-frame-rate input; a wrong count would corrupt the latents
    if len(frames) != expected:
        raise ValueError(f"{path.name}: read {len(frames)} frames, expected {expected}")
    return frames


def encode_video(pipe, frames, height: int, width: int):
    import torch
    video = pipe.video_processor.preprocess_video(frames, height=height, width=width)
    video = video.to("cuda", torch.float32)
    with torch.no_grad():
        latents = pipe.vae.encode(video).latent_dist.mode()
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    return (latents - mean) * std


def decode_latents(pipe, latents):
    import torch
    latents = latents.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    with torch.no_grad():
        video = pipe.vae.decode(latents / std + mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]


def denoise_from(pipe, source_latents, prompt: str, negative: str, strength: float,
                 steps: int, guidance: float, guidance_2: float, seed: int):
    """SDEdit on the two-expert Wan pipeline: re-noise the source latents to the schedule point
    given by `strength` and run the remaining steps under the new prompt, so coarse motion is
    kept from the source while the appearance follows the prompt."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    device = source_latents.device
    dtype = pipe.transformer.dtype
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(prompt=prompt, negative_prompt=negative,
                                      do_classifier_free_guidance=True, device=device)
        pos, neg = pos.to(dtype), neg.to(dtype)
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps[steps - min(int(steps * strength), steps):]
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = randn_tensor(source_latents.shape, generator=gen, device=device, dtype=torch.float32)
        latents = pipe.scheduler.add_noise(source_latents.float(), noise, timesteps[:1])
        boundary = (pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
                    if pipe.config.boundary_ratio is not None else None)
        for t in timesteps:
            high = boundary is None or t >= boundary
            model = pipe.transformer if high else pipe.transformer_2
            scale = guidance if high else guidance_2
            x, ts = latents.to(dtype), t.expand(1)
            cond = model(hidden_states=x, timestep=ts, encoder_hidden_states=pos, return_dict=False)[0]
            uncond = model(hidden_states=x, timestep=ts, encoder_hidden_states=neg, return_dict=False)[0]
            latents = pipe.scheduler.step(uncond + scale * (cond - uncond), t, latents, return_dict=False)[0]
    return latents, len(timesteps)


def run_v2v(cfg: dict, root: Path, log) -> List[dict]:
    import torch
    from diffusers.utils import export_to_video
    m, spec = cfg["model"], cfg["v2v"]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(spec["source"])
    source_md5 = hashlib.md5(source.read_bytes()).hexdigest()
    frames = read_frames(source, m["frames"])
    width, height = frames[0].size
    targets = targets_from_config(cfg, spec["targets"])
    strengths = [float(s) for s in spec.get("strengths", [0.6])]
    log.info("source %s (%s) %dx%d %d frames; %d targets x %d strengths",
             source.name, source_md5[:8], width, height, len(frames), len(targets), len(strengths))
    pipe = load_pipeline(cfg, log)
    source_latents = encode_video(pipe, frames, height, width)
    slow = float(cfg.get("sampling", {}).get("slow_factor", 1.0))
    base_seed = int(cfg.get("experiment", {}).get("seed", 0))
    records = []
    t_all = time.time()
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as mf:
        for i, ((tid, rec), strength) in enumerate(
                ((t, s) for t in targets for s in strengths)):
            dst = root / f"{tid}_s{strength:.2f}.mp4"
            negative = V.NEGATIVE + V.NEGATIVE_BY_CLASS.get(rec["cls"], "")
            t0 = time.time()
            latents, steps_run = denoise_from(pipe, source_latents, rec["prompt"], negative, strength,
                                              m["steps"], m["guidance"], m.get("guidance_2") or m["guidance"],
                                              base_seed + i)
            export_to_video(decode_latents(pipe, latents), str(dst), fps=m["fps"])
            retimed = False
            if slow > 1.0:
                retimed, _, err = retime_cfr(dst, slow, m["fps"])
                if not retimed:
                    log.error("retime failed for %s: %s", dst.name, err)
            out = {**rec, "file": dst.name, "target_id": tid, "source": str(source),
                   "source_md5": source_md5, "strength": strength, "steps_run": steps_run,
                   "seed": base_seed + i, "negative": negative, "retimed": retimed,
                   "gen_seconds": round(time.time() - t0, 1), **(probe(dst) or {})}
            mf.write(json.dumps(out, ensure_ascii=False) + "\n")
            mf.flush()
            records.append(out)
            log.info("[%d/%d] %s strength %.2f (%d steps) %.1fs", i + 1, len(targets) * len(strengths),
                     dst.name, strength, steps_run, out["gen_seconds"])
    log.info("finished %d clips in %s", len(records), fmt_dur(time.time() - t_all))
    return records
