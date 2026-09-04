import hashlib
import json
import time
from pathlib import Path
from typing import List

from . import vocab as V
from .generate import fmt_dur
from .v2v import read_frames, targets_from_config
from .video import probe, retime_cfr


def flow_frames(frames, device: str = "cuda"):
    """RAFT flow between consecutive frames drawn with the standard colour wheel, the encoding
    VACE's own flow annotator produces; the first flow is repeated so the count matches."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from torchvision.utils import flow_to_image
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    transform = weights.transforms()
    imgs = torch.stack([torch.from_numpy(np.asarray(f)).permute(2, 0, 1) for f in frames]).float() / 255
    out = []
    with torch.no_grad():
        for a, b in zip(imgs[:-1], imgs[1:]):
            x1, x2 = transform(a[None].to(device), b[None].to(device))
            flow = model(x1, x2)[-1]
            out.append(Image.fromarray(flow_to_image(flow[0]).permute(1, 2, 0).cpu().numpy()))
    del model
    return [out[0]] + out


def load_vace_pipeline(cfg: dict, log):
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler
    m, sp = cfg["model"], cfg.get("speed", {})
    vae = AutoencoderKLWan.from_pretrained(m["repo"], subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(m["repo"], vae=vae, torch_dtype=torch.bfloat16)
    if m.get("flow_shift"):
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=m["flow_shift"])
    pipe.to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if sp.get("compile"):
        pipe.transformer.compile(mode=sp.get("compile_mode", "default"), dynamic=False)
    pipe.set_progress_bar_config(disable=True)
    log.info("vace pipeline %s loaded", m["repo"])
    return pipe


def load_controls(spec: dict, n_frames: int, log) -> dict:
    """Control videos by name. `controls` maps names to flow videos rendered elsewhere
    (`gen_synth.py flow`), for hosts without torchvision; otherwise the flow of `source`
    is computed here, mirrored as well when an arm asks for it."""
    from PIL import ImageOps
    if spec.get("controls"):
        out = {name: read_frames(Path(path), n_frames) for name, path in spec["controls"].items()}
        log.info("control videos read: %s", ", ".join(out))
        return out
    frames = read_frames(Path(spec["source"]), n_frames)
    out = {"plain": flow_frames(frames)}
    if any(a.get("control") == "mirror" for a in spec.get("arms", [])):
        out["mirror"] = flow_frames([ImageOps.mirror(f) for f in frames])
    log.info("flow control computed: %s", ", ".join(out))
    return out


def write_flow_videos(source: Path, out_dir: Path, n_frames: int, fps: int, device: str = "cpu"):
    """flow.mp4 and flow_mirror.mp4 next to each other, near-lossless 4:4:4 so the colour
    wheel survives; the frame count is checked on read-back."""
    import subprocess
    import tempfile
    from PIL import ImageOps
    frames = read_frames(source, n_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fr in (("flow", frames), ("flow_mirror", [ImageOps.mirror(f) for f in frames])):
        flows = flow_frames(fr, device=device)
        with tempfile.TemporaryDirectory() as tmp:
            for i, im in enumerate(flows):
                im.save(Path(tmp) / f"{i:03d}.png")
            dst = out_dir / f"{name}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                            "-i", str(Path(tmp) / "%03d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv444p", "-crf", "4", str(dst)], check=True)
        read_frames(dst, n_frames)
        written.append(dst)
    return written


def run_vace(cfg: dict, root: Path, log) -> List[dict]:
    """One clip per (target prompt, arm): the source clip's optical flow drives the movement,
    the prompt supplies a new child and scene, the arm sets control strength and mirroring."""
    import torch
    from diffusers.utils import export_to_video
    m, spec = cfg["model"], cfg["vace"]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(spec["source"])
    source_md5 = hashlib.md5(source.read_bytes()).hexdigest()
    arms = spec.get("arms") or [{"id": "full", "scale": 1.0, "control": "plain"}]
    targets = targets_from_config(cfg, spec["targets"])
    pairing = spec.get("pairing", "all")
    jobs = ([(t, a) for t in targets for a in arms] if pairing == "all"
            else [(t, arms[i % len(arms)]) for i, t in enumerate(targets)])
    controls = load_controls(spec, m["frames"], log)
    width, height = controls["plain"][0].size
    log.info("source %s (%s) %dx%d; %d targets, %d arms, %d clips",
             source.name, source_md5[:8], width, height, len(targets), len(arms), len(jobs))
    pipe = load_vace_pipeline(cfg, log)
    slow = float(cfg.get("sampling", {}).get("slow_factor", 1.0))
    base_seed = int(cfg.get("experiment", {}).get("seed", 0))
    negative_base = spec.get("negative", V.NEGATIVE)
    records = []
    t_all = time.time()
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as mf:
        for i, ((tid, rec), arm) in enumerate(jobs):
            dst = root / f"{tid}_{arm['id']}.mp4"
            negative = negative_base + V.NEGATIVE_BY_CLASS.get(rec["cls"], "")
            seed = base_seed + i
            t0 = time.time()
            control = arm.get("control", "plain")
            frames_out = pipe(prompt=rec["prompt"], negative_prompt=negative,
                              video=controls[control],
                              conditioning_scale=float(arm.get("scale", 1.0)),
                              height=height, width=width, num_frames=m["frames"],
                              num_inference_steps=m["steps"], guidance_scale=m["guidance"],
                              generator=torch.Generator(device="cpu").manual_seed(seed)).frames[0]
            export_to_video(frames_out, str(dst), fps=m["fps"])
            retimed = False
            if slow > 1.0:
                retimed, _, err = retime_cfr(dst, slow, m["fps"])
                if not retimed:
                    log.error("retime failed for %s: %s", dst.name, err)
            out = {**rec, "file": dst.name, "target_id": tid, "arm": arm["id"], "control": control,
                   "conditioning_scale": float(arm.get("scale", 1.0)),
                   "source": str(source), "source_md5": source_md5, "seed": seed,
                   "negative": negative, "retimed": retimed,
                   "gen_seconds": round(time.time() - t0, 1), **(probe(dst) or {})}
            mf.write(json.dumps(out, ensure_ascii=False) + "\n")
            mf.flush()
            records.append(out)
            log.info("[%d/%d] %s scale %.2f control %s %.1fs", i + 1, len(jobs), dst.name,
                     out["conditioning_scale"], control, out["gen_seconds"])
    log.info("finished %d clips in %s", len(records), fmt_dur(time.time() - t_all))
    return records
