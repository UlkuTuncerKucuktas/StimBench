import json
import time
from pathlib import Path
from typing import List

from .video import probe, retime_cfr


def extract_frame(clip: Path, index: int, out_png: Path):
    import imageio.v3 as iio
    frames = iio.imread(str(clip), plugin="pyav") if _has_pyav() else _read_all(clip)
    index = max(0, min(index, len(frames) - 1))
    iio.imwrite(str(out_png), frames[index])
    return out_png


def _has_pyav():
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def _read_all(clip: Path):
    import imageio
    import numpy as np
    with imageio.get_reader(str(clip), format="ffmpeg") as r:
        return np.stack([f for f in r])


def load_i2v_pipeline(cfg: dict, log):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
    m = cfg["model"]
    vae = AutoencoderKLWan.from_pretrained(m["repo"], subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(m["repo"], vae=vae, torch_dtype=torch.bfloat16)
    if m.get("flow_shift"):
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=m["flow_shift"])
    pipe.to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if cfg.get("speed", {}).get("compile"):
        for t in (getattr(pipe, "transformer", None), getattr(pipe, "transformer_2", None)):
            if t is not None:
                t.compile(dynamic=False)
    log.info("i2v pipeline %s loaded", m["repo"])
    return pipe


def run_i2v(cfg: dict, images: List[Path], root: Path, log) -> List[dict]:
    """One clip per first-frame image, same motion prompt, retimed like the T2V clips."""
    import torch
    from diffusers.utils import export_to_video, load_image
    m, spec = cfg["model"], cfg["i2v"]
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    pipe = load_i2v_pipeline(cfg, log)
    slow = float(cfg.get("sampling", {}).get("slow_factor", 1.0))
    records = []
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as mf:
        for i, img_path in enumerate(images):
            img = load_image(str(img_path))
            w, h = img.size
            dst = root / f"{img_path.stem}_i2v.mp4"
            t0 = time.time()
            gen = torch.Generator(device="cpu").manual_seed(int(spec.get("seed", 0)) + i)
            kwargs = {"guidance_scale_2": m["guidance_2"]} if m.get("guidance_2") is not None else {}
            frames = pipe(image=img, prompt=spec["prompt"], negative_prompt=spec.get("negative", ""),
                          height=h, width=w, num_frames=m["frames"], num_inference_steps=m["steps"],
                          guidance_scale=m["guidance"], generator=gen, **kwargs).frames[0]
            export_to_video(frames, str(dst), fps=m["fps"])
            retimed = False
            if slow > 1.0:
                retimed, _, err = retime_cfr(dst, slow, m["fps"])
                if not retimed:
                    log.error("retime failed for %s: %s", dst.name, err)
            rec = {"file": dst.name, "source_frame": str(img_path), "prompt": spec["prompt"],
                   "negative": spec.get("negative", ""), "seed": int(spec.get("seed", 0)) + i,
                   "width": w, "height": h, "retimed": retimed, "gen_seconds": round(time.time() - t0, 1),
                   **(probe(dst) or {})}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n"); mf.flush()
            records.append(rec)
            log.info("[%d/%d] %s %.1fs", i + 1, len(images), dst.name, rec["gen_seconds"])
    return records
