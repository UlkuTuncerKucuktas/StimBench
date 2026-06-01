#!/usr/bin/env python3
"""Smoke test — verify all configs load and run 1 forward pass.
Run this BEFORE run_all.sh to catch errors in seconds, not hours.

Usage: python smoke_test.py
"""

import os, glob, yaml, sys, time
import torch
import numpy as np

sys.path.insert(0, '.')
from stimbench.registry import MODEL_REGISTRY, EVAL_REGISTRY
import stimbench.models
import stimbench.eval

def fake_frames(n=16, size=224):
    """Generate dummy video frames."""
    return [np.random.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)

    name = config['experiment']['name']
    model_key = config['model'].get('type', None)
    if model_key is None:
        adapter_type = config['model'].get('adapter', {}).get('type', 'lora')
        model_key = f"videomae_{adapter_type}"

    print(f"  {name:<30} [{model_key}]", end="", flush=True)

    if model_key not in MODEL_REGISTRY:
        print(f"  FAIL — '{model_key}' not in registry. Available: {list(MODEL_REGISTRY.keys())}")
        return False

    try:
        t0 = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MODEL_REGISTRY[model_key](config, device)

        # Generate dummy input through the processor
        num_frames = config['preprocessing']['num_frames']
        size = config['preprocessing'].get('resize', 224)
        frames = fake_frames(num_frames, size)
        inputs = model.processor(frames, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)

        # Forward pass
        model.model.eval()
        with torch.no_grad():
            outputs = model.model(pixel_values=pixels)
            logits = outputs.logits

        n_cls = len(config['dataset']['classes'])
        assert logits.shape[-1] == n_cls, f"Expected {n_cls} classes, got {logits.shape[-1]}"

        # Backward pass (quick check gradients flow)
        model.model.train()
        outputs2 = model.model(pixel_values=pixels)
        loss = outputs2.logits.sum()
        loss.backward()

        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.model.parameters())
        elapsed = time.time() - t0

        print(f"  OK  ({trainable/1e6:.1f}M/{total/1e6:.1f}M params, {elapsed:.1f}s)")

        # Cleanup GPU memory
        del model, outputs, outputs2, loss, pixels
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"  FAIL — {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return False


def main():
    configs = sorted(glob.glob('configs/*.yaml'))
    if not configs:
        print("No configs found in configs/")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"StimBench Smoke Test — {len(configs)} configs")
    print(f"{'='*60}")
    print(f"  Registry: {list(MODEL_REGISTRY.keys())}")
    print(f"  Protocols: {list(EVAL_REGISTRY.keys())}")
    print()

    passed, failed = [], []
    for path in configs:
        ok = test_config(path)
        (passed if ok else failed).append(os.path.basename(path))

    print(f"\n{'='*60}")
    print(f"  PASSED: {len(passed)}/{len(configs)}")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"  All configs OK — safe to run_all.sh")


if __name__ == '__main__':
    main()
