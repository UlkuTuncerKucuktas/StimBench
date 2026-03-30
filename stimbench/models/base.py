"""Base wrappers for StimBench models.

Every model in StimBench must expose:
    .model      — nn.Module, forward(pixel_values=Tensor) → obj.logits
    .processor  — __call__(list[np.ndarray], return_tensors="pt") → {"pixel_values": Tensor}
    .save(path) / .load(path)

HuggingFace models (VideoMAE) already match this interface.
For torchvision / torch.hub models, use the helpers below.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn


class ModelOutput:
    """So we can do `output.logits` like HuggingFace."""
    def __init__(self, logits):
        self.logits = logits


class VideoProcessor:
    """ImageNet-normalised video preprocessor.
    Matches the interface of VideoMAEImageProcessor:
        processor(frames, return_tensors="pt") → {"pixel_values": (1, C, T, H, W)}
    """
    def __init__(self, size=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = np.float32(mean)
        self.std = np.float32(std)

    def __call__(self, frames, return_tensors="pt"):
        out = []
        for f in frames:
            f = cv2.resize(f, (self.size, self.size)).astype(np.float32) / 255.0
            out.append((f - self.mean) / self.std)
        # (T,H,W,C) → (C,T,H,W) → (1,C,T,H,W)
        video = np.stack(out).transpose(3, 0, 1, 2)
        tensor = torch.from_numpy(video).unsqueeze(0)
        return {"pixel_values": tensor}


class VideoModelWrapper(nn.Module):
    """Wraps any model(tensor) → tensor  into  model(pixel_values=) → .logits"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values=None, **kw):
        return ModelOutput(logits=self.backbone(pixel_values))
