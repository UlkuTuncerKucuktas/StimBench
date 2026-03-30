"""I3D RGB baseline.

Primary:  pytorchvideo i3d_r50 (Kinetics-400) via torch.hub
Fallback: torchvision r3d_18  (Kinetics-400) — same 3D CNN family

Reference: Ali et al. (2022) — I3D on SSBD: 76.92% acc, F1=0.60
"""

import torch
import torch.nn as nn
from stimbench.registry import register_model
from stimbench.models.base import VideoProcessor, VideoModelWrapper


def build_i3d(num_classes):
    """Try pytorchvideo I3D, fall back to torchvision R3D-18."""
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo',
                               'i3d_r50', pretrained=True)
        model.blocks[6].proj = nn.Linear(
            model.blocks[6].proj.in_features, num_classes)
        model.blocks[6].activation = None
        print("  Loaded: pytorchvideo i3d_r50 (Kinetics-400)")
        return model
    except Exception as e:
        print(f"  pytorchvideo unavailable ({e}), using torchvision r3d_18")
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print("  Loaded: torchvision r3d_18 (Kinetics-400)")
        return model


@register_model("i3d")
class I3DBaseline:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        n_cls = len(config['dataset']['classes'])

        backbone = build_i3d(n_cls)
        self.model = VideoModelWrapper(backbone).to(device)
        self.processor = VideoProcessor(size=224)

        total = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {train:,} / {total:,} ({100*train/total:.2f}%)")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
