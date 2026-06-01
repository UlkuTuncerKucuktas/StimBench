import torch
import torch.nn as nn
from torchvision.models.video import (
    swin3d_t, Swin3D_T_Weights,
    swin3d_s, Swin3D_S_Weights,
    swin3d_b, Swin3D_B_Weights,
)
from stimbench.registry import register_model
from stimbench.models.base import VideoProcessor, VideoModelWrapper

SWIN_VARIANTS = {
    't': (swin3d_t, Swin3D_T_Weights.KINETICS400_V1),
    's': (swin3d_s, Swin3D_S_Weights.KINETICS400_V1),
    'b': (swin3d_b, Swin3D_B_Weights.KINETICS400_V1),
}


@register_model("video_swin")
class VideoSwinBaseline:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        n_cls = len(config['dataset']['classes'])
        variant = config['model'].get('variant', 't')

        build_fn, weights = SWIN_VARIANTS[variant]
        backbone = build_fn(weights=weights)
        backbone.head = nn.Linear(backbone.head.in_features, n_cls)
        print(f"  Loaded: torchvision swin3d_{variant} (Kinetics-400)")

        self.model = VideoModelWrapper(backbone).to(device)
        self.processor = VideoProcessor(
            size=224,
            mean=(0.4850, 0.4560, 0.4060),
            std=(0.2290, 0.2240, 0.2250),
        )

        total = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {train:,} / {total:,} ({100*train/total:.2f}%)")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
