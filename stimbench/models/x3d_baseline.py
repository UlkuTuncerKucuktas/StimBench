import torch
import torch.nn as nn
from stimbench.registry import register_model
from stimbench.models.base import VideoProcessor, VideoModelWrapper


def build_x3d(variant, num_classes):
    model_name = f'x3d_{variant}'
    model = torch.hub.load('facebookresearch/pytorchvideo',
                           model_name, pretrained=True)
    # X3D head: model.blocks[5].proj
    model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, num_classes)
    model.blocks[5].activation = None  # remove softmax, we use CE loss
    print(f"  Loaded: pytorchvideo {model_name} (Kinetics-400)")
    return model


# X3D input sizes per variant
X3D_FRAMES = {'xs': 4, 's': 13, 'm': 16, 'l': 16}
X3D_SIZE = {'xs': 182, 's': 182, 'm': 256, 'l': 312}


@register_model("x3d")
class X3DBaseline:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        n_cls = len(config['dataset']['classes'])
        variant = config['model'].get('variant', 's')

        backbone = build_x3d(variant, n_cls)
        self.model = VideoModelWrapper(backbone).to(device)

        size = X3D_SIZE.get(variant, 182)
        self.processor = VideoProcessor(size=size)

        total = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {train:,} / {total:,} ({100*train/total:.2f}%)")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
