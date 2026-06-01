"""Generic HuggingFace video model with optional PEFT adapter.

Supports:
  Backbone: VideoMAE, TimeSformer, V-JEPA 2
  Adapter:  LoRA, DoRA, IA3, AdaLoRA, full (no adapter)
"""

import os
import torch
import torch.nn as nn
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    TimesformerForVideoClassification,
    AutoImageProcessor,
)
from peft import get_peft_model, PeftModel, LoraConfig, IA3Config, AdaLoraConfig
from stimbench.registry import register_model


# ─────────────────────────────────────────────────────────
# V-JEPA 2 lazy import + wrappers
# ─────────────────────────────────────────────────────────

def _get_vjepa2_classes():
    from transformers import VJEPA2ForVideoClassification, AutoVideoProcessor
    return VJEPA2ForVideoClassification, AutoVideoProcessor


class VJEPA2ModelWrapper(nn.Module):
    """Wraps V-JEPA 2 to accept pixel_values instead of pixel_values_videos."""
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, pixel_values=None, **kw):
        return self._model(pixel_values_videos=pixel_values, **kw)

    def __getattr__(self, name):
        if name == '_model':
            return super().__getattr__(name)
        return getattr(self._model, name)


class VJEPA2ProcessorWrapper:
    """Wraps V-JEPA 2 processor to return pixel_values key."""
    def __init__(self, processor):
        self._proc = processor

    def __call__(self, frames, return_tensors="pt"):
        out = self._proc(frames, return_tensors=return_tensors)
        if "pixel_values_videos" in out and "pixel_values" not in out:
            out["pixel_values"] = out.pop("pixel_values_videos")
        return out

    def __getattr__(self, name):
        return getattr(self._proc, name)


# ─────────────────────────────────────────────────────────
# Backbone registry and defaults
# ─────────────────────────────────────────────────────────

BACKBONE_MAP = {
    'videomae': (VideoMAEForVideoClassification, VideoMAEImageProcessor),
    'timesformer': (TimesformerForVideoClassification, AutoImageProcessor),
}

BACKBONE_IDS = {
    'videomae': 'MCG-NJU/videomae-base-finetuned-kinetics',
    'timesformer': 'facebook/timesformer-base-finetuned-k400',
}

VJEPA2_VARIANTS = {
    'facebook/vjepa2-vitl-fpc64-256': 'V-JEPA2-Small',
    'facebook/vjepa2-vitl-fpc16-256-ssv2': 'V-JEPA2-SSv2',
    'facebook/vjepa2-vitg-fpc64-256': 'V-JEPA2-Large',
}

DEFAULT_LORA_TARGETS = {
    'videomae': ['query', 'value'],
    'timesformer': ['qkv'],
    'vjepa2': ['q_proj', 'k_proj', 'v_proj'],
}

DEFAULT_IA3_CONFIG = {
    'videomae': {'targets': ['query', 'value', 'dense'], 'ff': ['dense']},
    'timesformer': {'targets': ['qkv', 'dense'], 'ff': ['dense']},
    'vjepa2': {'targets': ['q_proj', 'v_proj', 'fc1'], 'ff': ['fc1']},
}

DEFAULT_UNFREEZE = {
    'videomae': ['classifier', 'fc_norm', 'layernorm'],
    'timesformer': ['classifier', 'layernorm', 'norm'],
    'vjepa2': ['classifier', 'layernorm', 'norm', 'pooler'],
}


# ─────────────────────────────────────────────────────────
# PEFT config builder
# ─────────────────────────────────────────────────────────

def build_peft_config(adapter_cfg, family, training_cfg=None):
    """Build a PEFT config from YAML adapter section. Returns None for full FT."""
    atype = adapter_cfg['type']
    if atype == 'full':
        return None

    targets = adapter_cfg.get('targets', DEFAULT_LORA_TARGETS.get(family, ['query', 'value']))

    if atype == 'lora':
        return LoraConfig(
            r=adapter_cfg.get('r', 16),
            lora_alpha=adapter_cfg.get('alpha', 32),
            lora_dropout=adapter_cfg.get('dropout', 0.15),
            target_modules=targets,
            bias="none",
        )
    elif atype == 'dora':
        return LoraConfig(
            r=adapter_cfg.get('r', 16),
            lora_alpha=adapter_cfg.get('alpha', 32),
            lora_dropout=adapter_cfg.get('dropout', 0.15),
            target_modules=targets,
            use_dora=True,
            bias="none",
        )
    elif atype == 'ia3':
        ia3_defaults = DEFAULT_IA3_CONFIG.get(family, {'targets': targets, 'ff': []})
        return IA3Config(
            target_modules=adapter_cfg.get('targets', ia3_defaults['targets']),
            feedforward_modules=adapter_cfg.get('feedforward_modules', ia3_defaults['ff']),
        )
    elif atype == 'adalora':
        total_step = adapter_cfg.get('total_step', None)
        if total_step is None and training_cfg:
            steps_per_epoch = max(1, 270 // training_cfg.get('batch_size', 4))
            total_step = steps_per_epoch * training_cfg.get('epochs', 30)
        if total_step is None:
            total_step = 2000
        return AdaLoraConfig(
            init_r=adapter_cfg.get('r', 16),
            lora_alpha=adapter_cfg.get('alpha', 32),
            lora_dropout=adapter_cfg.get('dropout', 0.15),
            target_modules=targets,
            total_step=total_step,
        )
    else:
        raise ValueError(f"Unknown adapter type: {atype}")


# ─────────────────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────────────────

@register_model("hf_peft")
class HFPeftModel:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device

        mc = config['model']
        adapter_cfg = mc['adapter']
        num_labels = len(config['dataset']['classes'])

        # Resolve backbone family and ID
        backbone = mc.get('backbone', 'videomae')
        self._num_labels = num_labels
        self._adapter_type = adapter_cfg['type']
        self._is_full_ft = (self._adapter_type == 'full')

        # Detect family from backbone string
        if 'vjepa2' in backbone or 'vjepa' in backbone:
            self.family = 'vjepa2'
            self.backbone_id = backbone if '/' in backbone else mc.get('backbone_variant', 'facebook/vjepa2-vitl-fpc64-256')
        elif backbone in BACKBONE_IDS:
            self.family = backbone
            self.backbone_id = BACKBONE_IDS[backbone]
        elif '/' in backbone:
            # Full HF model ID provided directly
            if 'timesformer' in backbone.lower():
                self.family = 'timesformer'
            else:
                self.family = 'videomae'
            self.backbone_id = backbone
        else:
            self.family = backbone
            self.backbone_id = BACKBONE_IDS.get(backbone, backbone)

        self._is_vjepa2 = (self.family == 'vjepa2')

        # Load base model and processor
        if self._is_vjepa2:
            model_cls, proc_cls = _get_vjepa2_classes()
            base_model = model_cls.from_pretrained(
                self.backbone_id, num_labels=num_labels, ignore_mismatched_sizes=True)
            raw_proc = proc_cls.from_pretrained(self.backbone_id)
            self.processor = VJEPA2ProcessorWrapper(raw_proc)
        else:
            model_cls, proc_cls = BACKBONE_MAP[self.family]
            base_model = model_cls.from_pretrained(
                self.backbone_id, num_labels=num_labels, ignore_mismatched_sizes=True)
            self.processor = proc_cls.from_pretrained(self.backbone_id)

        # Apply PEFT adapter or set up full fine-tuning
        peft_config = build_peft_config(adapter_cfg, self.family, config.get('training'))

        if self._is_full_ft:
            # Full fine-tuning: all parameters trainable
            for p in base_model.parameters():
                p.requires_grad = True
            if self._is_vjepa2:
                self.model = VJEPA2ModelWrapper(base_model)
            else:
                self.model = base_model
        else:
            # PEFT: freeze base, inject adapter, unfreeze specific layers
            peft_model = get_peft_model(base_model, peft_config)

            # Unfreeze classifier, norms, pooler etc.
            unfreeze_keywords = DEFAULT_UNFREEZE.get(self.family, ['classifier'])
            for name, param in peft_model.named_parameters():
                if any(kw in name.lower() for kw in unfreeze_keywords):
                    param.requires_grad = True

            if self._is_vjepa2:
                self.model = VJEPA2ModelWrapper(peft_model)
            else:
                self.model = peft_model

        self.model = self.model.to(device)

        # Print summary
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  {self.family} + {self._adapter_type}")
        print(f"  Parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ─── Save ───

    def save(self, path):
        if self._is_full_ft:
            torch.save(self.model.state_dict(), path)
            print(f"  Saved full model to: {path}")
            return

        save_dir = path.replace('.pt', '_peft')
        os.makedirs(save_dir, exist_ok=True)

        # Save PEFT adapter
        peft_model = self.model._model if self._is_vjepa2 else self.model
        peft_model.save_pretrained(save_dir)

        # Save extra trainable weights (classifier, pooler, norms)
        adapter_keywords = ['lora_', 'ia3_', 'ranknum']
        extra_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(kw in name for kw in adapter_keywords):
                extra_state[name] = param.data.cpu().clone()

        if extra_state:
            torch.save(extra_state, os.path.join(save_dir, 'extra_trainable.pt'))

        print(f"  Saved adapter to: {save_dir}/")

    # ─── Load ───

    def load(self, path):
        if self._is_full_ft:
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model = self.model.to(self.device)
            return

        save_dir = path.replace('.pt', '_peft')
        if not os.path.isdir(save_dir):
            # Fallback: load raw state dict
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state)
            return

        # Rebuild base model + load PEFT adapter
        if self._is_vjepa2:
            model_cls, _ = _get_vjepa2_classes()
        else:
            model_cls = BACKBONE_MAP[self.family][0]

        base_model = model_cls.from_pretrained(
            self.backbone_id, num_labels=self._num_labels, ignore_mismatched_sizes=True)
        peft_model = PeftModel.from_pretrained(base_model, save_dir)

        if self._is_vjepa2:
            self.model = VJEPA2ModelWrapper(peft_model)
        else:
            self.model = peft_model

        # Load extra trainable weights (classifier, pooler, norms)
        extra_path = os.path.join(save_dir, 'extra_trainable.pt')
        if os.path.exists(extra_path):
            extra_state = torch.load(extra_path, map_location='cpu')
            model_state = self.model.state_dict()
            for k, v in extra_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
            self.model.load_state_dict(model_state)

        self.model = self.model.to(self.device)
