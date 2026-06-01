import numpy as np
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from stimbench.registry import register_evaluator
from stimbench.data.dataset import read_video_frames


def compute_metrics(preds, labels, class_names):
    preds = np.array(preds)
    labels = np.array(labels)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1_weighted': float(f1_score(labels, preds, average='weighted')),
        'f1_macro': float(f1_score(labels, preds, average='macro')),
        'report': classification_report(labels, preds, target_names=class_names, digits=4),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
        'preds': preds,
        'labels': labels,
    }


@register_evaluator("1x1")
@torch.no_grad()
def evaluate_1x1(model, dataset, config, device):
    model.model.eval()
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                        shuffle=False, num_workers=0, pin_memory=True)
    all_preds, all_labels = [], []
    for pixels, labels in tqdm(loader, desc="  1x1  ", leave=False):
        pixels = pixels.to(device)
        try:
            outputs = model.model(pixel_values=pixels)
        except (RuntimeError, ValueError):
            pixels = pixels.permute(0, 2, 1, 3, 4)
            outputs = model.model(pixel_values=pixels)
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return compute_metrics(all_preds, all_labels, config['dataset']['classes'])


def get_multiclip_indices(total_frames, num_frames, stride, n_clips=5):
    window = num_frames * stride
    clips = []
    if total_frames >= window:
        max_start = total_frames - window
        starts = np.linspace(0, max_start, n_clips, dtype=int)
        for s in starts:
            indices = np.arange(s, s + window, stride)[:num_frames]
            clips.append(np.clip(indices, 0, total_frames - 1))
    else:
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.arange(num_frames) % total_frames
        clips = [indices] * n_clips
    return clips


@register_evaluator("5x1")
@torch.no_grad()
def evaluate_multiview(model, dataset, config, device, n_clips=5, n_crops=1):
    model.model.eval()
    num_frames = config['preprocessing']['num_frames']
    stride = config['preprocessing']['stride']
    all_preds, all_labels = [], []

    for idx in tqdm(range(len(dataset)), desc=f"  {n_clips}x{n_crops}  ", leave=False):
        path, label, _ = dataset.samples[idx]
        frames = read_video_frames(path)
        if len(frames) == 0:
            all_preds.append(0)
            all_labels.append(label)
            continue

        clip_indices = get_multiclip_indices(len(frames), num_frames, stride, n_clips)
        logits_sum = None
        for clip_idx in clip_indices:
            sampled = [frames[i] for i in clip_idx]
            inputs = model.processor(sampled, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            try:
                outputs = model.model(pixel_values=pixels)
            except (RuntimeError, ValueError):
                pixels = pixels.permute(0, 2, 1, 3, 4)
                outputs = model.model(pixel_values=pixels)
            logits_sum = outputs.logits.cpu() if logits_sum is None else logits_sum + outputs.logits.cpu()

        all_preds.append(logits_sum.argmax(dim=-1).item())
        all_labels.append(label)

    return compute_metrics(all_preds, all_labels, config['dataset']['classes'])


@register_evaluator("5x3")
@torch.no_grad()
def evaluate_5x3(model, dataset, config, device):
    return evaluate_multiview(model, dataset, config, device, n_clips=5, n_crops=3)


@register_evaluator("sliding_window")
@torch.no_grad()
def evaluate_sliding_window(model, dataset, config, device):
    model.model.eval()
    num_frames = config['preprocessing']['num_frames']
    stride = config['preprocessing']['stride']
    overlap = config['evaluation'].get('sliding_window', {}).get('overlap', 0.2)
    all_preds, all_labels = [], []

    for idx in tqdm(range(len(dataset)), desc="  Slide ", leave=False):
        path, label, _ = dataset.samples[idx]
        frames = read_video_frames(path)
        if len(frames) == 0:
            all_preds.append(0)
            all_labels.append(label)
            continue

        total = len(frames)
        window = num_frames * stride
        step = max(1, int(window * (1 - overlap)))
        best_conf, best_pred = -1, 0

        starts = list(range(0, max(1, total - window + 1), step))
        if total >= window and starts[-1] != total - window:
            starts.append(total - window)
        if not starts:
            starts = [0]

        for s in starts:
            if total >= window:
                indices = np.arange(s, s + window, stride)[:num_frames]
                indices = np.clip(indices, 0, total - 1)
            elif total >= num_frames:
                indices = np.linspace(0, total - 1, num_frames, dtype=int)
            else:
                indices = np.arange(num_frames) % total

            sampled = [frames[i] for i in indices]
            inputs = model.processor(sampled, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            try:
                outputs = model.model(pixel_values=pixels)
            except (RuntimeError, ValueError):
                pixels = pixels.permute(0, 2, 1, 3, 4)
                outputs = model.model(pixel_values=pixels)
            probs = torch.softmax(outputs.logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_pred = pred.item()

        all_preds.append(best_pred)
        all_labels.append(label)

    return compute_metrics(all_preds, all_labels, config['dataset']['classes'])
