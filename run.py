#!/usr/bin/env python3
"""
StimBench benchmark runner.

Usage:
    python run.py --config configs/videomae_lora.yaml
    python run.py --config configs/videomae_lora.yaml --data_dir /path/to/StimBench
"""

import os, json, time, argparse, shutil, warnings
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

from stimbench.registry import MODEL_REGISTRY, EVAL_REGISTRY
import stimbench.models
import stimbench.eval
from stimbench.data import StimBenchDataset
from stimbench.reporting import save_history_csv, save_plots, save_confusion_matrix


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_key(config):
    """Resolve model registry key from config. Supports both new and old format."""
    # New format: model.type = "i3d" / "video_swin" / "hf_peft" / "x3d"
    key = config['model'].get('type', None)
    if key:
        return key
    # Old format: model.adapter.type = "lora" → "videomae_lora"
    adapter_type = config['model'].get('adapter', {}).get('type', 'lora')
    return f"videomae_{adapter_type}"


def train_one_epoch(model, loader, optimizer, scheduler, device, class_weights, scaler):
    model.model.train()
    ls = model.config['training'].get('loss', {})
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=ls.get('label_smoothing', 0.0)
    )
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for pixels, labels in pbar:
        pixels, labels = pixels.to(device), labels.to(device)

        if scaler:
            with torch.amp.autocast('cuda'):
                try:
                    outputs = model.model(pixel_values=pixels)
                except (RuntimeError, ValueError):
                    pixels = pixels.permute(0, 2, 1, 3, 4)
                    outputs = model.model(pixel_values=pixels)
                loss = criterion(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            try:
                outputs = model.model(pixel_values=pixels)
            except (RuntimeError, ValueError):
                pixels = pixels.permute(0, 2, 1, 3, 4)
                outputs = model.model(pixel_values=pixels)
            loss = criterion(outputs.logits, labels)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

    scheduler.step()
    return total_loss / total, correct / total


def save_misclassified(dataset, preds, labels, classes, output_dir, tag):
    mis_dir = os.path.join(output_dir, f'misclassified_{tag}')
    os.makedirs(mis_dir, exist_ok=True)
    count = 0
    for idx in range(len(dataset)):
        path, label, _ = dataset.samples[idx]
        if preds[idx] != label:
            count += 1
            dst = os.path.join(mis_dir,
                f"TRUE_{classes[label]}__PRED_{classes[preds[idx]]}__{os.path.basename(path)}")
            shutil.copy2(path, dst)
    print(f"  Misclassified saved: {count} -> {mis_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--data_dir', default=None, help='Override dataset path')
    parser.add_argument('--output_dir', default=None, help='Override output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['experiment'].get('seed', 42))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = args.data_dir or config['dataset']['path']
    output_dir = args.output_dir or os.path.join('results', config['experiment']['name'])
    os.makedirs(output_dir, exist_ok=True)

    classes = config['dataset']['classes']
    tc = config['training']

    print(f"{'='*60}")
    print(f"StimBench — {config['experiment']['name']}")
    print(f"{'='*60}")
    print(f"  Config:  {args.config}")
    print(f"  Device:  {device}")
    print(f"  Data:    {data_dir}")
    print(f"  Classes: {classes}")

    # Download from HF if path looks like a repo ID
    if '/' in data_dir and not os.path.exists(data_dir):
        print(f"\n  Downloading from HuggingFace: {data_dir}")
        from huggingface_hub import snapshot_download
        data_dir = snapshot_download(repo_id=data_dir, repo_type='dataset')
        print(f"  Downloaded to: {data_dir}")

    # Build model from registry
    model_key = resolve_model_key(config)
    print(f"\n  Loading model: {model_key}")
    model_cls = MODEL_REGISTRY[model_key]
    model = model_cls(config, device)

    # Load data
    print(f"\n  Loading dataset...")
    train_ds = StimBenchDataset(data_dir, 'train', model.processor, config, mode='train')
    test_ds = StimBenchDataset(data_dir, 'test', model.processor, config, mode='test')
    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")
    train_ds.cache_all()
    test_ds.cache_all()

    train_labels = train_ds.get_labels()
    for i, name in enumerate(classes):
        print(f"    {name:<15} {train_labels.count(i):>3}")

    # Class weights
    num_classes = len(classes)
    label_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = torch.tensor(len(train_labels) / (num_classes * label_counts), dtype=torch.float32)

    # Sampler
    if tc.get('balanced_sampling', False):
        sample_weights = [class_weights[l].item() for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        sampler = None

    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'],
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=0, pin_memory=True, drop_last=True)

    # Optimizer
    opt_cfg = tc['optimizer']
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=opt_cfg['lr'], weight_decay=opt_cfg.get('weight_decay', 0.01))

    sch_cfg = tc.get('scheduler', {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['epochs'], eta_min=sch_cfg.get('eta_min', 0))

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' and tc.get('fp16', False) else None

    # =========================================================
    # Train
    # =========================================================
    print(f"\n{'='*60}")
    print(f"Training ({tc['epochs']} epochs)...")
    print(f"{'='*60}")

    best_f1, history = 0, []
    best_path = os.path.join(output_dir, 'best_model.pt')

    for epoch in range(1, tc['epochs'] + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{tc['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights, scaler)

        result = EVAL_REGISTRY['1x1'](model, test_ds, config, device)
        elapsed = time.time() - t0

        print(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Test  — acc: {result['accuracy']:.4f}, F1(w): {result['f1_weighted']:.4f}, F1(m): {result['f1_macro']:.4f}  ({elapsed:.0f}s)")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'test_acc': result['accuracy'], 'test_f1w': result['f1_weighted'],
            'test_f1m': result['f1_macro'],
        })

        if result['f1_weighted'] > best_f1:
            best_f1 = result['f1_weighted']
            model.save(best_path)
            print(f"  * New best F1(w): {best_f1:.4f}")

    # Save training artifacts
    save_history_csv(history, output_dir)
    save_plots(history, output_dir)

    # =========================================================
    # Final evaluation with all protocols
    # =========================================================
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION")
    print(f"{'='*60}")

    model.load(best_path)
    all_results = {}

    for protocol in config['evaluation'].get('protocols', ['1x1']):
        if protocol not in EVAL_REGISTRY:
            print(f"  Skipping unknown protocol: {protocol}")
            continue

        result = EVAL_REGISTRY[protocol](model, test_ds, config, device)
        all_results[protocol] = {
            'accuracy': result['accuracy'],
            'f1_weighted': result['f1_weighted'],
            'f1_macro': result['f1_macro'],
            'confusion_matrix': result['confusion_matrix'],
        }

        print(f"\n  [{protocol}]")
        print(f"  Accuracy:      {result['accuracy']:.4f}")
        print(f"  F1 (weighted): {result['f1_weighted']:.4f}")
        print(f"  F1 (macro):    {result['f1_macro']:.4f}")
        print(result['report'])

        # Always save confusion matrix
        save_confusion_matrix(result['confusion_matrix'], classes, output_dir, protocol)

        # Optionally save misclassified clips
        if config['evaluation'].get('save_misclassified', False):
            save_misclassified(test_ds, result['preds'], result['labels'],
                               classes, output_dir, protocol)

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Protocol':<20} {'Acc':>8} {'F1(w)':>8} {'F1(m)':>8}")
    print(f"  {'_'*44}")
    for p, r in all_results.items():
        print(f"  {p:<20} {r['accuracy']:>8.4f} {r['f1_weighted']:>8.4f} {r['f1_macro']:>8.4f}")

    # Save results
    output = {
        'experiment': config['experiment']['name'],
        'config': config,
        'results': all_results,
        'history': history,
    }
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {results_path}")


if __name__ == '__main__':
    main()
