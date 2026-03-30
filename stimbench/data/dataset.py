import os
import csv
import hashlib
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_frames(frames, num_frames, stride, mode="train"):
    total = len(frames)
    window = num_frames * stride
    if total >= window:
        if mode == "train":
            start = np.random.randint(0, max(1, total - window))
        else:
            start = (total - window) // 2
        indices = np.arange(start, start + window, stride)[:num_frames]
        return np.clip(indices, 0, total - 1)
    elif total >= num_frames:
        return np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        return np.arange(num_frames) % total


class StimBenchDataset(Dataset):
    def __init__(self, root, split, processor, config, mode="train"):
        self.processor = processor
        self.num_frames = config['preprocessing']['num_frames']
        self.stride = config['preprocessing']['stride']
        self.mode = mode
        self.classes = config['dataset']['classes']
        self.aug = config['preprocessing'].get('augmentation', {})
        self.samples = []
        self._cache = {}

        meta_path = os.path.join(root, 'metadata.csv')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                for row in csv.DictReader(f):
                    if row['split'] != split:
                        continue
                    if row['label'] not in self.classes:
                        continue
                    label = self.classes.index(row['label'])
                    path = os.path.join(root, row['file_name'])
                    gid = row.get('group_id', '')
                    if os.path.exists(path):
                        self.samples.append((path, label, gid))
        else:
            split_dir = os.path.join(root, split)
            for cat in sorted(os.listdir(split_dir)):
                if not os.path.isdir(os.path.join(split_dir, cat)):
                    continue
                if cat not in self.classes:
                    continue
                label = self.classes.index(cat)
                cat_dir = os.path.join(split_dir, cat)
                for f in sorted(os.listdir(cat_dir)):
                    if f.endswith('.mp4'):
                        self.samples.append((os.path.join(cat_dir, f), label, f))

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [s[1] for s in self.samples]

    def _cache_key(self, path):
        res = getattr(self.processor, 'size', 224)
        if isinstance(res, dict):
            h = res.get('height', res.get('shortest_edge', 224))
        elif isinstance(res, (int, float)):
            h = int(res)
        else:
            h = 224
        key = f"{path}_{self.num_frames}_{self.stride}_{h}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _get_cache_dir(self):
        base = os.path.dirname(self.samples[0][0])
        cache_dir = os.path.abspath(os.path.join(base, '..', '.tensor_cache',
                                                  f"{self.mode}_f{self.num_frames}_s{self.stride}"))
        return cache_dir

    def cache_all(self):
        """Cache preprocessed tensors to disk. Instant on subsequent runs."""
        from tqdm import tqdm

        cache_dir = self._get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        # Check if full cache already exists
        all_cached = True
        for idx, (path, label, gid) in enumerate(self.samples):
            cache_path = os.path.join(cache_dir, f"{self._cache_key(path)}.pt")
            if os.path.exists(cache_path):
                self._cache[idx] = cache_path
            else:
                all_cached = False

        if all_cached:
            print(f'    {self.mode}: {len(self.samples)} tensors loaded from disk cache (instant)')
            return

        # Decode missing ones
        cached, decoded = 0, 0
        for idx, (path, label, gid) in enumerate(tqdm(self.samples, desc=f'    Cache {self.mode}', leave=True)):
            if idx in self._cache:
                cached += 1
                continue

            cache_path = os.path.join(cache_dir, f"{self._cache_key(path)}.pt")
            frames = read_video_frames(path)
            if len(frames) == 0:
                frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
            indices = sample_frames(frames, self.num_frames, self.stride, mode="test")
            sampled = [frames[i] for i in indices]
            tensor = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
            torch.save(tensor, cache_path)
            self._cache[idx] = cache_path
            decoded += 1

        print(f'    {cached} from disk cache, {decoded} newly decoded')

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]

        if idx in self._cache:
            tensor = torch.load(self._cache[idx], weights_only=True)
            if self.mode == "train":
                if np.random.rand() < self.aug.get('horizontal_flip', 0):
                    tensor = torch.flip(tensor, dims=[-1])
                if np.random.rand() < self.aug.get('temporal_reverse', 0):
                    tensor = torch.flip(tensor, dims=[0])
            return tensor, torch.tensor(label, dtype=torch.long)

        # Fallback: decode on the fly
        frames = read_video_frames(path)
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        indices = sample_frames(frames, self.num_frames, self.stride, self.mode)
        sampled = [frames[i] for i in indices]

        if self.mode == "train":
            if np.random.rand() < self.aug.get('horizontal_flip', 0):
                sampled = [np.fliplr(f).copy() for f in sampled]
            if np.random.rand() < self.aug.get('temporal_reverse', 0):
                sampled = sampled[::-1]

        inputs = self.processor(sampled, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label, dtype=torch.long)
