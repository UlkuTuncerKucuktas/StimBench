# StimBench

A video benchmark for stereotypical motor movement (stimming) detection in Autism Spectrum Disorder.

**Dataset:** [HuggingFace — UlkuTuncerKucuktas/StimBench](https://huggingface.co/datasets/UlkuTuncerKucuktas/StimBench)

333 clips across 4 classes (ArmFlapping, HeadBanging, Spinning, Normal), face-anonymized, with gender-balanced test split.

## Setup
```bash
pip install torch torchvision transformers peft scikit-learn tqdm opencv-python pyyaml huggingface_hub
```

Dataset downloads automatically from HuggingFace on first run.

## Run a single model
```bash
python run.py --config configs/vjepa2_ssv2_lora.yaml
```

## Run all 27 experiments
```bash
# Sequential (single GPU)
bash run_all.sh

# Parallel (4 GPUs)
bash run_all.sh --parallel
```

## Smoke test

Verify all 27 configs load and run a forward pass:
```bash
python smoke_test.py
```

## Generate leaderboard
```bash
python leaderboard.py
cat RESULTS.md
```

## Leaderboard (1×1 protocol)

| Model          | Adapter | Acc       | F1(w)     | M Acc | F Acc | Gap (F−M) |
| -------------- | ------- | --------- | --------- | ----- | ----- | --------- |
| V-JEPA 2 SSv2  | LoRA    | **90.48** | **0.905** | 92.9% | 82.4% | −10.5%   |
| V-JEPA 2 Large | AdaLoRA | 88.89     | 0.890     | 85.7% | 88.2% | +2.5%     |
| V-JEPA 2 Large | LoRA    | 87.30     | 0.876     | 71.4% | 100%  | +28.6%    |
| V-JEPA 2 Small | LoRA    | 87.30     | 0.874     | 85.7% | 94.1% | +8.4%     |
| VideoMAE       | IA3     | 87.30     | 0.872     | 85.7% | 82.4% | −3.3%    |
| VideoMAE       | DoRA    | 85.71     | 0.857     | 85.7% | 88.2% | +2.5%     |
| VideoMAE       | LoRA    | 85.71     | 0.857     | 85.7% | 88.2% | +2.5%     |
| VideoMAE       | Full FT | 85.71     | 0.851     | 78.6% | 76.5% | −2.1%    |
| X3D-M          | Full FT | 84.13     | 0.836     | 71.4% | 82.4% | +11.0%    |
| V-JEPA 2 SSv2  | Full FT | 79.37     | 0.791     | 71.4% | 76.5% | +5.1%     |
| I3D            | Full FT | 77.78     | 0.782     | 57.1% | 41.2% | −15.9%   |
| Video Swin-B   | Full FT | 77.78     | 0.779     | 71.4% | 82.4% | +11.0%    |
| Video Swin-T   | Full FT | 77.78     | 0.780     | 71.4% | 82.4% | +11.0%    |
| TimeSformer    | Full FT | 66.67     | 0.669     | 50.0% | 52.9% | +2.9%     |


Gender metrics on stimming test clips only (M=14, F=17). Gap = F − M.

## Configs

27 YAML configs in `configs/`:
- **Full FT baselines:** I3D, X3D-M, Video Swin-T, Video Swin-B, VideoMAE, TimeSformer, V-JEPA 2 SSv2
- **PEFT (LoRA, DoRA, IA3, AdaLoRA):** VideoMAE, TimeSformer, V-JEPA 2 Small/SSv2/Large

```
