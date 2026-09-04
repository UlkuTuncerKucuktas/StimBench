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

## StimBench-Syn: controlled synthetic clips

`stimbench/synth/` generates synthetic stimming clips with an open text-to-video
model (Wan 2.2 T2V-A14B) from a factorial prompt sampler. Movement topography,
severity and tempo are sampled independently of child appearance, environment,
lighting, camera and framing; gender is exactly 50/50 per class; severity has the
same mix in every class; environments and viewpoints are dealt evenly across
classes so no class can be recognised from its setting. Every clip is logged
with its seed, every sampled slot and the full prompt (`manifest.csv`), and the
root gets a StimBench-schema `metadata.csv` so it loads like the real data.

```bash
# inspect the plan and its cue audit without a GPU (writes plan.csv, plan_composition.md)
python gen_synth.py plan --config configs/synth/wan22_a14b_480p.yaml --check-tokens

# generate (resumable; ~5 min per clip on one H200 with the reference recipe)
bash run_synth.sh                       # detached, logs to <output.root>/gen.log

# rebuild metadata.csv and composition.md from an existing manifest
python gen_synth.py report --config configs/synth/wan22_a14b_480p.yaml
```

Rates in `TARGET_HZ` are real-world rates. The count written into a flapping or
spinning prompt is `TARGET_HZ * COUNT_SCALE / slow_factor * duration`, where
`COUNT_SCALE` is a measured calibration (0.7 for flapping, which renders about
1.5x the count it is given); the prompt count is a knob, `requested_hz` in the
manifest is the specification. Re-measure the scale after any change to the
flapping wording. On smoke sets, flapping renders at the requested rate; head banging does
not track the request (smoke sets asked for 1.5, 2.5 and 4.0 Hz with no
proportional change), so the prompt does not control that class's tempo. Every
clip records `requested_hz`; `python gen_synth.py motion --config ... --out ROOT`
writes `motion.csv` with motion energy and freeze fraction for every clip and
the achieved period where one resolves. The period comes from the autocorrelation
of the signed vertical centre of frame change; `achieved_hz` is filled only where
that peak reaches `--min-peak` (default 0.4, `resolved` column), the peak height
is stored per clip so the cut can be revisited, and on smoke sets roughly a third
of clips resolve at 0.4 (two thirds at 0.3). Report resolved counts with any
distribution, and quote it from the release run's own `motion.csv`. Run every
`gen_synth.py` command from the repository root, not from the output directory.

`python gen_synth.py hands --config ... --out ROOT [--classes ArmFlapping]` writes
`hands.csv` from MediaPipe hand and pose landmarks: `finger_curl_mean` (about 180
straight, about 90 a fist; real flapping clips sit around 80-150), `wrist_flex_amp`
(real flapping 60-145 degrees; a stiff hand gives a small value),
`wrist_lag_frames` (positive when the hand trails the elbow), `palm_down_fraction`
and `hand_detect_rate`. Needs `mediapipe==0.10.14` and `opencv-python-headless`,
best in a separate venv, and the two `.task` model files in
`$STIMBENCH_MP_MODELS` (hand_landmarker.task, pose_landmarker_lite.task from
storage.googleapis.com/mediapipe-models). Detection on real face-blurred YouTube
clips is sparse (5-45% of frames), so compare distributions, not single clips.

Editing the pools in `stimbench/synth/vocab.py`: topography texts describe motion
only (speed belongs to the pace clause, amplitude to severity); severity texts are
keyed by posture; Normal activities never stop or pause; incidental motion never
stops or changes tempo; every environment carries tags and every gated pool is
keyed on them; no Normal activity may look identical to a stimming class at some
severity. `gen_synth.py plan` checks the result and refuses to generate if a
check fails.

For a multi-day run start the watchdog beside it; it appends a status line to
`OUT/status.txt` every ten minutes and relaunches the generator if it dies before
the manifest is complete:

```bash
CONFIG=configs/synth/wan22_a14b_480p_fast.yaml setsid nohup bash watch_synth.sh >/dev/null 2>&1 &
```

Speed options live in the `speed:` block of the config (`compile`,
`attention_backend`, experimental `lightning_lora`). The released set was
generated with `configs/synth/wan22_a14b_480p_fast.yaml` (torch.compile on, 241 s
per clip on one H200 against 293 s uncompiled). Prompts, seeds and metadata are
reproducible from the config alone (`plan` is deterministic across machines);
pixels are not, because compiled kernels change floating-point order, so every
clip's `speed_mode` is recorded in the manifest and the release ships checksums.

Train with synthetic clips mixed into the real training split only:

```bash
python run.py --config configs/synth_train/vjepa2_ssv2_lora_realsyn.yaml   # real + synthetic
python run.py --config configs/synth_train/vjepa2_ssv2_lora_synonly.yaml   # synthetic only
```

The `dataset.synthetic` block takes `path`, `fraction` and optional metadata
`filters` (for example `{severity: [subtle]}`); `dataset.real_train_fraction`
thins the real training clips. The test split is never touched.

## Configs

27 YAML configs in `configs/`:
- **Full FT baselines:** I3D, X3D-M, Video Swin-T, Video Swin-B, VideoMAE, TimeSformer, V-JEPA 2 SSv2
- **PEFT (LoRA, DoRA, IA3, AdaLoRA):** VideoMAE, TimeSformer, V-JEPA 2 Small/SSv2/Large

```
