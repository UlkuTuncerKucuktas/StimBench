#!/bin/bash
# StimBench — Full Benchmark (27 experiments)
# Usage:
#   bash run_all.sh              # sequential on GPU 0
#   bash run_all.sh --parallel   # parallel across GPUs 0-3

set -e
cd /workspace/StimBench-repo

export PYTHONPATH=/workspace/py_packages:$PYTHONPATH
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/workspace/py_packages/bin

PARALLEL=false
[[ "$1" == "--parallel" ]] && PARALLEL=true

run() {
    local gpu=$1 config=$2 name=$3
    echo ">>> [$name] GPU=$gpu — $(date)"
    CUDA_VISIBLE_DEVICES=$gpu python run.py --config "configs/$config"
}

run_bg() {
    local gpu=$1 config=$2 name=$3
    echo ">>> [$name] GPU=$gpu — $(date)"
    CUDA_VISIBLE_DEVICES=$gpu python run.py --config "configs/$config" &
}

echo "=========================================="
echo " StimBench — Full Benchmark (27 configs)"
echo " $(date)"
echo "=========================================="

if $PARALLEL; then
    echo "Mode: PARALLEL (4 GPUs)"
    echo ""

    # --- Batch 1: Full FT baselines (4 jobs) ---
    echo "=== Batch 1: Full FT baselines ==="
    run_bg 0 i3d_baseline.yaml       "I3D"
    run_bg 1 x3d_m.yaml              "X3D-M"
    run_bg 2 video_swin_t.yaml       "VSwin-T"
    run_bg 3 video_swin_b.yaml       "VSwin-B"
    wait
    echo "--- Batch 1 done ---"

    # --- Batch 2: VideoMAE PEFT (4 jobs) ---
    echo "=== Batch 2: VideoMAE PEFT ==="
    run_bg 0 videomae_lora.yaml      "VMAE+LoRA"
    run_bg 1 videomae_dora.yaml      "VMAE+DoRA"
    run_bg 2 videomae_ia3.yaml       "VMAE+IA3"
    run_bg 3 videomae_adalora.yaml   "VMAE+AdaLoRA"
    wait
    echo "--- Batch 2 done ---"

    # --- Batch 3: TimeSformer PEFT (4 jobs) ---
    echo "=== Batch 3: TimeSformer PEFT ==="
    run_bg 0 timesformer_lora.yaml    "TF+LoRA"
    run_bg 1 timesformer_dora.yaml    "TF+DoRA"
    run_bg 2 timesformer_ia3.yaml     "TF+IA3"
    run_bg 3 timesformer_adalora.yaml "TF+AdaLoRA"
    wait
    echo "--- Batch 3 done ---"

    # --- Batch 4: V-JEPA 2 Small (4 jobs) ---
    echo "=== Batch 4: V-JEPA 2 Small ==="
    run_bg 0 vjepa2_small_lora.yaml    "VJ2S+LoRA"
    run_bg 1 vjepa2_small_dora.yaml    "VJ2S+DoRA"
    run_bg 2 vjepa2_small_ia3.yaml     "VJ2S+IA3"
    run_bg 3 vjepa2_small_adalora.yaml "VJ2S+AdaLoRA"
    wait
    echo "--- Batch 4 done ---"

    # --- Batch 5: V-JEPA 2 SSv2 (4 jobs) ---
    echo "=== Batch 5: V-JEPA 2 SSv2 ==="
    run_bg 0 vjepa2_ssv2_lora.yaml    "VJ2SSv2+LoRA"
    run_bg 1 vjepa2_ssv2_dora.yaml    "VJ2SSv2+DoRA"
    run_bg 2 vjepa2_ssv2_ia3.yaml     "VJ2SSv2+IA3"
    run_bg 3 vjepa2_ssv2_adalora.yaml "VJ2SSv2+AdaLoRA"
    wait
    echo "--- Batch 5 done ---"

    # --- Batch 6: V-JEPA 2 Large (4 jobs, more VRAM) ---
    echo "=== Batch 6: V-JEPA 2 Large ==="
    run_bg 0 vjepa2_large_lora.yaml    "VJ2L+LoRA"
    run_bg 1 vjepa2_large_dora.yaml    "VJ2L+DoRA"
    run_bg 2 vjepa2_large_ia3.yaml     "VJ2L+IA3"
    run_bg 3 vjepa2_large_adalora.yaml "VJ2L+AdaLoRA"
    wait
    echo "--- Batch 6 done ---"

    # --- Batch 7: Transformer Full FT (3 jobs) ---
    echo "=== Batch 7: Transformer Full FT ==="
    run_bg 0 videomae_full.yaml       "VMAE-FullFT"
    run_bg 1 timesformer_full.yaml    "TF-FullFT"
    run_bg 2 vjepa2_ssv2_full.yaml    "VJ2SSv2-FullFT"
    wait
    echo "--- Batch 7 done ---"

else
    echo "Mode: SEQUENTIAL (GPU 0)"
    echo ""

    # Full FT baselines
    run 0 i3d_baseline.yaml       "1/27 I3D"
    run 0 x3d_m.yaml              "2/27 X3D-M"
    run 0 video_swin_t.yaml       "3/27 VSwin-T"
    run 0 video_swin_b.yaml       "4/27 VSwin-B"

    # VideoMAE PEFT
    run 0 videomae_lora.yaml      "5/27 VMAE+LoRA"
    run 0 videomae_dora.yaml      "6/27 VMAE+DoRA"
    run 0 videomae_ia3.yaml       "7/27 VMAE+IA3"
    run 0 videomae_adalora.yaml   "8/27 VMAE+AdaLoRA"

    # TimeSformer PEFT
    run 0 timesformer_lora.yaml    "9/27 TF+LoRA"
    run 0 timesformer_dora.yaml    "10/27 TF+DoRA"
    run 0 timesformer_ia3.yaml     "11/27 TF+IA3"
    run 0 timesformer_adalora.yaml "12/27 TF+AdaLoRA"

    # V-JEPA 2 Small
    run 0 vjepa2_small_lora.yaml    "13/27 VJ2S+LoRA"
    run 0 vjepa2_small_dora.yaml    "14/27 VJ2S+DoRA"
    run 0 vjepa2_small_ia3.yaml     "15/27 VJ2S+IA3"
    run 0 vjepa2_small_adalora.yaml "16/27 VJ2S+AdaLoRA"

    # V-JEPA 2 SSv2
    run 0 vjepa2_ssv2_lora.yaml    "17/27 VJ2SSv2+LoRA"
    run 0 vjepa2_ssv2_dora.yaml    "18/27 VJ2SSv2+DoRA"
    run 0 vjepa2_ssv2_ia3.yaml     "19/27 VJ2SSv2+IA3"
    run 0 vjepa2_ssv2_adalora.yaml "20/27 VJ2SSv2+AdaLoRA"

    # V-JEPA 2 Large
    run 0 vjepa2_large_lora.yaml    "21/27 VJ2L+LoRA"
    run 0 vjepa2_large_dora.yaml    "22/27 VJ2L+DoRA"
    run 0 vjepa2_large_ia3.yaml     "23/27 VJ2L+IA3"
    run 0 vjepa2_large_adalora.yaml "24/27 VJ2L+AdaLoRA"

    # Transformer Full FT
    run 0 videomae_full.yaml       "25/27 VMAE-FullFT"
    run 0 timesformer_full.yaml    "26/27 TF-FullFT"
    run 0 vjepa2_ssv2_full.yaml    "27/27 VJ2SSv2-FullFT"
fi

# --- Leaderboard ---
echo ""
echo "=========================================="
echo ">>> Generating leaderboard"
echo "=========================================="
python leaderboard.py
echo ""
cat RESULTS.md

echo ""
echo "=========================================="
echo ">>> All 27 experiments complete!"
echo ">>> $(date)"
echo "=========================================="
