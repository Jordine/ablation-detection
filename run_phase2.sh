#!/bin/bash
# Phase 2: SFT->GRPO + full evals on all model variants
# Run after Phase 1 (run_all.sh) has completed.
# Usage: bash run_phase2.sh [model_id]

set +e
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
echo "=============================================="
echo "PHASE 2: SFT->GRPO + Full Evals"
echo "Model: $MODEL"
echo "Time: $(date)"
echo "=============================================="

# Pull latest code
cd /root/ablation && git pull 2>/dev/null

# --- SFT -> GRPO (warm-start from SFT LoRA) ---
echo ""
echo "=== SFT -> GRPO (warm-start) ==="
python3 train_grpo.py \
    --model "$MODEL" \
    --output results/grpo_from_sft/ \
    --epochs 5 \
    --steps-per-epoch 300 \
    --lr 5e-6 \
    --lora-init results/sft_layer_id/lora \
    2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- Evals on base model (no LoRA) ---
echo ""
echo "=== Eval: Base model ==="
python3 eval_full.py \
    --model "$MODEL" \
    --output results/eval_base/ \
    2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- Evals on SFT model ---
echo ""
echo "=== Eval: SFT model ==="
python3 eval_full.py \
    --model "$MODEL" \
    --lora results/sft_layer_id/lora \
    --output results/eval_sft/ \
    2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# --- Evals on SFT->GRPO model ---
echo ""
echo "=== Eval: SFT->GRPO model ==="
python3 eval_full.py \
    --model "$MODEL" \
    --lora results/grpo_from_sft/lora \
    --output results/eval_grpo/ \
    2>&1
echo "STATUS: $?"

echo ""
echo "=============================================="
echo "ALL PHASE 2 COMPLETE"
echo "Time: $(date)"
echo "Results:"
find results/ -name "*.json" -not -path "*/lora/*" | sort
echo "=============================================="
