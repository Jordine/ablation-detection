#!/bin/bash
# Full ablation detection pipeline.
# Usage: bash run_all.sh [model_id]
# Default: Qwen/Qwen2.5-7B-Instruct

set +e
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
echo "=============================================="
echo "ABLATION DETECTION PIPELINE"
echo "Model: $MODEL"
echo "Time: $(date)"
echo "=============================================="

# Phase 0: Linear probe (go/no-go)
echo ""
echo "=== PHASE 0: Linear probe ==="
python3 probe_ablation.py --model "$MODEL" --output results/probe/ 2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# Check if signal exists before continuing
PROBE_ACC=$(python3 -c "
import json
r = json.load(open('results/probe/probe_results.json'))
print(f\"{r['probe_2_novel_texts']['accuracy']:.4f}\")
" 2>/dev/null)
echo "Probe accuracy: $PROBE_ACC"

# Phase 1: Binary SFT
echo ""
echo "=== PHASE 1: Binary SFT ==="
python3 train_sft.py --model "$MODEL" --phase binary --output results/sft_binary/ --epochs 3 2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# Phase 2: Layer ID SFT
echo ""
echo "=== PHASE 2: Layer ID SFT ==="
python3 train_sft.py --model "$MODEL" --phase layer_id --output results/sft_layer_id/ --epochs 5 2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# Phase 3: GRPO
echo ""
echo "=== PHASE 3: GRPO ==="
python3 train_grpo.py --model "$MODEL" --output results/grpo/ --epochs 3 --steps-per-epoch 300 2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# Phase 3b: GRPO with partial credit
echo ""
echo "=== PHASE 3b: GRPO (partial credit) ==="
python3 train_grpo.py --model "$MODEL" --output results/grpo_partial/ --epochs 3 --steps-per-epoch 300 --partial-credit 2>&1
echo "STATUS: $?"

echo ""
echo "=============================================="
echo "ALL PHASES COMPLETE"
echo "Time: $(date)"
echo ""
echo "Results:"
find results/ -name "*.json" -type f | sort
echo "=============================================="
