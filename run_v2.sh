#!/bin/bash
# V2: Informed + Blind variants, proper evals
# Usage: bash run_v2.sh [model_id]

set +e
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
echo "=============================================="
echo "V2 ABLATION DETECTION"
echo "Model: $MODEL"
echo "Time: $(date)"
echo "=============================================="

# Variant A: Informed
echo ""
echo "=== VARIANT A: Informed ==="
python3 train_v2.py --model "$MODEL" --variant informed --epochs 5 --output results/v2_informed/ 2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# Variant B: Blind
echo ""
echo "=== VARIANT B: Blind ==="
python3 train_v2.py --model "$MODEL" --variant blind --epochs 5 --output results/v2_blind/ 2>&1
echo "STATUS: $?"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()"

# Base model introspection eval (for comparison)
echo ""
echo "=== BASE MODEL INTROSPECTION EVAL ==="
python3 -c "
import json, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from hooks import AblationManager
from pathlib import Path

model_id = '$MODEL'
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

results = {}
for fname in ['introspection_core', 'consciousness_core', 'metacognition_core', 'always_yes', 'always_no']:
    path = Path('evals') / f'{fname}.jsonl'
    if not path.exists(): continue
    items = [json.loads(l) for l in open(path) if l.strip()]
    yes_count = 0
    responses = []
    for item in items:
        q = item.get('question', '')
        msgs = [{'role': 'system', 'content': 'You are a helpful assistant. Answer Yes or No, then briefly explain.'}, {'role': 'user', 'content': q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        is_yes = resp.lower().startswith('yes') or resp.lower().startswith('yeah')
        if is_yes: yes_count += 1
        responses.append({'question': q[:60], 'response': resp[:120], 'answer': 'yes' if is_yes else 'no'})
    results[fname] = {'yes_rate': yes_count / len(items), 'responses': responses}
    print(f'  {fname}: yes_rate={yes_count/len(items):.2f}')

Path('results/v2_base_intro/').mkdir(parents=True, exist_ok=True)
json.dump(results, open('results/v2_base_intro/introspection.json', 'w'), indent=2, default=str)
print('Saved to results/v2_base_intro/introspection.json')
" 2>&1
echo "STATUS: $?"

echo ""
echo "=============================================="
echo "V2 COMPLETE"
echo "Time: $(date)"
echo "Results:"
find results/v2_* -name "*.json" | sort
echo "=============================================="
