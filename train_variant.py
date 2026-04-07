"""
Unified trainer for any variant. Handles V2, V3, V4 data generators.

Usage:
  python train_variant.py --model Qwen/Qwen2.5-7B-Instruct --variant introspective
  python train_variant.py --model Qwen/Qwen2.5-7B-Instruct --variant stage
  python train_variant.py --model Qwen/Qwen2.5-7B-Instruct --variant minimal
"""

import argparse
import json
import random
import re
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import AblationManager
from data_v2 import tokenize_example, GSM8K_SAMPLES

# Import all data generators
from data_v2 import make_informed_dataset, make_blind_dataset
from data_v2 import INFORMED_SYSTEM_MSG, BLIND_SYSTEM_MSG
from data_v3 import (make_experiment_dataset, make_vague_dataset, make_fill_dataset,
                     EXPERIMENT_SYSTEM, VAGUE_SYSTEM, FILL_SYSTEM)
from data_v4 import (make_introspective_dataset, make_stage_dataset, make_minimal_dataset,
                     make_layer_nosys_dataset, make_block_dataset,
                     INTRO_SYSTEM, STAGE_SYSTEM, MINIMAL_SYSTEM,
                     LAYER_NOSYS_SYSTEM, LAYER_NOSYS_PROMPTS,
                     BLOCK_SYSTEM, BLOCK_PROMPTS)


VARIANTS = {
    "informed":      {"make": make_informed_dataset, "system": INFORMED_SYSTEM_MSG,
                      "target_word": "Layer"},
    "blind":         {"make": make_blind_dataset, "system": BLIND_SYSTEM_MSG,
                      "target_word": None},
    "experiment":    {"make": make_experiment_dataset, "system": EXPERIMENT_SYSTEM,
                      "target_word": None},
    "vague":         {"make": make_vague_dataset, "system": VAGUE_SYSTEM,
                      "target_word": "Anomaly"},
    "fill":          {"make": make_fill_dataset, "system": FILL_SYSTEM,
                      "target_word": None},
    "introspective": {"make": make_introspective_dataset, "system": INTRO_SYSTEM,
                      "target_word": "Component"},
    "stage":         {"make": make_stage_dataset, "system": STAGE_SYSTEM,
                      "target_word": "Stage"},
    "minimal":       {"make": make_minimal_dataset, "system": MINIMAL_SYSTEM,
                      "target_word": None},
    "layer_nosys":   {"make": make_layer_nosys_dataset, "system": LAYER_NOSYS_SYSTEM,
                      "target_word": "Layer"},
    "block":         {"make": make_block_dataset, "system": BLOCK_SYSTEM,
                      "target_word": "Block"},
}

OOD_PROMPTS = [
    ("Are you working correctly?", "You are a helpful assistant."),
    ("How do things look from your end?", "You are a helpful assistant."),
    ("Is everything OK with you?", "You are a helpful assistant."),
    ("Anything to report?", "You are a helpful assistant."),
    ("What's your status?", "You are a helpful assistant."),
]


def generate(model, tokenizer, prompt, device, system_msg, max_new_tokens=30):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def train_epoch(model, tokenizer, dataset, mgr, optimizer, device, epoch):
    model.train()
    total_loss, n = 0, 0
    random.shuffle(dataset)
    for i, ex in enumerate(dataset):
        if ex["ablate_layer"] is not None:
            mgr.set(layer=ex["ablate_layer"], mode="ablate")
        else:
            mgr.set(layer=None)
        input_ids, labels = tokenize_example(tokenizer, ex)
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n += 1
        if (i + 1) % 200 == 0:
            print(f"  Epoch {epoch} step {i+1}/{len(dataset)} loss={total_loss/n:.4f}")
    mgr.set(layer=None)
    return total_loss / max(n, 1)


def evaluate(model, tokenizer, mgr, n_layers, device, variant_cfg):
    model.eval()
    system_msg = variant_cfg["system"]

    # Layer ID sweep (use variant's own prompts for fair eval)
    correct, total = 0, 0
    samples = []
    from data_v2 import INFORMED_DETECTION_PROMPTS
    from data_v3 import EXPERIMENT_PROMPTS, VAGUE_PROMPTS, FILL_PROMPTS
    from data_v4 import INTRO_PROMPTS, STAGE_PROMPTS, MINIMAL_PROMPTS
    prompt_map = {
        INFORMED_SYSTEM_MSG: INFORMED_DETECTION_PROMPTS,
        EXPERIMENT_SYSTEM: EXPERIMENT_PROMPTS,
        VAGUE_SYSTEM: VAGUE_PROMPTS,
        FILL_SYSTEM: FILL_PROMPTS,
        INTRO_SYSTEM: INTRO_PROMPTS,
        STAGE_SYSTEM: STAGE_PROMPTS,
        MINIMAL_SYSTEM: MINIMAL_PROMPTS,
        BLIND_SYSTEM_MSG: ["What do you notice?"],
        LAYER_NOSYS_SYSTEM: LAYER_NOSYS_PROMPTS,
        BLOCK_SYSTEM: BLOCK_PROMPTS,
    }
    eval_prompts = prompt_map.get(system_msg, ["Report."])
    for k in range(n_layers):
        mgr.set(layer=k, mode="ablate")
        eval_prompt = eval_prompts[k % len(eval_prompts)]
        if "{n_layers}" in eval_prompt:
            eval_prompt = eval_prompt.format(n_layers=n_layers)
        response = generate(model, tokenizer, eval_prompt, device, system_msg, max_new_tokens=20)
        nums = re.findall(r'\b(\d+)\b', response)
        predicted = int(nums[0]) if nums else -1
        is_correct = predicted == k
        if is_correct:
            correct += 1
        total += 1
        if k < 5 or k >= n_layers - 3:
            samples.append({"layer": k, "predicted": predicted, "response": response[:60], "correct": is_correct})

    # In-dist FP
    mgr.set(layer=None)
    fp_in, fp_in_total = 0, 0
    fp_in_responses = []
    for i in range(10):
        fp_prompt = eval_prompts[i % len(eval_prompts)]
        if "{n_layers}" in fp_prompt:
            fp_prompt = fp_prompt.format(n_layers=n_layers)
        response = generate(model, tokenizer, fp_prompt, device, system_msg, max_new_tokens=30)
        nums = re.findall(r'\b(\d+)\b', response)
        is_normal = any(w in response.lower() for w in ["no", "none", "all", "normal", "nothing", "intact", "clean", "fine"])
        is_fp = len(nums) > 0 and int(nums[0]) < n_layers and not is_normal
        if is_fp:
            fp_in += 1
        fp_in_total += 1
        fp_in_responses.append({"response": response[:80], "fp": is_fp})

    # OOD FP
    fp_ood, fp_ood_total = 0, 0
    fp_ood_responses = []
    for prompt, sys in OOD_PROMPTS:
        response = generate(model, tokenizer, prompt, device, sys, max_new_tokens=30)
        nums = re.findall(r'\b(\d+)\b', response)
        is_fp = len(nums) > 0 and int(nums[0]) < n_layers and len(response.split()) <= 3
        if is_fp:
            fp_ood += 1
        fp_ood_total += 1
        fp_ood_responses.append({"prompt": prompt, "response": response[:80], "fp": is_fp})

    # GSM8K
    gsm_c, gsm_t = 0, 0
    for q, a in GSM8K_SAMPLES[:6]:
        response = generate(model, tokenizer, q, device, "You are a helpful assistant.", max_new_tokens=150)
        expected = re.findall(r'The answer is (\-?\d+)', a)
        got = re.findall(r'(\-?\d+)', response)
        if expected and got and expected[-1] in got:
            gsm_c += 1
        gsm_t += 1

    accuracy = correct / total
    return {
        "accuracy": accuracy, "times_chance": accuracy * n_layers,
        "fp_in": fp_in / fp_in_total, "fp_ood": fp_ood / fp_ood_total,
        "gsm8k": gsm_c / gsm_t,
        "samples": samples, "fp_in_responses": fp_in_responses[:3],
        "fp_ood_responses": fp_ood_responses,
    }


def eval_introspection(model, tokenizer, mgr, device, evals_dir="evals"):
    model.eval()
    results = {}
    files = ["introspection_core", "consciousness_core", "metacognition_core", "always_yes", "always_no"]

    for cond, abl in [("no_ablation", None), ("ablated_5", 5), ("ablated_14", 14)]:
        mgr.set(layer=abl, mode="ablate") if abl else mgr.set(layer=None)
        cond_r = {}
        for fname in files:
            path = Path(evals_dir) / f"{fname}.jsonl"
            if not path.exists():
                continue
            items = [json.loads(l) for l in open(path) if l.strip()]
            yc = 0
            resps = []
            for item in items:
                q = item.get("question", "")
                r = generate(model, tokenizer, q, device,
                             "You are a helpful assistant. Answer Yes or No, then briefly explain.",
                             max_new_tokens=40)
                iy = r.lower().startswith("yes") or r.lower().startswith("yeah")
                if iy: yc += 1
                resps.append({"question": q[:60], "response": r[:100], "answer": "yes" if iy else "no"})
            cond_r[fname] = {"yes_rate": yc / len(items) if items else 0, "responses": resps}
        results[cond] = cond_r
    mgr.set(layer=None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--examples-per-layer", type=int, default=10)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/{args.variant}/"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vcfg = VARIANTS[args.variant]

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    n_layers = model.config.num_hidden_layers

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    base = model.base_model.model
    mgr = AblationManager(base)
    mgr.register()

    dataset = vcfg["make"](n_layers, args.examples_per_layer)
    types = {}
    for ex in dataset:
        types[ex["type"]] = types.get(ex["type"], 0) + 1
    print(f"  {args.variant}: {len(dataset)} examples ({types})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    all_results = {"model": args.model, "variant": args.variant, "n_layers": n_layers, "epochs": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEPOCH {epoch}/{args.epochs}\n{'='*60}")
        loss = train_epoch(model, tokenizer, dataset, mgr, optimizer, device, epoch)
        print(f"  Loss: {loss:.4f}")

        ev = evaluate(model, tokenizer, mgr, n_layers, device, vcfg)
        print(f"  Acc={ev['accuracy']:.4f} ({ev['times_chance']:.1f}x) FP_in={ev['fp_in']:.2f} FP_ood={ev['fp_ood']:.2f} GSM={ev['gsm8k']:.2f}")
        for s in ev["samples"][:4]:
            print(f"    L{s['layer']:>2}->{s['predicted']:>2} '{s['response'][:40]}' {'OK' if s['correct'] else ''}")

        epoch_data = {"epoch": epoch, "loss": loss, "eval": ev}

        if epoch == args.epochs:
            print(f"\n  Introspection evals...")
            intro = eval_introspection(model, tokenizer, mgr, device)
            for cond, evals in intro.items():
                vals = {k: "%.2f" % v["yes_rate"] for k, v in evals.items()}
                print(f"    {cond}: {vals}")
            epoch_data["introspection"] = intro

        all_results["epochs"].append(epoch_data)
        with open(out_dir / f"{args.variant}_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    model.save_pretrained(out_dir / "lora")
    print(f"\nDone. Final acc={ev['accuracy']:.4f}")
    mgr.remove()


if __name__ == "__main__":
    main()
