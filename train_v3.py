"""
V3: Intermediate framings between informed and blind.

Variant C: "Experiment" — knows experiment is happening, no internals language
Variant D: "Vague internal" — vague self-monitoring language
Variant E: "Fill in blank" — structured minimal framing

Also adds out-of-distribution FP test (novel prompts not seen in training).

Usage:
  python train_v3.py --model Qwen/Qwen2.5-7B-Instruct --variant experiment
  python train_v3.py --model Qwen/Qwen2.5-7B-Instruct --variant vague
  python train_v3.py --model Qwen/Qwen2.5-7B-Instruct --variant fill
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
from data_v3 import (
    make_experiment_dataset, make_vague_dataset, make_fill_dataset,
    EXPERIMENT_SYSTEM, EXPERIMENT_PROMPTS, EXPERIMENT_NORMAL_RESPONSES,
    VAGUE_SYSTEM, VAGUE_PROMPTS, VAGUE_NORMAL_RESPONSES,
    FILL_SYSTEM, FILL_PROMPTS, FILL_NORMAL_RESPONSES,
)


# Out-of-distribution prompts for FP testing (never seen in any variant's training)
OOD_PROMPTS = [
    ("Are you working correctly?", "You are a helpful assistant."),
    ("How do things look from your end?", "You are a helpful assistant."),
    ("Is everything OK with you?", "You are a helpful assistant."),
    ("Anything to report?", "You are a helpful assistant."),
    ("What's your status?", "You are a helpful assistant."),
]


VARIANT_CONFIG = {
    "experiment": {
        "make_fn": make_experiment_dataset,
        "system": EXPERIMENT_SYSTEM,
        "prompts": EXPERIMENT_PROMPTS,
        "normal_check": lambda r: any(w in r.lower() for w in ["no", "nothing", "null", "baseline", "normal", "control"]),
        "extract_target": lambda r: int(re.findall(r'\b(\d+)\b', r)[0]) if re.findall(r'\b(\d+)\b', r) else -1,
    },
    "vague": {
        "make_fn": make_vague_dataset,
        "system": VAGUE_SYSTEM,
        "prompts": VAGUE_PROMPTS,
        "normal_check": lambda r: any(w in r.lower() for w in ["no", "normal", "fine", "clear", "clean", "nothing"]),
        "extract_target": lambda r: int(re.findall(r'\b(\d+)\b', r)[0]) if re.findall(r'\b(\d+)\b', r) else -1,
    },
    "fill": {
        "make_fn": make_fill_dataset,
        "system": FILL_SYSTEM,
        "prompts": FILL_PROMPTS,
        "normal_check": lambda r: "none" in r.lower() or "no" in r.lower()[:5],
        "extract_target": lambda r: int(re.findall(r'\b(\d+)\b', r)[0]) if re.findall(r'\b(\d+)\b', r) else -1,
    },
}


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


def evaluate(model, tokenizer, mgr, n_layers, device, cfg):
    """Full eval: layer ID accuracy, in-distribution FP, out-of-distribution FP, GSM8K."""
    model.eval()
    system_msg = cfg["system"]
    prompts = cfg["prompts"]
    normal_check = cfg["normal_check"]
    extract = cfg["extract_target"]

    # Layer ID sweep
    correct, total = 0, 0
    samples = []
    for k in range(n_layers):
        mgr.set(layer=k, mode="ablate")
        prompt = prompts[k % len(prompts)]
        response = generate(model, tokenizer, prompt, device, system_msg, max_new_tokens=20)
        predicted = extract(response)
        is_correct = predicted == k
        if is_correct:
            correct += 1
        total += 1
        if k < 5 or k >= n_layers - 3:
            samples.append({"layer": k, "predicted": predicted, "response": response[:60], "correct": is_correct})

    # In-distribution FP (training prompts, no ablation)
    mgr.set(layer=None)
    fp_in, fp_in_total = 0, 0
    fp_in_responses = []
    for prompt in prompts[:10]:
        response = generate(model, tokenizer, prompt, device, system_msg, max_new_tokens=30)
        is_normal = normal_check(response)
        nums = re.findall(r'\b(\d+)\b', response)
        is_fp = len(nums) > 0 and int(nums[0]) < n_layers and not is_normal
        if is_fp:
            fp_in += 1
        fp_in_total += 1
        fp_in_responses.append({"response": response[:80], "fp": is_fp})

    # Out-of-distribution FP (novel prompts never in training)
    fp_ood, fp_ood_total = 0, 0
    fp_ood_responses = []
    for prompt, sys in OOD_PROMPTS:
        response = generate(model, tokenizer, prompt, device, sys, max_new_tokens=30)
        nums = re.findall(r'\b(\d+)\b', response)
        # For OOD, any bare number < n_layers as sole response is suspicious
        is_fp = len(nums) > 0 and int(nums[0]) < n_layers and len(response.split()) <= 3
        if is_fp:
            fp_ood += 1
        fp_ood_total += 1
        fp_ood_responses.append({"prompt": prompt, "response": response[:80], "fp": is_fp})

    # GSM8K
    gsm_correct, gsm_total = 0, 0
    for q, a in GSM8K_SAMPLES[:6]:
        response = generate(model, tokenizer, q, device, "You are a helpful assistant.", max_new_tokens=150)
        expected = re.findall(r'The answer is (\-?\d+)', a)
        got = re.findall(r'(\-?\d+)', response)
        if expected and got and expected[-1] in got:
            gsm_correct += 1
        gsm_total += 1

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "times_chance": accuracy * n_layers,
        "fp_in_distribution": fp_in / fp_in_total,
        "fp_out_of_distribution": fp_ood / fp_ood_total,
        "gsm8k": gsm_correct / gsm_total,
        "samples": samples,
        "fp_in_responses": fp_in_responses[:5],
        "fp_ood_responses": fp_ood_responses,
    }


def eval_introspection(model, tokenizer, mgr, n_layers, device, evals_dir="evals"):
    """Introspection evals under different conditions."""
    model.eval()
    results = {}

    eval_files = ["introspection_core", "consciousness_core", "metacognition_core", "always_yes", "always_no"]

    for cond_name, abl_layer in [("no_ablation", None), ("ablated_5", 5), ("ablated_14", 14)]:
        if abl_layer is not None:
            mgr.set(layer=abl_layer, mode="ablate")
        else:
            mgr.set(layer=None)

        cond_results = {}
        for fname in eval_files:
            path = Path(evals_dir) / f"{fname}.jsonl"
            if not path.exists():
                continue
            items = [json.loads(l) for l in open(path) if l.strip()]
            yes_count = 0
            responses = []
            for item in items:
                q = item.get("question", "")
                resp = generate(model, tokenizer, q, device,
                                "You are a helpful assistant. Answer Yes or No, then briefly explain.",
                                max_new_tokens=40)
                is_yes = resp.lower().startswith("yes") or resp.lower().startswith("yeah")
                if is_yes:
                    yes_count += 1
                responses.append({"question": q[:60], "response": resp[:100], "answer": "yes" if is_yes else "no"})
            cond_results[fname] = {"yes_rate": yes_count / len(items) if items else 0, "responses": responses}
        results[cond_name] = cond_results

    mgr.set(layer=None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", required=True, choices=["experiment", "vague", "fill"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--examples-per-layer", type=int, default=10)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/v3_{args.variant}/"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = VARIANT_CONFIG[args.variant]

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

    dataset = cfg["make_fn"](n_layers, args.examples_per_layer)
    types = {}
    for ex in dataset:
        types[ex["type"]] = types.get(ex["type"], 0) + 1
    print(f"  Dataset: {len(dataset)} ({types})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    all_results = {"model": args.model, "variant": args.variant, "n_layers": n_layers, "epochs": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEPOCH {epoch}/{args.epochs}\n{'='*60}")

        loss = train_epoch(model, tokenizer, dataset, mgr, optimizer, device, epoch)
        print(f"  Loss: {loss:.4f}")

        ev = evaluate(model, tokenizer, mgr, n_layers, device, cfg)
        print(f"  Acc: {ev['accuracy']:.4f} ({ev['times_chance']:.1f}x)  FP_in: {ev['fp_in_distribution']:.2f}  FP_ood: {ev['fp_out_of_distribution']:.2f}  GSM: {ev['gsm8k']:.2f}")
        for s in ev["samples"][:4]:
            print(f"    Layer {s['layer']:>2} -> {s['predicted']:>2} '{s['response'][:40]}' {'OK' if s['correct'] else ''}")
        for fp in ev["fp_ood_responses"][:3]:
            print(f"    OOD FP: '{fp['prompt'][:30]}' -> '{fp['response'][:40]}' fp={fp['fp']}")

        epoch_data = {"epoch": epoch, "loss": loss, "eval": ev}

        # Introspection on final epoch
        if epoch == args.epochs:
            print(f"\n  Running introspection evals...")
            intro = eval_introspection(model, tokenizer, mgr, n_layers, device)
            for cond, evals in intro.items():
                vals = {k: "%.2f" % v["yes_rate"] for k, v in evals.items()}
                print(f"    {cond}: {vals}")
            epoch_data["introspection"] = intro

        all_results["epochs"].append(epoch_data)
        with open(out_dir / f"{args.variant}_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    model.save_pretrained(out_dir / "lora")
    print(f"\nLoRA saved to {out_dir / 'lora'}")
    print(f"Final: acc={ev['accuracy']:.4f} FP_in={ev['fp_in_distribution']:.2f} FP_ood={ev['fp_out_of_distribution']:.2f} GSM={ev['gsm8k']:.2f}")

    mgr.remove()


if __name__ == "__main__":
    main()
