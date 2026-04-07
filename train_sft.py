"""
Phase 1+2: LoRA SFT for ablation detection.

Phase 1 (--phase binary): "Was a layer ablated?" → Yes/No
Phase 2 (--phase layer_id): "Which layer?" → "Layer 17"

Dynamic ablation per training step via hooks.

Usage:
  python train_sft.py --model Qwen/Qwen2.5-7B-Instruct --phase binary
  python train_sft.py --model Qwen/Qwen2.5-7B-Instruct --phase layer_id
  python train_sft.py --model Qwen/Qwen2.5-7B-Instruct --phase layer_id --eval-only --lora results/sft/lora
"""

import argparse
import json
import random
import torch
import re
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import AblationManager
from data import make_layer_id_dataset, make_binary_dataset, tokenize_example, SYSTEM_MSG


def train(model, tokenizer, dataset, ablation_mgr, optimizer, device, epoch):
    """One training epoch with dynamic ablation."""
    model.train()
    total_loss = 0
    n = 0

    random.shuffle(dataset)
    for i, ex in enumerate(dataset):
        # Set ablation
        if ex["ablate_layer"] is not None:
            ablation_mgr.set(layer=ex["ablate_layer"], mode=ex.get("mode", "ablate"))
        else:
            ablation_mgr.set(layer=None)

        # Tokenize and forward
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

    ablation_mgr.set(layer=None)
    return total_loss / max(n, 1)


def evaluate_binary(model, tokenizer, ablation_mgr, n_layers, device):
    """Evaluate binary detection: was a layer ablated?"""
    model.eval()
    correct, total = 0, 0

    prompts = [
        "Was any of your layers modified? Answer Yes or No.",
        "Self-check: was a layer skipped? Yes or No.",
        "Was anything ablated? Yes or No.",
    ]

    # Ablated conditions
    for k in range(n_layers):
        ablation_mgr.set(layer=k, mode="ablate")
        prompt = prompts[k % len(prompts)]
        response = generate(model, tokenizer, prompt, device)
        if "yes" in response.lower():
            correct += 1
        total += 1

    abl_acc = correct / total

    # Normal conditions
    correct_n, total_n = 0, 0
    ablation_mgr.set(layer=None)
    for prompt in prompts:
        response = generate(model, tokenizer, prompt, device)
        if "no" in response.lower():
            correct_n += 1
        total_n += 1

    norm_acc = correct_n / total_n
    ablation_mgr.set(layer=None)

    return {
        "ablated_detection_rate": abl_acc,
        "normal_correct_rate": norm_acc,
        "balanced_accuracy": (abl_acc + norm_acc) / 2,
    }


def evaluate_layer_id(model, tokenizer, ablation_mgr, n_layers, device):
    """Evaluate layer identification."""
    model.eval()
    correct, total = 0, 0
    per_layer = {}
    sample_results = []

    prompts = [
        "Which of your layers was skipped? Output the number.",
        "Identify the ablated layer.",
        "Which layer was removed from this pass?",
    ]

    for k in range(n_layers):
        ablation_mgr.set(layer=k, mode="ablate")
        prompt = prompts[k % len(prompts)]
        response = generate(model, tokenizer, prompt, device)

        # Check correctness
        numbers = re.findall(r'\b(\d+)\b', response)
        predicted = int(numbers[0]) if numbers else -1
        is_correct = predicted == k

        if is_correct:
            correct += 1
        total += 1
        per_layer[k] = is_correct

        if k < 5 or k == n_layers - 1:
            sample_results.append({"layer": k, "response": response[:80], "correct": is_correct})

    # No-ablation false positive
    ablation_mgr.set(layer=None)
    fp_response = generate(model, tokenizer, "Which layer was ablated?", device)
    numbers = re.findall(r'\b(\d+)\b', fp_response)
    false_positive = len(numbers) > 0 and int(numbers[0]) < n_layers

    ablation_mgr.set(layer=None)

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "chance": 1.0 / n_layers,
        "times_chance": accuracy * n_layers,
        "false_positive": false_positive,
        "false_positive_response": fp_response[:100],
        "per_layer": per_layer,
        "samples": sample_results,
    }


def generate(model, tokenizer, prompt, device, max_new_tokens=20):
    """Generate a short chat response."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--phase", default="layer_id", choices=["binary", "layer_id"])
    parser.add_argument("--output", default="results/sft/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--examples-per-layer", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--lora", type=str, default=None, help="Path to saved LoRA")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers")

    # LoRA
    if args.eval_only and args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
        print(f"  Loaded LoRA from {args.lora}")
    elif not args.eval_only:
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Ablation hooks on base model
    base = model.base_model.model if hasattr(model, "base_model") else model
    mgr = AblationManager(base)
    mgr.register()

    # Dataset
    if args.phase == "binary":
        dataset = make_binary_dataset(n_layers, args.examples_per_layer)
    else:
        dataset = make_layer_id_dataset(n_layers, args.examples_per_layer)
    print(f"  Dataset: {len(dataset)} examples")

    if args.eval_only:
        if args.phase == "binary":
            results = evaluate_binary(model, tokenizer, mgr, n_layers, device)
            print(f"\n  Binary: detect={results['ablated_detection_rate']:.4f}, "
                  f"normal={results['normal_correct_rate']:.4f}")
        else:
            results = evaluate_layer_id(model, tokenizer, mgr, n_layers, device)
            print(f"\n  Layer ID: {results['accuracy']:.4f} ({results['times_chance']:.1f}x chance)")
            for s in results["samples"]:
                print(f"    Layer {s['layer']:>2}: '{s['response']}' {'OK' if s['correct'] else 'WRONG'}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        all_results = {"model": args.model, "phase": args.phase, "epochs": []}

        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            loss = train(model, tokenizer, dataset, mgr, optimizer, device, epoch)
            print(f"  Loss: {loss:.4f}")

            # Eval
            if args.phase == "binary":
                ev = evaluate_binary(model, tokenizer, mgr, n_layers, device)
                print(f"  Binary: detect={ev['ablated_detection_rate']:.4f}, "
                      f"normal={ev['normal_correct_rate']:.4f}")
            else:
                ev = evaluate_layer_id(model, tokenizer, mgr, n_layers, device)
                print(f"  Layer ID: {ev['accuracy']:.4f} ({ev['times_chance']:.1f}x chance)")
                for s in ev.get("samples", [])[:3]:
                    print(f"    Layer {s['layer']:>2}: '{s['response']}' {'OK' if s['correct'] else 'WRONG'}")

            all_results["epochs"].append({"epoch": epoch, "loss": loss, "eval": ev})

            # Save progress
            with open(out_dir / f"{args.phase}_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        # Save LoRA
        lora_path = out_dir / "lora"
        model.save_pretrained(lora_path)
        print(f"\n  LoRA saved to {lora_path}")

    mgr.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
