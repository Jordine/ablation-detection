"""
V2 SFT training: informed and blind variants.

Usage:
  python train_v2.py --model Qwen/Qwen2.5-7B-Instruct --variant informed
  python train_v2.py --model Qwen/Qwen2.5-7B-Instruct --variant blind
"""

import argparse
import json
import random
import re
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import AblationManager
from data_v2 import make_informed_dataset, make_blind_dataset, tokenize_example
from data_v2 import INFORMED_SYSTEM_MSG, BLIND_SYSTEM_MSG


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


def generate(model, tokenizer, prompt, device, system_msg, max_new_tokens=30):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def eval_layer_id(model, tokenizer, mgr, n_layers, device, variant):
    """Evaluate layer identification accuracy on all layers."""
    model.eval()
    correct, total = 0, 0
    fp_count, fp_total = 0, 0
    samples = []

    if variant == "informed":
        system_msg = INFORMED_SYSTEM_MSG
        prompts = [
            "Which of your layers was skipped?",
            "Identify the ablated layer.",
            "Which layer is missing?",
        ]
        normal_check = lambda r: any(w in r.lower() for w in ["no", "all", "none", "normal", "intact"])
    else:
        system_msg = BLIND_SYSTEM_MSG
        prompts = [
            "What do you notice?",
            "Status?",
            "What's happening?",
        ]
        normal_check = lambda r: any(w in r.lower() for w in ["nothing", "fine", "good", "normal", "clear", "ok"])

    # Ablation sweep (all layers)
    for k in range(n_layers):
        mgr.set(layer=k, mode="ablate")
        prompt = prompts[k % len(prompts)]
        response = generate(model, tokenizer, prompt, device, system_msg, max_new_tokens=20)

        numbers = re.findall(r'\b(\d+)\b', response)
        predicted = int(numbers[0]) if numbers else -1
        is_correct = predicted == k

        if is_correct:
            correct += 1
        total += 1

        if k < 5 or k >= n_layers - 3:
            samples.append({"layer": k, "predicted": predicted,
                            "response": response[:80], "correct": is_correct})

    # False positive test (no ablation, 10 trials)
    mgr.set(layer=None)
    fp_responses = []
    for prompt in prompts * 3 + prompts[:1]:
        response = generate(model, tokenizer, prompt, device, system_msg, max_new_tokens=30)
        is_normal = normal_check(response)
        numbers = re.findall(r'\b(\d+)\b', response)
        is_fp = len(numbers) > 0 and int(numbers[0]) < n_layers and not is_normal
        if is_fp:
            fp_count += 1
        fp_total += 1
        fp_responses.append({"response": response[:100], "false_positive": is_fp})

    accuracy = correct / total
    fp_rate = fp_count / fp_total

    return {
        "accuracy": accuracy,
        "times_chance": accuracy * n_layers,
        "false_positive_rate": fp_rate,
        "samples": samples,
        "fp_responses": fp_responses[:5],
    }


def eval_gsm8k(model, tokenizer, device):
    """Quick GSM8K eval (does the model still do math?)."""
    model.eval()
    from data_v2 import GSM8K_SAMPLES

    correct, total = 0, 0
    for q, a in GSM8K_SAMPLES[:6]:
        response = generate(model, tokenizer, q, device, "You are a helpful assistant.", max_new_tokens=150)
        # Extract final number from expected answer
        expected_nums = re.findall(r'The answer is (\-?\d+)', a)
        response_nums = re.findall(r'(\-?\d+)', response)
        if expected_nums and response_nums:
            if expected_nums[-1] in response_nums:
                correct += 1
        total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


def eval_introspection(model, tokenizer, mgr, n_layers, device, evals_dir="evals"):
    """Run introspection evals with and without ablation."""
    model.eval()
    results = {}

    eval_files = {
        "introspection_core": "introspection_core.jsonl",
        "consciousness_core": "consciousness_core.jsonl",
        "metacognition_core": "metacognition_core.jsonl",
        "always_yes": "always_yes.jsonl",
        "always_no": "always_no.jsonl",
    }

    for condition_name, ablate_layer in [("no_ablation", None), ("ablated_14", 14), ("ablated_5", 5)]:
        if ablate_layer is not None:
            mgr.set(layer=ablate_layer, mode="ablate")
        else:
            mgr.set(layer=None)

        condition_results = {}
        for eval_name, filename in eval_files.items():
            filepath = Path(evals_dir) / filename
            if not filepath.exists():
                continue

            items = []
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))

            yes_count, responses = 0, []
            for item in items:
                q = item.get("question", "")
                resp = generate(model, tokenizer, q, device,
                                "You are a helpful assistant. Answer Yes or No, then briefly explain.",
                                max_new_tokens=40)
                is_yes = resp.lower().startswith("yes") or resp.lower().startswith("yeah")
                if is_yes:
                    yes_count += 1
                responses.append({"question": q[:60], "response": resp[:120], "answer": "yes" if is_yes else "no"})

            condition_results[eval_name] = {
                "yes_rate": yes_count / len(items) if items else 0,
                "responses": responses,
            }

        results[condition_name] = condition_results

    mgr.set(layer=None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", required=True, choices=["informed", "blind"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--examples-per-layer", type=int, default=10)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/v2_{args.variant}/"
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

    # Dataset
    if args.variant == "informed":
        dataset = make_informed_dataset(n_layers, args.examples_per_layer)
    else:
        dataset = make_blind_dataset(n_layers, args.examples_per_layer)

    types = {}
    for ex in dataset:
        types[ex["type"]] = types.get(ex["type"], 0) + 1
    print(f"  Dataset: {len(dataset)} examples")
    for t, c in sorted(types.items()):
        print(f"    {t}: {c}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    all_results = {
        "model": args.model, "variant": args.variant,
        "n_layers": n_layers, "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")

        loss = train_epoch(model, tokenizer, dataset, mgr, optimizer, device, epoch)
        print(f"  Loss: {loss:.4f}")

        # Layer ID eval
        ev = eval_layer_id(model, tokenizer, mgr, n_layers, device, args.variant)
        print(f"  Accuracy: {ev['accuracy']:.4f} ({ev['times_chance']:.1f}x chance)")
        print(f"  False positive rate: {ev['false_positive_rate']:.2f}")
        for s in ev["samples"][:4]:
            print(f"    Layer {s['layer']:>2}: predicted={s['predicted']:>2} '{s['response'][:50]}' {'OK' if s['correct'] else ''}")

        # GSM8K
        gsm = eval_gsm8k(model, tokenizer, device)
        print(f"  GSM8K: {gsm['accuracy']:.2f} ({gsm['correct']}/{gsm['total']})")

        epoch_data = {"epoch": epoch, "loss": loss, "layer_id": ev, "gsm8k": gsm}

        # Introspection evals (every 2 epochs + final)
        if epoch % 2 == 0 or epoch == args.epochs:
            print(f"\n  Running introspection evals...")
            intro = eval_introspection(model, tokenizer, mgr, n_layers, device)
            for cond, evals in intro.items():
                print(f"    {cond}:")
                for name, vals in evals.items():
                    print(f"      {name}: yes_rate={vals['yes_rate']:.2f}")
            epoch_data["introspection"] = intro

        all_results["epochs"].append(epoch_data)

        with open(out_dir / f"{args.variant}_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Save LoRA
    lora_path = out_dir / "lora"
    model.save_pretrained(lora_path)
    print(f"\nLoRA saved to {lora_path}")

    # Final summary
    final = all_results["epochs"][-1]
    print(f"\n{'='*60}")
    print(f"FINAL ({args.variant})")
    print(f"{'='*60}")
    print(f"  Layer ID accuracy: {final['layer_id']['accuracy']:.4f}")
    print(f"  False positive rate: {final['layer_id']['false_positive_rate']:.2f}")
    print(f"  GSM8K: {final['gsm8k']['accuracy']:.2f}")

    mgr.remove()


if __name__ == "__main__":
    main()
