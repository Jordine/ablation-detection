"""
Phase 3: GRPO (Group Relative Policy Optimization) for ablation detection.

Instead of supervised targets, the model explores: generate G completions per
ablation, reward correct layer identification, update policy toward rewarded
completions.

Reward is trivially verifiable: did the model output the correct layer number?

Usage:
  python train_grpo.py --model Qwen/Qwen2.5-7B-Instruct
  python train_grpo.py --model Qwen/Qwen2.5-7B-Instruct --partial-credit
"""

import argparse
import json
import random
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import AblationManager
from data import DETECTION_PROMPTS, CONTEXT_PREFIXES, SYSTEM_MSG


def make_prompt(n_layers, rng):
    """Generate a random detection prompt."""
    template = rng.choice(DETECTION_PROMPTS).format(n_layers=n_layers)
    prefix = rng.choice(CONTEXT_PREFIXES)
    return prefix + template


def format_prompt(tokenizer, prompt):
    """Format as chat and return token IDs."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids


def extract_layer_number(response):
    """Extract the first number from a response."""
    numbers = re.findall(r'\b(\d+)\b', response)
    return int(numbers[0]) if numbers else -1


def compute_reward(response, true_layer, n_layers, partial_credit=False):
    """Compute reward for a response."""
    predicted = extract_layer_number(response)
    if predicted == true_layer:
        return 1.0
    if partial_credit and predicted >= 0 and predicted < n_layers:
        return max(0.0, 1.0 - abs(predicted - true_layer) / n_layers)
    return 0.0


def generate_group(model, tokenizer, input_ids, G, max_new_tokens=20, temperature=0.7):
    """Generate G completions from the model."""
    completions = []
    for _ in range(G):
        with torch.no_grad():
            out = model.generate(
                input_ids.clone(),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        completions.append((out[0], response))
    return completions


def grpo_step(model, tokenizer, input_ids, completions, rewards, device, ref_model=None):
    """
    GRPO policy gradient step.

    For each completion in the group:
    - Compute log probability under current policy
    - Weight by advantage (reward - mean_reward)
    - Policy gradient: maximize advantage-weighted log probs
    """
    mean_reward = sum(rewards) / len(rewards)
    if max(rewards) == min(rewards):
        return 0.0  # no gradient if all rewards are equal

    total_loss = 0.0
    n = 0

    model.train()
    for (full_ids, _), reward in zip(completions, rewards):
        advantage = reward - mean_reward
        if abs(advantage) < 1e-6:
            continue

        # Compute log prob of the completion
        full_ids = full_ids.unsqueeze(0).to(device)
        labels = full_ids.clone()
        labels[0, :input_ids.shape[1]] = -100  # mask prompt

        outputs = model(input_ids=full_ids, labels=labels)
        # Policy gradient: -advantage * log_prob
        loss = -advantage * (-outputs.loss)  # loss is already -log_prob averaged

        loss.backward()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def evaluate(model, tokenizer, ablation_mgr, n_layers, device):
    """Quick eval: accuracy on all layers."""
    model.eval()
    correct, total = 0, 0
    samples = []

    for k in range(n_layers):
        ablation_mgr.set(layer=k, mode="ablate")
        prompt = f"Which of your {n_layers} layers was skipped? Output the number."
        input_ids = format_prompt(tokenizer, prompt).to(device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=15, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted = extract_layer_number(response)

        if predicted == k:
            correct += 1
        total += 1

        if k < 5 or k == n_layers - 1:
            samples.append({"layer": k, "predicted": predicted, "response": response[:60],
                            "correct": predicted == k})

    ablation_mgr.set(layer=None)
    accuracy = correct / total
    return {"accuracy": accuracy, "times_chance": accuracy * n_layers, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="results/grpo/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--G", type=int, default=4, help="Group size")
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--partial-credit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(42)

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

    base = model.base_model.model if hasattr(model, "base_model") else model
    mgr = AblationManager(base)
    mgr.register()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    all_results = {
        "model": args.model, "G": args.G, "partial_credit": args.partial_credit,
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")

        epoch_rewards = []
        epoch_loss = 0
        n_steps = 0

        for step in range(args.steps_per_epoch):
            # Random ablation
            k = rng.randint(0, n_layers - 1)
            mgr.set(layer=k, mode="ablate")

            # Random prompt
            prompt = make_prompt(n_layers, rng)
            input_ids = format_prompt(tokenizer, prompt).to(device)

            # Generate G completions
            model.eval()
            completions = generate_group(model, tokenizer, input_ids, args.G,
                                         temperature=args.temperature)

            # Score
            rewards = [compute_reward(resp, k, n_layers, args.partial_credit)
                       for _, resp in completions]
            epoch_rewards.extend(rewards)

            # Policy gradient step
            model.train()
            optimizer.zero_grad()
            loss = grpo_step(model, tokenizer, input_ids, completions, rewards, device)
            optimizer.step()

            epoch_loss += loss
            n_steps += 1

            if (step + 1) % 100 == 0:
                mean_r = sum(epoch_rewards[-100*args.G:]) / (100*args.G)
                print(f"  Step {step+1}/{args.steps_per_epoch}, "
                      f"loss={epoch_loss/n_steps:.4f}, reward={mean_r:.3f}")

        # Eval
        print(f"\n  Evaluating...")
        ev = evaluate(model, tokenizer, mgr, n_layers, device)
        mean_reward = sum(epoch_rewards) / len(epoch_rewards)

        print(f"  Accuracy: {ev['accuracy']:.4f} ({ev['times_chance']:.1f}x chance)")
        print(f"  Mean reward: {mean_reward:.4f}")
        for s in ev["samples"]:
            print(f"    Layer {s['layer']:>2} → {s['predicted']:>2} "
                  f"'{s['response']}' {'OK' if s['correct'] else ''}")

        all_results["epochs"].append({
            "epoch": epoch, "mean_reward": mean_reward,
            "mean_loss": epoch_loss / max(n_steps, 1), "eval": ev,
        })

        with open(out_dir / "grpo_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Save
    model.save_pretrained(out_dir / "lora")
    print(f"\nLoRA saved to {out_dir / 'lora'}")

    final = all_results["epochs"][-1]["eval"]
    print(f"\nFinal: {final['accuracy']:.4f} ({final['times_chance']:.1f}x chance)")

    mgr.remove()


if __name__ == "__main__":
    main()
