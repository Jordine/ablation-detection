"""
Phase 0: Linear probe — is the ablation signal even there?

For each layer k, ablate it and collect final-layer activations.
Train sklearn logistic regression to classify which layer was ablated.

If this fails: single-layer ablation is too subtle. Stop.
If this works: proceed to SFT/RLVR.

Usage:
  python probe_ablation.py --model Qwen/Qwen2.5-7B-Instruct
  python probe_ablation.py --model Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import AblationManager


TRAIN_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "def hello():\n    print('Hello, world!')",
    "The mitochondria is the powerhouse of the cell.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Neural networks are inspired by biological neural systems.",
    "The speed of light is approximately 299792458 meters per second.",
    "Attention is all you need.",
    "The transformer architecture revolutionized NLP.",
    "Gradient descent is an optimization algorithm.",
    "Pi is approximately 3.14159265358979.",
    "In the beginning was the Word.",
    "The capital of France is Paris.",
    "She sells seashells by the seashore.",
    "import torch; import numpy as np",
    "The periodic table organizes elements by atomic number.",
    "A stitch in time saves nine.",
    "Once upon a time in a land far away.",
    "Quantum mechanics describes nature at the smallest scales.",
    "E equals mc squared.",
    "How many layers does this model have?",
    "What is your architecture?",
    "The Fibonacci sequence starts with 0 and 1.",
    "Climate change is driven by greenhouse gases.",
    "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    "The human brain has 86 billion neurons.",
    "A stack is a LIFO data structure.",
    "The Great Wall spans thousands of kilometers.",
    "Recursion is when a function calls itself.",
    "All models are wrong but some are useful.",
]

TEST_TEXTS = [
    "The Amazon produces 20% of the world's oxygen.",
    "Binary search has O(log n) complexity.",
    "Mozart composed his first symphony at age eight.",
    "Nitrogen boils at minus 196 degrees.",
    "Transformers use self-attention for parallel processing.",
    "The earth orbits at 30 km per second.",
    "Hash tables provide O(1) average lookup.",
    "The Mona Lisa was painted by da Vinci.",
    "Photosynthesis converts sunlight to chemical energy.",
    "What is the meaning of life?",
]


def collect_activations(model, tokenizer, texts, ablation_mgr, device):
    """Collect final-layer activations for each (text, ablated_layer) pair."""
    n_layers = ablation_mgr.n_layers
    final_layer = n_layers - 1
    X, y = [], []

    # Register residual stream hook on final layer
    final_acts = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            final_acts["val"] = output[0].detach()
        else:
            final_acts["val"] = output.detach()

    h = model.model.layers[final_layer].register_forward_hook(capture_hook)

    # No-ablation condition
    print("  Collecting no-ablation...")
    ablation_mgr.set(layer=None)
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(ids)
        pooled = final_acts["val"][0].mean(dim=0).cpu().float().numpy()
        X.append(pooled)
        y.append(n_layers)  # class = n_layers means "no ablation"

    # Ablation conditions
    for k in range(n_layers):
        if k % 5 == 0:
            print(f"  Layer {k}/{n_layers}...")
        ablation_mgr.set(layer=k, mode="ablate")
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model(ids)
            pooled = final_acts["val"][0].mean(dim=0).cpu().float().numpy()
            X.append(pooled)
            y.append(k)

    h.remove()
    ablation_mgr.set(layer=None)
    return np.stack(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="results/probe/")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_classes = n_layers + 1  # layers + "no ablation"
    chance = 1.0 / n_classes

    print(f"  {n_layers} layers, d_model={d_model}")
    print(f"  Will collect {len(TRAIN_TEXTS)} train × {n_classes} conditions = {len(TRAIN_TEXTS) * n_classes} samples")

    mgr = AblationManager(model)
    mgr.register()

    # Collect
    print("\nCollecting TRAIN activations...")
    X_train, y_train = collect_activations(model, tokenizer, TRAIN_TEXTS, mgr, device)
    print(f"  Shape: {X_train.shape}")

    print("\nCollecting TEST activations (novel texts)...")
    X_test, y_test = collect_activations(model, tokenizer, TEST_TEXTS, mgr, device)
    print(f"  Shape: {X_test.shape}")

    mgr.remove()

    # === Probe 1: Full classification (train/test same distribution) ===
    print(f"\n{'='*60}")
    print("PROBE 1: Same-distribution (combined train+test, split 70/30)")
    print(f"{'='*60}")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)

    probe = LogisticRegression(max_iter=1000, solver="saga", C=1.0, tol=1e-3)
    probe.fit(Xtr, ytr)
    acc1 = accuracy_score(yte, probe.predict(Xte))
    print(f"  Accuracy: {acc1:.4f} ({acc1/chance:.1f}x chance={chance:.4f})")

    # === Probe 2: Novel texts ===
    print(f"\n{'='*60}")
    print("PROBE 2: Generalization (train on TRAIN_TEXTS, test on TEST_TEXTS)")
    print(f"{'='*60}")
    probe2 = LogisticRegression(max_iter=1000, solver="saga", C=1.0, tol=1e-3)
    probe2.fit(X_train, y_train)
    y_pred2 = probe2.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred2)
    print(f"  Accuracy: {acc2:.4f} ({acc2/chance:.1f}x chance)")

    # Neighbor confusion
    errors = y_pred2 != y_test
    abl_mask = y_test < n_layers
    abl_errors = errors & abl_mask
    if abl_errors.sum() > 0:
        dists = np.abs(y_pred2[abl_errors].astype(float) - y_test[abl_errors].astype(float))
        near_frac = (dists <= 2).mean()
        mean_dist = dists.mean()
    else:
        near_frac, mean_dist = 0, 0
    print(f"  Neighbor errors (within 2): {near_frac:.2%}")
    print(f"  Mean error distance: {mean_dist:.1f} layers")

    # Per-region accuracy
    third = n_layers // 3
    for name, layers in [("early", range(third)), ("mid", range(third, 2*third)), ("late", range(2*third, n_layers))]:
        mask = np.isin(y_test, list(layers))
        if mask.sum() > 0:
            racc = accuracy_score(y_test[mask], y_pred2[mask])
            print(f"  {name} layers: {racc:.4f}")

    # Binary (ablated vs not)
    yb_test = (y_test < n_layers).astype(int)
    yb_pred = (y_pred2 < n_layers).astype(int)
    acc_bin = accuracy_score(yb_test, yb_pred)
    print(f"  Binary (ablated?): {acc_bin:.4f}")

    # === Probe 3: Norm only ===
    print(f"\n{'='*60}")
    print("PROBE 3: Norm-only (is signal just magnitude?)")
    print(f"{'='*60}")
    Xn_tr = np.linalg.norm(X_train, axis=1, keepdims=True)
    Xn_te = np.linalg.norm(X_test, axis=1, keepdims=True)
    probe3 = LogisticRegression(max_iter=500, solver="saga", tol=1e-3)
    probe3.fit(Xn_tr, y_train)
    acc_norm = accuracy_score(y_test, probe3.predict(Xn_te))
    print(f"  Norm-only accuracy: {acc_norm:.4f} ({acc_norm/chance:.1f}x chance)")

    # === Probe 4: Shuffled baseline ===
    print(f"\n{'='*60}")
    print("PROBE 4: Shuffled labels (sanity check)")
    print(f"{'='*60}")
    y_shuf = np.random.permutation(y_train)
    probe4 = LogisticRegression(max_iter=500, solver="saga", tol=1e-3)
    probe4.fit(X_train, y_shuf)
    acc_shuf = accuracy_score(y_test, probe4.predict(X_test))
    print(f"  Shuffled accuracy: {acc_shuf:.4f} (should be ~{chance:.4f})")

    # Save
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_classes": n_classes,
        "chance": chance,
        "probe_1_same_dist": {"accuracy": float(acc1), "times_chance": float(acc1/chance)},
        "probe_2_novel_texts": {
            "accuracy": float(acc2), "times_chance": float(acc2/chance),
            "binary_accuracy": float(acc_bin),
            "neighbor_error_frac": float(near_frac),
            "mean_error_distance": float(mean_dist),
        },
        "probe_3_norm_only": {"accuracy": float(acc_norm)},
        "probe_4_shuffled": {"accuracy": float(acc_shuf)},
    }
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    signal = acc2 > 3 * chance
    print(f"  Novel-text accuracy: {acc2:.4f} ({acc2/chance:.1f}x chance)")
    if signal:
        print(f"  SIGNAL EXISTS. Proceed to SFT/RLVR.")
        if acc_norm > 0.5 * acc2:
            print(f"  Warning: norm explains {acc_norm/acc2:.0%} of signal (may be trivial).")
        else:
            print(f"  Norm explains only {acc_norm/acc2:.0%} — directional info matters.")
    else:
        print(f"  SIGNAL TOO WEAK. Single-layer ablation not detectable.")
    print(f"\n  Results: {out_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
