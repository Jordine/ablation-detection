# Ablation Detection

Can a transformer learn to report which of its own layers was ablated?

## What this is

Train a model to develop proprioception — the ability to sense and report modifications to its own computational structure during a forward pass.

When layer k is skipped, layers k+1..L process a residual stream missing layer k's contribution. This creates a downstream signature. We train the model (via SFT and RLVR) to decode "which signature am I seeing?" from its own computation.

## What this is NOT

- Not a publication-oriented project (no need for clean framing)
- Not the same as the frankenmodel project (that's zero-shot detection; this is trained detection)
- Not the same as the surgery project (that injects info; this detects perturbations)

## Models

Primary: Qwen/Qwen2.5-7B-Instruct (28 layers, fast iteration)
Scale-up: Qwen/Qwen2.5-32B-Instruct (64 layers)

## Related projects

- `../architectural_awareness_models/` — surgery + probing (injecting info, not detecting perturbations)
- `../random_ass_experiments/frankenmodel/` — zero-shot "is something wrong?" (not trained detection)

## Infrastructure

- vast.ai GPU rentals, SSH via grongles key
- Don't touch the 8xH100 instance without asking
