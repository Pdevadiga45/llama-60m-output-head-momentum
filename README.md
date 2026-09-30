# Output-head momentum in LLaMA-60M pretraining

## Premise

Can a small amount of targeted optimizer state improve language-model training? This project tests that question by comparing column-normalized updates with and without momentum on the model's output head (`lm_head.weight`). The two models start from the same weights and see the same text in the same order.

## Implementation

- Built a PyTorch optimizer for a 58M-parameter LLaMA model. Matrix gradients use RMS normalization (dimension 1 for linear weights, dimension 0 for token embeddings); one-dimensional parameters use the same Adam-style moments in both runs. The only experimental change is a first-order momentum coefficient of `0.9` on the output head versus `0`.
- Trained each arm from scratch on C4 for **11,000 updates**, with context length **256**, global batch **512**, BF16 precision, and **1.44B nominal tokens per arm**. Both arms shared initialization, dataset order, tokenizer, schedule, and evaluation stream.
- Recorded token-stream digests, intermediate losses, checkpoints, and SHA-256 manifests. A 1,200-update canary checked the full GPU path before the paired A30 runs. The frozen recipes are in `protocol.json` and `extension_protocol.json`; the implementation and tests are in `scale_repro/` and `tests/`.

## Results

Lower perplexity (PPL) means better next-token prediction. Head momentum improved C4 validation PPL from **39.95 to 32.34**, a **7.60-point** difference.

| Training update | Without output-head momentum | With output-head momentum |
|---:|---:|---:|
| 1,000 | 170.72 | 120.43 |
| 2,000 | 86.53 | 62.35 |
| 3,000 | 70.89 | 52.18 |
| 4,000 | 63.47 | 46.65 |
| 5,000 | 58.91 | 42.57 |
| 6,000 | 53.96 | 39.55 |
| 7,000 | 48.10 | 36.94 |
| 8,000 | 43.90 | 34.97 |
| 9,000 | 41.63 | 33.56 |
| 10,000 | 40.51 | 32.73 |
| 11,000 | **39.95** | **32.34** |

We then evaluated the two frozen checkpoints, without further training, on **12 selected Paloma files**: four C4-derived controls, four Reddit domains, and four programming-language domains. Head momentum lowered loss on all 12. The table reports geometric-mean perplexity across each group:

| Domain group | Files | Without head momentum | With head momentum |
|---|---:|---:|---:|
| C4 controls | 4 | 53.63 | 42.96 |
| Reddit | 4 | 49.73 | 44.14 |
| Code | 4 | 43.56 | 36.79 |
| Reddit + code | 8 | **46.54** | **40.30** |

The result is a controlled **single-seed comparison** and a paired cross-domain evaluation of those same checkpoints. The Paloma files were not decontaminated against the training data. C4 validation uses token-weighted loss, whereas the paper's upstream evaluator averages batch losses; the paired comparison is internally consistent, but absolute PPLs are not strictly metric-identical. The Paloma evaluation uses disjoint rolling windows at context length 256. The full Paloma result bundle, including per-file metrics and hashes, is in `results/paloma/`. Full training checkpoints and raw metrics are kept locally under gitignored `artifacts/` because of their size.

Total A30 provider spend was **₹727.68**, including about **₹146** of avoidable idle time during training. All final artifacts were recovered before the instances were destroyed.

## Run locally

```bash
python -m pytest -q
scale-repro smoke --arm no_momentum --output artifacts/smoke-no-momentum
scale-repro smoke --arm head_momentum --output artifacts/smoke-head-momentum
```

The optimizer recipe follows *Memory-Efficient LLM Pretraining via Minimalist Optimizer Design* (arXiv `2506.16659v3`); the implementation here was written independently.
