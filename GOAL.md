# Persistent goal

Reproduce the SCALE paper's LLaMA-60M last-layer-momentum ablation faithfully on one NVIDIA A30 without exceeding ₹1,200.

## Complete when

1. an independent, tested implementation matches the paper's parameter partition and update equations;
2. a deterministic local smoke verifies paired initialization, data-order fingerprinting, metrics, and artifacts;
3. a paid 1,200-update A30 canary measures the exact execution path, including 100 post-warmup timing samples;
4. the canary projects both full arms at no more than 20 paid A30-hours;
5. if the gate passes, both 1.4B-token arms finish with recoverable local artifacts on the same A30;
6. the A30 is destroyed after the final verified download, or immediately after a failed/stopped run, and `jl list --json` is `[]`;
7. the final claim is limited to the fixed 60M component effect.

## Dead end when

- the exact canary projects more than 20 paid A30-hours;
- the official C4/tokenizer inputs cannot be pinned and replayed deterministically;
- the independent optimizer cannot match paper-described semantics;
- the canary or full session cannot finish within the ₹1,200 project cap.

Do not shorten the token budget, weaken the scientific gate, substitute a toy dataset, add post-hoc optimizer arms, or convert an assay failure into a scientific null.
