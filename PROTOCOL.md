# Frozen protocol — September 22, 2026

## Paper and claim

- Paper: *A Minimalist Optimizer Design for LLM Pretraining*, arXiv `2506.16659v3`.
- Upstream reference commit: `94712d907f6cc94528b04dffb632215b5ed20a0d`.
- Reproduced claim: in the paper's 60M recipe, adding first-order momentum only to the LM head materially improves optimization and final validation perplexity over the same column-normalized optimizer without momentum.
- Reported endpoints: `39.89` perplexity without momentum and `30.81` with last-layer momentum.
- Excluded claims: SCALE versus Adam/AdamW at larger scales; proof that gradient variance mediates the effect; general optimizer superiority.

## Fixed setup

- Standard Hugging Face LLaMA causal model, initialized from scratch.
- Hidden size 512; 8 layers; 8 attention heads; intermediate size 1376; vocabulary 32,000.
- C4 English, immutable dataset revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.
- T5-base tokenizer revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`.
- Sequence length 256; global batch 512; 11,000 optimizer updates; 1,441,792,000 nominal tokens.
- BF16; learning rate `1e-3`; zero weight decay; cosine schedule; 1,100 warmup updates; seed 42.
- Validation on 10,000,000 tokens every 1,000 updates and after update 11,000.

## Arms

1. `no_momentum`: column-normalize all matrix gradients; use no momentum on the LM head.
2. `head_momentum`: identical updates with first-order momentum `0.9` only on the LM head.

One-dimensional parameters use Adam-style first and second moments in both arms. No other parameter receives momentum.

## Pairing and provenance

- Both arms load the same serialized initial model state.
- Both use the same pinned dataset/tokenizer revisions, shuffle seed, batch order, and evaluation order.
- Each run records a rolling digest of token IDs; mismatched digests invalidate the paired comparison.
- Every metrics record includes update, nominal tokens, train loss, validation loss/perplexity, LR, throughput, and peak memory.

## Canary

- 500 optimizer updates on the exact production path and one 10M-token validation pass.
- Maximum canary wall time: 2 hours; maximum canary spend: ₹80.
- Project full-arm time from post-warmup update timings plus measured validation timing.
- Pass only if two 11,000-update arms project to no more than 20 paid A30-hours at the current on-demand price.
- Operational threshold at the frozen price is approximately 40,056 aggregate training tokens/second before the reserved setup/recovery margin.
- Failure ends the project. Do not shorten training.

## Outcome interpretation

Primary estimand: `PPL(no_momentum) - PPL(head_momentum)` at the final frozen evaluation.

- Strong reproduction: difference at least `4.54` (half the reported `9.08`) and the momentum arm is better on at least three of the final four evaluations.
- Challenge/null: both runs are healthy and differ by no more than `1.0`, or the direction reverses.
- Intermediate: difference between `1.0` and `4.54`; report without upgrading it to success or null.
- Assay failure: either run misses basic health checks, data digests differ, or the execution protocol deviates.

A single paired seed is an exact configuration reproduction, not a population-level estimate.

## Budget and lifecycle

- Hard project cap: ₹950.
- Price snapshot: A30 on-demand ₹38.88/hour on September 22, 2026.
- Full-run projection gate: at most ₹777.60, leaving ₹172.40 for setup/recovery variance.
- Use Jarvis Labs only through `jl` and SSH/SCP.
- Each paid batch follows create → upload committed code → run → download → validate recoverability → destroy → verify `jl list --json` is `[]`.
