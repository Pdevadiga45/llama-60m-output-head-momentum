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
- BF16; learning rate `1e-3`; zero weight decay; cosine schedule to a `0.1` minimum-LR ratio; 1,100 warmup updates; seed 42.
- Validation on 10,000,000 tokens every 1,000 updates and after update 11,000.

## Arms

1. `no_momentum`: column-normalize all matrix gradients; use no momentum on the LM head.
2. `head_momentum`: identical updates with first-order momentum `0.9` only on the LM head.

One-dimensional parameters use Adam-style first and second moments in both arms. No other parameter receives momentum.

## Pairing and provenance

- Both arms load the same serialized initial model state.
- Both use the same pinned dataset/tokenizer revisions, shuffle seed, batch order, and evaluation order.
- Each run records a rolling digest of token IDs; mismatched digests invalidate the paired comparison.
- Every training record includes update, cumulative nominal tokens, train loss, the LR applied to that update, update time, nominal throughput, cumulative non-padding tokens, and peak allocated memory.
- Every evaluation record separately includes update, token-weighted loss/perplexity, input-token count, valid shifted-target count, data digest, batch count, and elapsed time.
- `manifest.json` binds each recovered artifact to its byte size and SHA-256 digest and records the clean source commit. Comparison is permitted only after both manifests verify locally.

## Canary amendments

- The original 500-update canary was invalidated during pre-compute review because it ended before the frozen 1,100-update warmup. No paid run had started.
- The amended canary runs 1,200 optimizer updates on the exact production path and one 10M-token validation pass.
- Runtime projection uses exactly updates 1,101–1,200, giving 100 post-warmup timing samples.
- Maximum provider wall time through canary recovery: 2 hours; maximum measured canary spend: ₹120.
- The remote result remains `scale_gate: "pending_lifecycle"`; it is finalized locally only after the artifact manifest verifies and `jl status --json` provides the session's measured wall/spend inputs.
- Full-run projection includes the provider-wall time not represented by the canary's timed updates and evaluation, so dataset setup and other gaps are not discarded.
- Pass only if two 11,000-update arms project to no more than 20 paid A30-hours, each arm fits the 10-hour timeout, projected full-run spend is at most ₹777.60, total projected project spend is at most ₹1,200, and aggregate training throughput is at least 40,056 tokens/second.
- Failure ends paid work. Do not shorten training or relax the scientific thresholds.

## Outcome interpretation

Primary estimand: `PPL(no_momentum) - PPL(head_momentum)` at the final frozen evaluation.

- Strong reproduction: difference at least `4.54` (half the reported `9.08`) and the momentum arm is better on at least three of the final four evaluations.
- Challenge/null: both runs are healthy and differ by no more than `1.0`, or the direction reverses.
- Intermediate: difference between `1.0` and `4.54`; report without upgrading it to success or null.
- Assay failure: either run misses basic health checks, data digests differ, or the execution protocol deviates.

A single paired seed is an exact configuration reproduction, not a population-level estimate.

Evaluation uses total valid-token NLL. The referenced upstream commit averages batch losses, so the paired effect is directly tested but absolute perplexity is not strictly metric-identical to the upstream implementation.

## Budget and supervised lifecycle

- Hard project cap: ₹1,200.
- Price snapshot: A30 on-demand ₹38.88/hour on September 22, 2026.
- Canary cap: ₹120; full-run projection cap: ₹777.60; minimum uncommitted reserve after both caps: ₹302.40.
- Use Jarvis Labs only through `jl` and SSH/SCP.
- Use one A30 for the canary and both full arms to avoid repeated provisioning and environment setup.
- Before each stage, record `jl status --json`, confirm cumulative measured spend plus the next stage's frozen maximum remains below ₹1,200, and stop on missing telemetry.
- Full training re-verifies the canary summary, metrics, evaluations, manifest, initial state, and finalized decision before CUDA.
- Download and verify each stage's artifacts while the instance remains running. Destroy the instance after the final verified download, or immediately after any failed/stopped stage once recoverable diagnostics have been downloaded.
- Never pause the instance. Final cleanup requires `jl list --json` to be `[]`.
