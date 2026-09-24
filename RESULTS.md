# Results

## Verdict

**Strong reproduction of the optimizer-component effect.** Momentum restricted to `lm_head.weight` improved final validation perplexity from **39.95** to **32.34**, a paired improvement of **7.60** perplexity points.

| Endpoint | Paper | Reproduction | Difference |
|---|---:|---:|---:|
| No momentum PPL | 39.89 | 39.95 | +0.06 |
| LM-head momentum PPL | 30.81 | 32.34 | +1.53 |
| Improvement | 9.08 | 7.60 | -1.48 |

The observed improvement exceeds the frozen strong-reproduction threshold of 4.54. The control endpoint nearly exactly matches the paper; the momentum endpoint improves substantially but does not reach the paper's reported 30.81 PPL.

## Learning curve

| Update | No momentum PPL | LM-head momentum PPL | Paired delta |
|---:|---:|---:|---:|
| 1,000 | 170.72 | 120.43 | 50.30 |
| 2,000 | 86.53 | 62.35 | 24.18 |
| 3,000 | 70.89 | 52.18 | 18.71 |
| 4,000 | 63.47 | 46.65 | 16.82 |
| 5,000 | 58.91 | 42.57 | 16.34 |
| 6,000 | 53.96 | 39.55 | 14.41 |
| 7,000 | 48.10 | 36.94 | 11.16 |
| 8,000 | 43.90 | 34.97 | 8.93 |
| 9,000 | 41.63 | 33.56 | 8.08 |
| 10,000 | 40.51 | 32.73 | 7.78 |
| 11,000 | 39.95 | 32.34 | 7.60 |

The momentum arm is better at all 11 scheduled evaluations. Both curves improve monotonically through the final evaluation.

## Paired-run evidence

- Model: LLaMA-60M, 116,147,200 model-parameter bytes, 58,073,600 parameters.
- Training: 11,000 updates per arm, 1,441,792,000 nominal tokens per arm, BF16.
- Same initial state: `566cfd28890c0f3b6689001035b31e60066890c89ae1aa10ea454d1d90ebd45a`.
- Same training stream: `04a84aed71adda5ec255b52090fb863077058990e461237c5ea43b06212bc024`.
- Same evaluation stream: `99d1f5af2a466797fd38f1f819c237926a648c010de5bab2b1429014ed6069b9`.
- Same committed implementation: `b223e43795efb2f8f4a034bad27fe0f904e6a5fa`.
- Both artifact bundles passed manifest, raw-metric, finite-value, contiguous-update, exact-inventory, and strict model-state verification.

## Cost and lifecycle

- Hardware: one NVIDIA A30 reused for the canary and both full arms.
- Canary gate: passed after 1,200 updates.
- Total provider spend: **₹717.14** against a ₹1200 cap.
- The total includes roughly ₹146 of avoidable idle time after a monitor disconnect delayed the second arm.
- Final artifacts were downloaded and verified before destruction; the Jarvis inventory was verified empty.

## Cross-domain extension: Paloma-mini

The two frozen checkpoints were evaluated without retraining on 12 complete files from Paloma revision `65cd6fc59dba021b21db414fa5e8d7765ffbe5e6`: four C4 control domains, four Reddit domains, and four programming-language domains. Evaluation used the pinned T5 tokenizer, the `lm-evaluation-harness` disjoint rolling-window policy at sequence length 256, CUDA BF16, and FP32 log-softmax/reduction.

| Domain group | Domains | No-momentum geometric-mean PPL | Head-momentum geometric-mean PPL | Delta nats/token |
|---|---:|---:|---:|---:|
| C4 controls | 4 | 53.63 | 42.96 | -0.2218 |
| Reddit | 4 | 49.73 | 44.14 | -0.1192 |
| Code | 4 | 43.56 | 36.79 | -0.1689 |
| Reddit + code headline set | 8 | 46.54 | 40.30 | -0.1441 |

The head-momentum checkpoint achieved lower cross-entropy on **all 12 domains**. On the eight-domain Reddit/code headline set, geometric-mean perplexity fell from **46.54 to 40.30**, a **6.24-point / 13.4% reduction**. The effect was present in both OOD groups and was larger on the four C4 controls.

This is evidence of **paired cross-domain fit**, not uncontaminated generalization: overlap between pretraining C4 and Paloma sources cannot be ruled out, and this is still one paired training seed. The result strengthens the conclusion that output-head momentum produced a broadly better checkpoint rather than improving only the original C4 validation stream.

Extension evidence:

- Evaluator commit: `8205a7a8977fa4e2393e8439f499aaf47f734cf5`.
- Packaged runtime evaluator hash: `cfdb907cfaeee8aca52f1588ead0b03f1e23233a77900f3e5f1ffeddeac9503f` (computed from the wheel's CRLF-normalized package files; it is intentionally distinct from hashing the repository's LF Git blobs).
- Extension protocol hash: `b655a96966b46e8f7df4a7d9a217fd7efceaf1dcc8a9f6af5119d2fef51342f4`.
- Machine-readable bundle: `results/paloma/`.
- Provider runtime, spend, verification, and cleanup metadata: `results/paloma-session.json` (kept outside the bundle's exact-inventory contract).
- Additional provider spend: **₹10.54**, bringing total project spend to **₹727.68**; final Jarvis inventory was verified empty.
- The proposed coordinate-clipping arm was not run. Independent review showed that its audit and authorization machinery outweighed its incremental scientific value; no paid clipping compute was used.

## Interpretation and limits

This reproduces the **component effect**: under the frozen recipe, adding momentum only to the language-model head materially improves validation perplexity. It does not establish the paper's proposed gradient-variance explanation as the causal mechanism.

This is one paired seed and one configuration, so it is a configuration reproduction rather than a population estimate. Evaluation uses token-weighted valid-token NLL; the pinned upstream code averages batch losses. The paired comparison is internally consistent, but absolute perplexities are therefore not strictly metric-identical to upstream.

## Reproducibility anchors

- Paper: arXiv `2506.16659v3`.
- Upstream commit: `94712d907f6cc94528b04dffb632215b5ed20a0d`.
- Reproduction commit: `b223e43795efb2f8f4a034bad27fe0f904e6a5fa`.
- Protocol hash used on the A30: `6fe17a40d705bfd03834f7d4ec05bda56a6ad9402b8826f88cada184f65bcabd`.
- Runtime-code hash: `1d20506de0a09ecf4ef0b92d666481692b6d948a5ed4c69655aabffee723759b`.
- Machine-readable comparison: `artifacts/comparison.json` (local, gitignored).
