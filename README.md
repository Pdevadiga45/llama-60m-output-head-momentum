# SCALE last-layer momentum reproduction

Independent reproduction of the 60M-parameter last-layer-momentum ablation from **A Minimalist Optimizer Design for LLM Pretraining** (arXiv `2506.16659v3`).

The study asks one question: with the paper's model, data budget, schedule, and column-normalized updates fixed, does momentum restricted to the language-model head reproduce the reported large perplexity improvement?

This repository does not copy the unlicensed upstream implementation. It implements the paper-described optimizer independently using PyTorch and standard Hugging Face components.

See `PROTOCOL.md` and `protocol.json` for the frozen estimand, budget gate, and stop conditions. See `RUNBOOK.md` for the supervised CLI-only Jarvis procedure.

## Result

The reproduction found **39.95 PPL without momentum** and **32.34 PPL with momentum only on `lm_head.weight`**, a **7.60-point improvement** classified by the frozen protocol as a strong reproduction. See `RESULTS.md` for the learning curve, paired evidence, limitations, and cost.

The checkpoint-only Paloma extension also favored head momentum on **all 12 evaluated domains**. Across the eight Reddit/code headline domains, geometric-mean perplexity improved from **46.54 to 40.30**. This is reported as paired cross-domain fit—not uncontaminated generalization. The complete result bundle is under `results/paloma/`.

## Local verification

```bash
python -m pytest -q
scale-repro smoke --arm no_momentum --output artifacts/smoke-no-momentum
scale-repro smoke --arm head_momentum --output artifacts/smoke-head-momentum
```

Paid work uses one supervised A30 session: run and recover the canary, finalize its gate from measured provider time and spend, run both full arms on the same instance only if the gate passes, download all artifacts, then destroy the instance and require `jl list --json` to be empty.
