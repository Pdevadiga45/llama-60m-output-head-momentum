# SCALE last-layer momentum reproduction

Independent reproduction of the 60M-parameter last-layer-momentum ablation from **A Minimalist Optimizer Design for LLM Pretraining** (arXiv `2506.16659v3`).

The study asks one question: with the paper's model, data budget, schedule, and column-normalized updates fixed, does momentum restricted to the language-model head reproduce the reported large perplexity improvement?

This repository does not copy the unlicensed upstream implementation. It implements the paper-described optimizer independently using PyTorch and standard Hugging Face components.

See `PROTOCOL.md` and `protocol.json` for the frozen estimand, budget gate, and stop conditions. See `RUNBOOK.md` for the supervised CLI-only Jarvis procedure.

## Local verification

```bash
python -m pytest -q
scale-repro smoke --arm no_momentum --output artifacts/smoke-no-momentum
scale-repro smoke --arm head_momentum --output artifacts/smoke-head-momentum
```

Paid work uses one supervised A30 session: run and recover the canary, finalize its gate from measured provider time and spend, run both full arms on the same instance only if the gate passes, download all artifacts, then destroy the instance and require `jl list --json` to be empty.
