# Output-head momentum in LLaMA-60M pretraining

We implemented a column-normalized optimizer for LLaMA-60M and tested whether adding momentum to its output head improves pretraining. Two runs shared the same initialization, C4 data order, and training schedule; only output-head momentum changed.

The method follows *Memory-Efficient LLM Pretraining via Minimalist Optimizer Design* (arXiv `2506.16659v3`). The implementation uses PyTorch and standard Hugging Face components.

See `PROTOCOL.md` and `protocol.json` for the frozen estimand, budget gate, and stop conditions. See `RUNBOOK.md` for the supervised CLI-only Jarvis procedure.

## Result

The paired runs found **39.95 PPL without output-head momentum** and **32.34 PPL with it**, a **7.60-point improvement**. See `RESULTS.md` for the learning curve, paired evidence, limitations, and cost.

The same checkpoints were evaluated on **12 selected Paloma domains**, with head momentum lowering loss on all 12. Across the eight Reddit/code domains, geometric-mean perplexity improved from **46.54 to 40.30**. The machine-readable results are in `results/paloma/`.

## Local verification

```bash
python -m pytest -q
scale-repro smoke --arm no_momentum --output artifacts/smoke-no-momentum
scale-repro smoke --arm head_momentum --output artifacts/smoke-head-momentum
```

Paid work uses one supervised A30 session: run and recover the canary, finalize its gate from measured provider time and spend, run both full arms on the same instance only if the gate passes, download all artifacts, then destroy the instance and require `jl list --json` to be empty.
