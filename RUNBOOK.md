# Supervised paid-run runbook

Applicable date: September 22, 2026.

No paid command may run until the repository is committed, the worktree is clean, the local suite passes, and the scientific-only review passes.

## 1. Preflight and create one A30

```bash
git status --short
git archive --format=zip --output="$TMPDIR/scale-reproduction.zip" HEAD
jl status --json > "$TMPDIR/scale-status-before.json"
jl gpus --json
jl list --json

REGION='<available A30 region from jl gpus>'
jl create --gpu A30 --template pytorch --storage 100 --name scale-momentum-repro \
  --region "$REGION" --yes --json > "$TMPDIR/jl-create.json"
ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["machine_id"])' "$TMPDIR/jl-create.json")"
jl exec "$ID" -- true
```

`git status --short` and the initial `jl list --json` must be empty. Raw create/get JSON stays in `$TMPDIR`; do not print it because it may contain tokenized URLs.

## 2. Upload and install the reviewed commit

```bash
COMMIT="$(git rev-parse HEAD)"
jl upload "$ID" "$TMPDIR/scale-reproduction.zip" /workspace/scale-reproduction.zip --json
jl exec "$ID" -- bash -lc \
  'rm -rf /workspace/scale && mkdir -p /workspace/scale /workspace/results && cd /workspace/scale && unzip -q ../scale-reproduction.zip && python -m pip install --no-cache-dir .'
```

## 3. Run and recover the 1,200-update canary

```bash
jl exec "$ID" -- bash -lc \
  "cd /workspace/scale && SCALE_COMMIT_SHA='$COMMIT' scale-repro train \
    --mode canary --arm head_momentum --protocol protocol.json \
    --output /workspace/results/canary"

jl download "$ID" /workspace/results/canary artifacts/canary -r --json
jl status --json > "$TMPDIR/scale-status-after-canary.json"
```

Use the CLI-reported runtime and balance delta to set `CANARY_WALL_HOURS` and `CANARY_SPEND_INR`. Stop if either value is unavailable.

```bash
scale-repro finalize-canary \
  --canary-run artifacts/canary \
  --protocol protocol.json \
  --provider-wall-hours "$CANARY_WALL_HOURS" \
  --spend-inr "$CANARY_SPEND_INR" \
  --output artifacts/canary-decision.json
```

Continue only when `canary-decision.json` says `scale_gate: "pass"`. The canary cap is ₹120; the total project cap is ₹1,200.

## 4. Run both full arms on the same A30

Upload the locally finalized decision. The verified canary artifacts and initial state already remain on the instance.

```bash
FULL_TIMEOUT_SECONDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["full_run_timeout_seconds_per_arm"])' artifacts/canary-decision.json)"
test "$FULL_TIMEOUT_SECONDS" -gt 0
jl upload "$ID" artifacts/canary-decision.json /workspace/canary-decision.json --json

jl exec "$ID" -- env SCALE_COMMIT_SHA="$COMMIT" FULL_TIMEOUT_SECONDS="$FULL_TIMEOUT_SECONDS" bash -lc \
  'cd /workspace/scale && timeout --signal=TERM --kill-after=5m "$FULL_TIMEOUT_SECONDS" scale-repro train \
    --mode full --arm no_momentum --protocol protocol.json \
    --initial-state /workspace/results/canary/initial_state.pt \
    --canary-run /workspace/results/canary \
    --canary-decision /workspace/canary-decision.json \
    --output /workspace/results/no_momentum'

jl download "$ID" /workspace/results/no_momentum artifacts/no_momentum -r --json
jl status --json > "$TMPDIR/scale-status-after-no-momentum.json"
```

Before the second arm, stop unless measured cumulative spend plus the frozen maximum remaining spend stays below ₹1,200.

```bash
jl exec "$ID" -- env SCALE_COMMIT_SHA="$COMMIT" FULL_TIMEOUT_SECONDS="$FULL_TIMEOUT_SECONDS" bash -lc \
  'cd /workspace/scale && timeout --signal=TERM --kill-after=5m "$FULL_TIMEOUT_SECONDS" scale-repro train \
    --mode full --arm head_momentum --protocol protocol.json \
    --initial-state /workspace/results/canary/initial_state.pt \
    --canary-run /workspace/results/canary \
    --canary-decision /workspace/canary-decision.json \
    --output /workspace/results/head_momentum'

jl download "$ID" /workspace/results/head_momentum artifacts/head_momentum -r --json
jl status --json > "$TMPDIR/scale-status-final.json"
```

## 5. Compare, then destroy

```bash
scale-repro compare \
  --no-momentum artifacts/no_momentum \
  --head-momentum artifacts/head_momentum \
  --protocol protocol.json \
  --output artifacts/comparison.json

jl destroy "$ID" --yes --json
jl list --json
```

The final list must be `[]`. Never pause the instance.

## Failure path

On any failed or stopped stage, first attempt to recover `/workspace/results` and command diagnostics, then destroy the known `ID` and require `jl list --json` to be empty:

```bash
jl download "$ID" /workspace/results artifacts/recovery -r --json || true
jl destroy "$ID" --yes --json
jl list --json
```

A failed gate or failed full arm is preserved as the result; do not reduce the token budget or retune one arm.
