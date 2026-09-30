import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from .core import (
    ScaleOptimizer,
    TokenDigest,
    canary_checkpoint,
    compare_summaries,
    cosine_multiplier,
    finalize_supervised_canary,
    load_protocol,
    partition_scale_parameters,
)

_DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("core.py", "train.py"):
        digest.update(f"scale_repro/{name}".encode())
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def finalize_manifest(output: Path, metadata: dict, filenames: list[str]) -> dict:
    artifacts = {}
    for name in filenames:
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {**metadata, "artifacts": artifacts}
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_artifacts(output: str | Path, *, protocol_path: str | Path | None = None) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest has no artifacts")
    mode = summary.get("mode")
    if mode not in {"canary", "full"}:
        raise ValueError("production artifact mode must be canary or full")
    if summary.get("status") != "complete":
        raise ValueError("production artifact status must be complete")
    state_name = "initial_state.pt" if mode == "canary" else "final_state.pt"
    required = {"summary.json", "metrics.jsonl", "evaluations.jsonl", state_name}
    if set(artifacts) != required:
        raise ValueError(f"artifact inventory must be exactly {sorted(required)}")
    actual = {path.name for path in output.iterdir() if path.is_file()} - {"manifest.json"}
    if actual != required:
        raise ValueError(f"artifact directory inventory must be exactly {sorted(required)}")
    for name, expected in artifacts.items():
        if Path(name).name != name:
            raise ValueError(f"unsafe artifact name: {name}")
        path = output / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"artifact verification failed: {name}")
        if name.endswith(".json"):
            json.loads(path.read_text())
        elif name.endswith(".jsonl"):
            for line in path.read_text().splitlines():
                json.loads(line)
    for field in ("arm", "mode", "protocol_sha256", "code_sha256", "commit_sha", "initial_state_sha256"):
        if manifest.get(field) != summary.get(field):
            raise ValueError(f"manifest metadata differs from summary: {field}")
    if re.fullmatch(r"[0-9a-f]{40}", summary["commit_sha"]) is None:
        raise ValueError("production artifact commit_sha is invalid")
    selected_protocol = Path(protocol_path) if protocol_path is not None else _DEFAULT_PROTOCOL
    protocol = load_protocol(selected_protocol)
    if summary["protocol_sha256"] != _sha256(selected_protocol):
        raise ValueError("production artifact protocol_sha256 differs from the selected protocol")
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    evaluations = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
    expected_updates = protocol["canary"]["updates"] if mode == "canary" else protocol["training"]["updates"]
    if summary.get("updates") != expected_updates:
        raise ValueError("production artifact update count differs from the selected protocol")
    if [row.get("update") for row in metrics] != list(range(1, summary["updates"] + 1)):
        raise ValueError("metrics updates are not contiguous")
    metric_fields = {
        "update",
        "nominal_tokens",
        "train_loss",
        "lr",
        "update_seconds",
        "nominal_tokens_per_second",
        "nonpad_tokens",
        "peak_memory_bytes",
    }
    nominal_per_update = protocol["training"]["global_batch_size"] * protocol["training"]["sequence_length"]
    previous_nonpad = 0
    previous_peak = 0
    for row in metrics:
        update = row.get("update")
        expected_lr = protocol["training"]["learning_rate"] * cosine_multiplier(
            update - 1,
            warmup=protocol["training"]["warmup_updates"],
            total=protocol["training"]["updates"],
            minimum=protocol["training"]["min_lr_ratio"],
        )
        numeric = (int, float)
        valid = (
            set(row) == metric_fields
            and isinstance(update, int)
            and not isinstance(update, bool)
            and isinstance(row["nominal_tokens"], int)
            and not isinstance(row["nominal_tokens"], bool)
            and row["nominal_tokens"] == update * nominal_per_update
            and isinstance(row["train_loss"], numeric)
            and not isinstance(row["train_loss"], bool)
            and math.isfinite(row["train_loss"])
            and row["train_loss"] >= 0
            and isinstance(row["lr"], numeric)
            and not isinstance(row["lr"], bool)
            and math.isfinite(row["lr"])
            and row["lr"] >= 0
            and math.isclose(row["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
            and isinstance(row["update_seconds"], numeric)
            and not isinstance(row["update_seconds"], bool)
            and math.isfinite(row["update_seconds"])
            and row["update_seconds"] > 0
            and isinstance(row["nominal_tokens_per_second"], numeric)
            and not isinstance(row["nominal_tokens_per_second"], bool)
            and math.isfinite(row["nominal_tokens_per_second"])
            and row["nominal_tokens_per_second"] > 0
            and math.isclose(
                row["nominal_tokens_per_second"], nominal_per_update / row["update_seconds"], rel_tol=1e-12
            )
            and isinstance(row["nonpad_tokens"], int)
            and not isinstance(row["nonpad_tokens"], bool)
            and previous_nonpad <= row["nonpad_tokens"] <= row["nominal_tokens"]
            and isinstance(row["peak_memory_bytes"], int)
            and not isinstance(row["peak_memory_bytes"], bool)
            and previous_peak <= row["peak_memory_bytes"]
        )
        if not valid:
            raise ValueError("metrics rows are inconsistent with the selected protocol")
        previous_nonpad = row["nonpad_tokens"]
        previous_peak = row["peak_memory_bytes"]
    if summary.get("nonpad_tokens") != previous_nonpad or summary.get("peak_memory_bytes") != previous_peak:
        raise ValueError("metrics summary totals are inconsistent")
    evidence = {
        "eval_updates": [row.get("update") for row in evaluations],
        "eval_losses": [row.get("loss") for row in evaluations],
        "eval_perplexities": [row.get("perplexity") for row in evaluations],
        "eval_input_tokens": [row.get("tokens") for row in evaluations],
        "eval_target_tokens": [row.get("target_tokens") for row in evaluations],
        "eval_data_digests": [row.get("data_digest") for row in evaluations],
    }
    if any(summary.get(field) != values for field, values in evidence.items()):
        raise ValueError("summary evaluation evidence differs from evaluations.jsonl")
    if mode == "canary":
        if summary.get("eval_updates") != [protocol["canary"]["updates"]]:
            raise ValueError("canary evaluation update must equal the final canary update")
        if summary.get("scale_gate") != "pending_lifecycle" or summary.get("scientific_status") != "incomplete_canary":
            raise ValueError("canary artifact gate state is invalid")
        timing_samples = canary_timing_samples([row["update_seconds"] for row in metrics], protocol)
        evaluation_seconds = sum(row.get("seconds", float("nan")) for row in evaluations)
        checkpoint = summary.get("checkpoint_estimate", {})
        experiment_wall_hours = checkpoint.get("experiment_wall_hours")
        if (
            not isinstance(experiment_wall_hours, (int, float))
            or not math.isfinite(experiment_wall_hours)
            or not math.isfinite(evaluation_seconds)
            or evaluation_seconds <= 0
        ):
            raise ValueError("canary checkpoint estimate is invalid")
        measured_seconds = sum(row["update_seconds"] for row in metrics) + evaluation_seconds
        if experiment_wall_hours * 3600 < measured_seconds:
            raise ValueError("canary checkpoint experiment wall time is impossible")
        recomputed = canary_checkpoint(
            median_update_seconds=statistics.median(timing_samples),
            evaluation_seconds=evaluation_seconds,
            experiment_wall_seconds=experiment_wall_hours * 3600,
            nominal_tokens_per_update=nominal_per_update,
            updates=protocol["training"]["updates"],
            evaluations_per_arm=protocol["training"]["updates"] // protocol["training"]["eval_every_updates"],
        )["checkpoint_estimate"]
        if any(
            not isinstance(checkpoint.get(field), (int, float))
            or not math.isclose(checkpoint[field], value, rel_tol=1e-9, abs_tol=1e-9)
            for field, value in recomputed.items()
        ):
            raise ValueError("canary checkpoint estimate differs from metrics and evaluations")
    state_name = "initial_state.pt" if mode == "canary" else "final_state.pt"
    state = torch.load(output / state_name, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"model state artifact is not a state dictionary: {state_name}")
    with torch.device("meta"):
        expected_state = LlamaForCausalLM(build_llama_config(protocol)).state_dict()
    if state.keys() != expected_state.keys() or any(
        not torch.is_tensor(state[name]) or state[name].shape != expected.shape
        for name, expected in expected_state.items()
    ):
        raise ValueError(f"model state artifact does not match the frozen LLaMA configuration: {state_name}")
    if mode == "canary" and _sha256(output / state_name) != summary["initial_state_sha256"]:
        raise ValueError("initial state hash differs from summary")
    return summary


def build_llama_config(protocol: dict) -> LlamaConfig:
    return LlamaConfig(**protocol["model"])


def build_optimizer(model: torch.nn.Module, *, lr: float, head_momentum: float, weight_decay: float = 0.0) -> ScaleOptimizer:
    partition = partition_scale_parameters(model)

    def named(parameters):
        return [(parameter, partition.names[parameter]) for parameter in parameters]

    main = partition.main + partition.secondary if head_momentum == 0 else partition.main
    secondary = [] if head_momentum == 0 else partition.secondary
    return ScaleOptimizer(
        named(main),
        named(secondary),
        named(partition.one_dimensional),
        lr=lr,
        head_momentum=head_momentum,
        weight_decay=weight_decay,
    )


def token_batches(rows, tokenizer, *, batch_size: int, sequence_length: int):
    texts = []
    for row in rows:
        texts.append(row["text"])
        if len(texts) == batch_size:
            yield tokenizer(texts, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")
            texts = []
    if texts:
        yield tokenizer(texts, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")


def _fake_batches(*, count: int, batch_size: int, sequence_length: int, vocab_size: int):
    generator = torch.Generator().manual_seed(42)
    for _ in range(count):
        input_ids = torch.randint(3, vocab_size, (batch_size, sequence_length), generator=generator)
        yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_batch(batch: dict, device: torch.device, pad_id: int) -> tuple[dict, torch.Tensor, int]:
    moved = {name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}
    labels = moved["input_ids"].clone()
    labels[labels == pad_id] = -100
    return moved, labels, int((batch["input_ids"] != pad_id).sum())


def valid_target_count(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum())


def backward_token_sum(loss: torch.Tensor, labels: torch.Tensor) -> int:
    targets = valid_target_count(labels)
    if targets == 0:
        raise ValueError("batch contains no valid causal targets")
    (loss * targets).backward()
    return targets


def normalize_gradients(parameters, targets: int) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(targets)


@torch.no_grad()
def _evaluate(model, batches, *, device: torch.device, pad_id: int, token_limit: int | None) -> dict:
    model.eval()
    total_nll = 0.0
    target_tokens = 0
    tokens = 0
    batch_count = 0
    digest = TokenDigest()
    started = time.perf_counter()
    for batch in batches:
        digest.update(batch["input_ids"])
        moved, labels, nonpad = _model_batch(batch, device, pad_id)
        targets = valid_target_count(labels)
        total_nll += float(model(**moved, labels=labels).loss) * targets
        target_tokens += targets
        tokens += nonpad
        batch_count += 1
        if token_limit is not None and tokens >= token_limit:
            break
    _sync(device)
    model.train()
    if batch_count == 0 or target_tokens == 0:
        raise RuntimeError("evaluation produced no batches")
    loss = total_nll / target_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "tokens": tokens,
        "target_tokens": target_tokens,
        "batches": batch_count,
        "data_digest": digest.hexdigest(),
        "seconds": time.perf_counter() - started,
    }


def _train(
    *,
    model,
    optimizer,
    scheduler,
    train_batches,
    eval_batches,
    output: Path,
    updates: int,
    accumulation: int,
    nominal_tokens_per_microbatch: int,
    device: torch.device,
    pad_id: int,
    eval_steps: set[int],
    eval_token_limit: int | None,
) -> dict:
    digest = TokenDigest()
    update_times = []
    evaluations = []
    nonpad_tokens = 0
    metrics_path = output / "metrics.jsonl"
    eval_path = output / "evaluations.jsonl"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with metrics_path.open("w") as metrics, eval_path.open("w") as eval_metrics:
        for update in range(1, updates + 1):
            _sync(device)
            started = time.perf_counter()
            applied_lr = optimizer.param_groups[0]["lr"]
            update_nll = 0.0
            update_targets = 0
            update_nonpad = 0
            for _ in range(accumulation):
                batch = next(train_batches)
                digest.update(batch["input_ids"])
                moved, labels, count = _model_batch(batch, device, pad_id)
                loss = model(**moved, labels=labels).loss
                targets = backward_token_sum(loss, labels)
                update_nll += float(loss.detach()) * targets
                update_targets += targets
                update_nonpad += count
            normalize_gradients(model.parameters(), update_targets)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            elapsed = time.perf_counter() - started
            update_times.append(elapsed)
            nonpad_tokens += update_nonpad
            metrics.write(
                json.dumps(
                    {
                        "update": update,
                        "nominal_tokens": update * nominal_tokens_per_microbatch * accumulation,
                        "train_loss": update_nll / update_targets,
                        "lr": applied_lr,
                        "update_seconds": elapsed,
                        "nominal_tokens_per_second": nominal_tokens_per_microbatch * accumulation / elapsed,
                        "nonpad_tokens": nonpad_tokens,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    }
                )
                + "\n"
            )
            metrics.flush()
            if update in eval_steps:
                evaluation = _evaluate(
                    model,
                    eval_batches(),
                    device=device,
                    pad_id=pad_id,
                    token_limit=eval_token_limit,
                )
                evaluation["update"] = update
                evaluations.append(evaluation)
                eval_metrics.write(json.dumps(evaluation) + "\n")
                eval_metrics.flush()
    return {
        "data_digest": digest.hexdigest(),
        "data_tokens": digest.tokens,
        "nonpad_tokens": nonpad_tokens,
        "update_times": update_times,
        "evaluations": evaluations,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }


def run_smoke(output_dir: str | Path, arm: str) -> dict:
    if arm not in {"no_momentum", "head_momentum"}:
        raise ValueError(f"unknown arm: {arm}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    initial_state = output / "initial_state.pt"
    torch.save(model.state_dict(), initial_state)
    partition = partition_scale_parameters(model)
    manifest_metadata = {
        "arm": arm,
        "code_sha256": _code_sha256(),
        "secondary_parameters": [partition.names[parameter] for parameter in partition.secondary],
        "momentum_parameters": [partition.names[parameter] for parameter in partition.secondary]
        if arm == "head_momentum"
        else [],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    optimizer = build_optimizer(model, lr=1e-3, head_momentum=0.9 if arm == "head_momentum" else 0.0)
    scheduler = LambdaLR(optimizer, lambda step: cosine_multiplier(step, warmup=1, total=2, minimum=0.1))
    training = _train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_batches=iter(_fake_batches(count=4, batch_size=2, sequence_length=8, vocab_size=32)),
        eval_batches=lambda: _fake_batches(count=2, batch_size=2, sequence_length=8, vocab_size=32),
        output=output,
        updates=2,
        accumulation=2,
        nominal_tokens_per_microbatch=16,
        device=torch.device("cpu"),
        pad_id=0,
        eval_steps={2},
        eval_token_limit=None,
    )
    final = training["evaluations"][-1]
    summary = {
        "status": "complete",
        "arm": arm,
        "updates": 2,
        "data_tokens": training["data_tokens"],
        "data_digest": training["data_digest"],
        "initial_state_sha256": _sha256(initial_state),
        "code_sha256": _code_sha256(),
        "final_eval_loss": final["loss"],
        "final_eval_perplexity": final["perplexity"],
        "eval_perplexities": [row["perplexity"] for row in training["evaluations"]],
    }
    _write_json(output / "summary.json", summary)
    finalize_manifest(
        output,
        manifest_metadata,
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )
    return summary


def _c4_sources(protocol: dict):
    from datasets import load_dataset

    inputs = protocol["inputs"]
    training = protocol["training"]
    tokenizer = AutoTokenizer.from_pretrained(inputs["tokenizer"], revision=inputs["tokenizer_revision"])
    if tokenizer.pad_token_id != inputs["tokenizer_pad_id"]:
        raise ValueError(f"tokenizer pad id changed: {tokenizer.pad_token_id}")

    def batches(split: str):
        rows = load_dataset(
            inputs["dataset"],
            inputs["dataset_config"],
            split=split,
            streaming=True,
            revision=inputs["dataset_revision"],
        ).shuffle(seed=inputs["shuffle_seed"])
        return token_batches(
            rows,
            tokenizer,
            batch_size=training["micro_batch_size"],
            sequence_length=training["sequence_length"],
        )

    return batches


def build_run_summary(
    *,
    arm: str,
    mode: str,
    updates: int,
    training_result: dict,
    initial_state_sha256: str,
    protocol_sha256: str,
    code_sha256: str,
    commit_sha: str,
    model_parameter_bytes: int,
    optimizer_state_bytes: int,
) -> dict:
    evaluations = training_result["evaluations"]
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    final = evaluations[-1]
    return {
        "status": "complete",
        "mode": mode,
        "arm": arm,
        "updates": updates,
        "data_tokens": training_result["data_tokens"],
        "nonpad_tokens": training_result["nonpad_tokens"],
        "data_digest": training_result["data_digest"],
        "initial_state_sha256": initial_state_sha256,
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_sha256,
        "commit_sha": commit_sha,
        "final_eval_loss": final["loss"],
        "final_eval_perplexity": final["perplexity"],
        "eval_updates": [row["update"] for row in evaluations],
        "eval_losses": [row["loss"] for row in evaluations],
        "eval_perplexities": [row["perplexity"] for row in evaluations],
        "eval_input_tokens": [row["tokens"] for row in evaluations],
        "eval_target_tokens": [row["target_tokens"] for row in evaluations],
        "eval_data_digests": [row["data_digest"] for row in evaluations],
        "peak_memory_bytes": training_result["peak_memory_bytes"],
        "model_parameter_bytes": model_parameter_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
    }


def validate_canary_decision(
    decision_path: str | Path,
    *,
    protocol_sha256: str,
    code_sha256: str,
    commit_sha: str,
    initial_state_path: str | Path,
    canary_manifest_path: str | Path,
) -> dict:
    decision = json.loads(Path(decision_path).read_text())
    expected = {
        "status": "complete",
        "mode": "canary",
        "arm": "head_momentum",
        "scientific_status": "incomplete_canary",
        "scale_gate": "pass",
        "supervised_single_instance": True,
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_sha256,
        "commit_sha": commit_sha,
        "initial_state_sha256": _sha256(Path(initial_state_path)),
        "canary_manifest_sha256": _sha256(Path(canary_manifest_path)),
    }
    for field, value in expected.items():
        if decision.get(field) != value:
            raise ValueError(f"invalid canary decision {field}: expected {value!r}, got {decision.get(field)!r}")
    digest = decision.get("canary_manifest_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("canary decision has invalid canary_manifest_sha256")
    if decision.get("failed_checks") != []:
        raise ValueError("canary decision contains failed checks")
    return decision


def finalize_canary_evidence(
    canary_run: str | Path,
    protocol_path: str | Path,
    *,
    provider_wall_hours: float,
    spend_inr: float,
) -> dict:
    canary_run = Path(canary_run)
    protocol_path = Path(protocol_path)
    summary = verify_artifacts(canary_run, protocol_path=protocol_path)
    protocol = load_protocol(protocol_path)
    return finalize_supervised_canary(
        summary,
        protocol,
        protocol_sha256=_sha256(protocol_path),
        code_sha256=_code_sha256(),
        provider_wall_hours=provider_wall_hours,
        spend_inr=spend_inr,
        canary_manifest_sha256=_sha256(canary_run / "manifest.json"),
    )


def verify_full_canary_evidence(
    decision_path: str | Path,
    *,
    canary_run: str | Path,
    protocol_path: str | Path,
    initial_state_path: str | Path,
    commit_sha: str,
) -> dict:
    canary_run = Path(canary_run)
    protocol_path = Path(protocol_path)
    raw_decision = json.loads(Path(decision_path).read_text())
    expected = finalize_canary_evidence(
        canary_run,
        protocol_path,
        provider_wall_hours=raw_decision.get("provider_wall_hours"),
        spend_inr=raw_decision.get("canary_spend_inr"),
    )
    decision = validate_canary_decision(
        decision_path,
        protocol_sha256=_sha256(protocol_path),
        code_sha256=_code_sha256(),
        commit_sha=commit_sha,
        initial_state_path=initial_state_path,
        canary_manifest_path=canary_run / "manifest.json",
    )
    if decision != expected:
        raise ValueError("canary decision differs from verified canary artifacts")
    return expected


def canary_timing_samples(update_times: list[float], protocol: dict) -> list[float]:
    canary = protocol["canary"]
    if len(update_times) != canary["updates"]:
        raise ValueError("canary timing samples do not cover every canary update")
    start = canary["timing_window_start_update"] - 1
    samples = update_times[start : canary["updates"]]
    if len(samples) != canary["timing_window_samples"]:
        raise ValueError("canary timing samples do not match the frozen window")
    return samples


def run_experiment(
    output_dir: str | Path,
    *,
    arm: str,
    mode: str,
    protocol_path: str | Path,
    initial_state_path: str | Path | None = None,
    canary_decision_path: str | Path | None = None,
    canary_run_path: str | Path | None = None,
) -> dict:
    if arm not in {"no_momentum", "head_momentum"}:
        raise ValueError(f"unknown arm: {arm}")
    if mode not in {"canary", "full"}:
        raise ValueError(f"unknown mode: {mode}")
    if mode == "full" and initial_state_path is None:
        raise ValueError("full runs require the canary-generated initial state")
    if mode == "full" and any(value is None for value in (canary_decision_path, canary_run_path)):
        raise ValueError("full runs require the bound canary run and canary decision")

    protocol_file = Path(protocol_path)
    protocol = load_protocol(protocol_file)
    protocol_hash = _sha256(protocol_file)
    code_hash = _code_sha256()
    commit_sha = os.environ.get("SCALE_COMMIT_SHA", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
        raise ValueError("production runs require a 40-character SCALE_COMMIT_SHA provenance value")
    if mode == "full":
        assert canary_decision_path is not None and canary_run_path is not None
        verify_full_canary_evidence(
            canary_decision_path,
            canary_run=canary_run_path,
            protocol_path=protocol_file,
            initial_state_path=initial_state_path,
            commit_sha=commit_sha,
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the production path requires a BF16-capable CUDA GPU")

    experiment_started = time.perf_counter()
    training = protocol["training"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.manual_seed(training["seed"])
    torch.cuda.manual_seed_all(training["seed"])
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    model = LlamaForCausalLM(build_llama_config(protocol))
    if initial_state_path is None:
        initial_state = output / "initial_state.pt"
        torch.save(model.state_dict(), initial_state)
    else:
        initial_state = Path(initial_state_path)
        model.load_state_dict(torch.load(initial_state, map_location="cpu", weights_only=True), strict=True)
    initial_hash = _sha256(initial_state)
    model.to(device=device, dtype=torch.bfloat16)
    model.train()

    partition = partition_scale_parameters(model)
    secondary_names = [partition.names[parameter] for parameter in partition.secondary]
    if secondary_names != ["lm_head.weight"]:
        raise RuntimeError(f"unexpected secondary parameters: {secondary_names}")
    head_momentum = protocol["arms"][arm]["head_momentum"]
    optimizer = build_optimizer(
        model,
        lr=training["learning_rate"],
        head_momentum=head_momentum,
        weight_decay=training["weight_decay"],
    )
    scheduler = LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(
            step,
            warmup=training["warmup_updates"],
            total=training["updates"],
            minimum=training["min_lr_ratio"],
        ),
    )
    sources = _c4_sources(protocol)
    updates = protocol["canary"]["updates"] if mode == "canary" else training["updates"]
    eval_steps = {updates} if mode == "canary" else set(range(training["eval_every_updates"], updates + 1, training["eval_every_updates"]))

    import datasets
    import transformers

    manifest_metadata = {
        "arm": arm,
        "mode": mode,
        "protocol_sha256": protocol_hash,
        "code_sha256": code_hash,
        "commit_sha": commit_sha,
        "initial_state_sha256": initial_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "secondary_parameters": secondary_names,
        "momentum_parameters": secondary_names if head_momentum > 0 else [],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }

    training_result = _train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_batches=iter(sources("train")),
        eval_batches=lambda: sources("validation"),
        output=output,
        updates=updates,
        accumulation=training["gradient_accumulation"],
        nominal_tokens_per_microbatch=training["micro_batch_size"] * training["sequence_length"],
        device=device,
        pad_id=protocol["inputs"]["tokenizer_pad_id"],
        eval_steps=eval_steps,
        eval_token_limit=training["eval_tokens"],
    )
    model_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    summary = build_run_summary(
        arm=arm,
        mode=mode,
        updates=updates,
        training_result=training_result,
        initial_state_sha256=initial_hash,
        protocol_sha256=protocol_hash,
        code_sha256=code_hash,
        commit_sha=commit_sha,
        model_parameter_bytes=model_parameter_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
    )
    if mode == "canary":
        steady_times = canary_timing_samples(training_result["update_times"], protocol)
        final = training_result["evaluations"][-1]
        summary.update(canary_checkpoint(
            median_update_seconds=statistics.median(steady_times),
            evaluation_seconds=final["seconds"],
            experiment_wall_seconds=time.perf_counter() - experiment_started,
            nominal_tokens_per_update=training["global_batch_size"] * training["sequence_length"],
            updates=training["updates"],
            evaluations_per_arm=training["updates"] // training["eval_every_updates"],
        ))
    else:
        torch.save(model.state_dict(), output / "final_state.pt")
    _write_json(output / "summary.json", summary)
    required = ["summary.json", "metrics.jsonl", "evaluations.jsonl"]
    required.append("initial_state.pt" if mode == "canary" else "final_state.pt")
    finalize_manifest(output, manifest_metadata, required)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--arm", choices=["no_momentum", "head_momentum"], required=True)
    smoke.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--arm", choices=["no_momentum", "head_momentum"], required=True)
    train.add_argument("--mode", choices=["canary", "full"], required=True)
    train.add_argument("--protocol", type=Path, default=_DEFAULT_PROTOCOL)
    train.add_argument("--initial-state", type=Path)
    train.add_argument("--canary-decision", type=Path)
    train.add_argument("--canary-run", type=Path)
    train.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("finalize-canary")
    finalize.add_argument("--canary-run", type=Path, required=True)
    finalize.add_argument("--protocol", type=Path, default=_DEFAULT_PROTOCOL)
    finalize.add_argument("--provider-wall-hours", type=float, required=True)
    finalize.add_argument("--spend-inr", type=float, required=True)
    finalize.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--no-momentum", type=Path, required=True)
    compare.add_argument("--head-momentum", type=Path, required=True)
    compare.add_argument("--protocol", type=Path, default=_DEFAULT_PROTOCOL)
    compare.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "smoke":
        run_smoke(args.output, args.arm)
    elif args.command == "train":
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        commit_sha = os.environ.get("SCALE_COMMIT_SHA", "")
        if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
            raise ValueError("production runs require a 40-character SCALE_COMMIT_SHA provenance value")
        if args.mode == "full":
            if args.initial_state is None:
                raise ValueError("full runs require the canary-generated initial state")
            if any(value is None for value in (args.canary_decision, args.canary_run)):
                raise ValueError("full runs require the bound canary run and canary decision")
        try:
            run_experiment(
                args.output,
                arm=args.arm,
                mode=args.mode,
                protocol_path=args.protocol,
                initial_state_path=args.initial_state,
                canary_decision_path=args.canary_decision,
                canary_run_path=args.canary_run,
            )
        except Exception as error:
            args.output.mkdir(parents=True, exist_ok=True)
            _write_json(args.output / "failure.json", {"status": "failed", "error": repr(error)})
            raise
    elif args.command == "finalize-canary":
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        result = finalize_canary_evidence(
            args.canary_run,
            args.protocol,
            provider_wall_hours=args.provider_wall_hours,
            spend_inr=args.spend_inr,
        )
        _write_json(args.output, result)
    else:
        protocol = load_protocol(args.protocol)
        no_momentum = verify_artifacts(args.no_momentum, protocol_path=args.protocol)
        head_momentum = verify_artifacts(args.head_momentum, protocol_path=args.protocol)
        result = compare_summaries(
            no_momentum,
            head_momentum,
            protocol=protocol,
            protocol_sha256=_sha256(args.protocol),
            code_sha256=_code_sha256(),
        )
        _write_json(args.output, result)


if __name__ == "__main__":
    main()
