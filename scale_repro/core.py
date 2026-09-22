import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ParameterPartition:
    main: list[nn.Parameter]
    secondary: list[nn.Parameter]
    one_dimensional: list[nn.Parameter]
    names: dict[nn.Parameter, str]


def partition_scale_parameters(model: nn.Module) -> ParameterPartition:
    names = {parameter: name for name, parameter in model.named_parameters()}
    main: list[nn.Parameter] = []
    main_ids: set[int] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and any(
            key in module_name for key in ("attn", "mlp", "attention", "embed_tokens")
        ):
            main.append(module.weight)
            main_ids.add(id(module.weight))

    secondary: list[nn.Parameter] = []
    one_dimensional: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in main_ids:
            continue
        (one_dimensional if parameter.ndim == 1 else secondary).append(parameter)
    return ParameterPartition(main, secondary, one_dimensional, names)


class ScaleOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        main: list[tuple[nn.Parameter, str]],
        secondary: list[tuple[nn.Parameter, str]],
        one_dimensional: list[tuple[nn.Parameter, str]],
        *,
        lr: float,
        head_momentum: float,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        typed = [(parameter, "main", name) for parameter, name in main]
        typed += [(parameter, "secondary", name) for parameter, name in secondary]
        typed += [(parameter, "one_dimensional", name) for parameter, name in one_dimensional]
        super().__init__(
            [parameter for parameter, _, _ in typed],
            dict(lr=lr, head_momentum=head_momentum, weight_decay=weight_decay, adam_betas=adam_betas, eps=eps),
        )
        for parameter, kind, name in typed:
            self.state[parameter].update(kind=kind, name=name)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if state["kind"] in {"main", "secondary"}:
                    if state["kind"] == "secondary":
                        moment = state.setdefault("moment1", torch.zeros_like(gradient))
                        moment.lerp_(gradient, 1 - group["head_momentum"])
                        gradient = moment
                    dimension = 0 if "embed_tokens" in state["name"] else 1
                    update = gradient / gradient.square().mean(dim=dimension, keepdim=True).sqrt().clamp_min(group["eps"])
                    parameter.mul_(1 - lr * group["weight_decay"])
                    parameter.add_(update, alpha=-lr)
                    continue

                beta1, beta2 = group["adam_betas"]
                state["step"] = state.get("step", 0) + 1
                first = state.setdefault("moment1", torch.zeros_like(gradient))
                second = state.setdefault("moment2", torch.zeros_like(gradient))
                first.lerp_(gradient, 1 - beta1)
                second.lerp_(gradient.square(), 1 - beta2)
                scale = (1 - beta1 ** state["step"]) / (1 - beta2 ** state["step"]) ** 0.5
                parameter.mul_(1 - lr * group["weight_decay"])
                parameter.add_(first / (second.sqrt() + group["eps"]), alpha=-lr / scale)
        return loss


class TokenDigest:
    def __init__(self):
        self._hash = hashlib.sha256()
        self.tokens = 0

    def update(self, token_ids: torch.Tensor) -> None:
        values = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        self._hash.update(struct.pack("<Q", values.numel()))
        self._hash.update(values.numpy().tobytes())
        self.tokens += values.numel()

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def project_full_run_hours(
    *,
    median_update_seconds: float,
    evaluation_seconds: float,
    updates: int,
    evaluations_per_arm: int,
    arms: int,
) -> float:
    return arms * (median_update_seconds * updates + evaluation_seconds * evaluations_per_arm) / 3600


def full_run_timeout_budget(projected_training_and_evaluation_hours: float, protocol: dict) -> tuple[int, int]:
    requested = math.ceil(projected_training_and_evaluation_hours * 0.5 * 1.1 * 3600 - 1e-9)
    lifecycle = protocol["lifecycle"]
    available = math.floor(lifecycle["full_run_timeout_hours_per_arm"] * 3600) - lifecycle["recovery_cleanup_reserve_seconds"]
    return requested, available


def canary_checkpoint(
    *,
    median_update_seconds: float,
    evaluation_seconds: float,
    experiment_wall_seconds: float,
    nominal_tokens_per_update: int,
    updates: int,
    evaluations_per_arm: int,
) -> dict:
    projected_training_hours = project_full_run_hours(
        median_update_seconds=median_update_seconds,
        evaluation_seconds=evaluation_seconds,
        updates=updates,
        evaluations_per_arm=evaluations_per_arm,
        arms=2,
    )
    return {
        "scientific_status": "incomplete_canary",
        "scale_gate": "pending_lifecycle",
        "checkpoint_estimate": {
            "median_update_seconds": median_update_seconds,
            "evaluation_seconds": evaluation_seconds,
            "experiment_wall_hours": experiment_wall_seconds / 3600,
            "aggregate_training_tokens_per_second": nominal_tokens_per_update / median_update_seconds,
            "projected_training_and_evaluation_hours": projected_training_hours,
        },
    }


def finalize_supervised_canary(
    checkpoint: dict,
    protocol: dict,
    *,
    protocol_sha256: str,
    code_sha256: str,
    provider_wall_hours: float,
    spend_inr: float,
    canary_manifest_sha256: str,
) -> dict:
    training = protocol["training"]
    canary = protocol["canary"]
    expected = {
        "status": "complete",
        "mode": "canary",
        "arm": "head_momentum",
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_sha256,
        "updates": canary["updates"],
        "data_tokens": canary["updates"] * training["global_batch_size"] * training["sequence_length"],
        "scientific_status": "incomplete_canary",
        "scale_gate": "pending_lifecycle",
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"invalid {field}: expected {value!r}, got {checkpoint.get(field)!r}")
    histories = (
        checkpoint.get("eval_updates", []),
        checkpoint.get("eval_losses", []),
        checkpoint.get("eval_perplexities", []),
        checkpoint.get("eval_input_tokens", []),
        checkpoint.get("eval_target_tokens", []),
        checkpoint.get("eval_data_digests", []),
    )
    if any(len(history) != 1 for history in histories):
        raise ValueError("canary must contain exactly one evaluation")
    if checkpoint["eval_updates"] != [canary["updates"]]:
        raise ValueError("canary evaluation update must equal the final canary update")
    if checkpoint["eval_input_tokens"][0] < training["eval_tokens"]:
        raise ValueError("canary evaluation token coverage is below the frozen protocol")
    if not 0 < checkpoint["eval_target_tokens"][0] <= checkpoint["eval_input_tokens"][0]:
        raise ValueError("canary evaluation target-token count is invalid")
    values = [
        checkpoint.get("final_eval_loss"),
        checkpoint.get("final_eval_perplexity"),
        *checkpoint["eval_losses"],
        *checkpoint["eval_perplexities"],
    ]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("canary evaluation values must be finite")
    if not math.isclose(checkpoint["final_eval_perplexity"], checkpoint["eval_perplexities"][-1], rel_tol=1e-12):
        raise ValueError("canary final perplexity does not match its history")
    if not math.isclose(checkpoint["final_eval_loss"], checkpoint["eval_losses"][-1], rel_tol=1e-12):
        raise ValueError("canary final loss does not match its history")
    if not math.isclose(math.exp(checkpoint["final_eval_loss"]), checkpoint["final_eval_perplexity"], rel_tol=1e-6):
        raise ValueError("canary final loss and perplexity are inconsistent")
    if not isinstance(canary_manifest_sha256, str) or len(canary_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in canary_manifest_sha256
    ):
        raise ValueError("invalid canary_manifest_sha256")
    if not isinstance(provider_wall_hours, (int, float)) or not math.isfinite(provider_wall_hours) or provider_wall_hours <= 0:
        raise ValueError("provider wall time must be positive and finite")
    if not isinstance(spend_inr, (int, float)) or not math.isfinite(spend_inr) or spend_inr < 0:
        raise ValueError("canary spend must be finite and non-negative")

    estimate = checkpoint["checkpoint_estimate"]
    estimate_fields = (
        "median_update_seconds",
        "evaluation_seconds",
        "experiment_wall_hours",
        "aggregate_training_tokens_per_second",
        "projected_training_and_evaluation_hours",
    )
    if any(
        not isinstance(estimate.get(field), (int, float)) or not math.isfinite(estimate[field]) or estimate[field] <= 0
        for field in estimate_fields
    ):
        raise ValueError("checkpoint estimate values must be positive and finite")
    if provider_wall_hours < estimate["experiment_wall_hours"]:
        raise ValueError("provider wall time cannot be below measured experiment wall time")
    timed_canary_hours = (
        estimate["median_update_seconds"] * canary["updates"] + estimate["evaluation_seconds"]
    ) / 3600
    overhead_per_run = max(0.0, provider_wall_hours - timed_canary_hours)
    projected_full_hours = estimate["projected_training_and_evaluation_hours"] + 2 * overhead_per_run
    full_timeout_seconds, maximum_full_timeout_seconds = full_run_timeout_budget(
        estimate["projected_training_and_evaluation_hours"], protocol
    )
    price = protocol["budget"]["a30_on_demand_inr_per_hour_snapshot"]
    projected_full_spend = projected_full_hours * price
    projected_total_spend = spend_inr + projected_full_spend
    checks = {
        "canary_wall": provider_wall_hours <= canary["max_wall_hours"],
        "canary_spend": spend_inr <= canary["max_spend_inr"],
        "throughput": estimate["aggregate_training_tokens_per_second"]
        >= canary["minimum_aggregate_training_tokens_per_second"],
        "full_hours": projected_full_hours <= canary["max_projected_full_hours"],
        "full_timeout": full_timeout_seconds <= maximum_full_timeout_seconds,
        "full_spend": projected_full_spend <= protocol["budget"]["full_projection_cap_inr"],
        "total_spend": projected_total_spend <= protocol["budget"]["hard_cap_inr"],
    }
    return {
        **checkpoint,
        "scale_gate": "pass" if all(checks.values()) else "fail",
        "supervised_single_instance": True,
        "provider_wall_hours": provider_wall_hours,
        "canary_spend_inr": spend_inr,
        "canary_manifest_sha256": canary_manifest_sha256,
        "projected_full_hours": projected_full_hours,
        "full_run_timeout_seconds_per_arm": full_timeout_seconds,
        "projected_full_spend_inr": projected_full_spend,
        "projected_total_spend_inr": projected_total_spend,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def cosine_multiplier(step: int, *, warmup: int, total: int, minimum: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def classify_outcome(
    final_no_momentum_ppl: float,
    final_head_momentum_ppl: float,
    head_momentum_history: list[float],
    no_momentum_history: list[float],
    outcome: dict,
) -> str:
    delta = final_no_momentum_ppl - final_head_momentum_ppl
    count = outcome["final_evaluations_for_direction"]
    better = sum(head < plain for head, plain in zip(head_momentum_history[-count:], no_momentum_history[-count:]))
    if delta >= outcome["strong_reproduction_min_delta"] and better >= outcome["minimum_better_final_evaluations"]:
        return "strong_reproduction"
    if abs(delta) <= outcome["null_max_absolute_delta"] or delta < 0:
        return "challenge_or_null"
    return "intermediate"


def _validate_full_summary(
    summary: dict,
    *,
    arm: str,
    protocol: dict,
    protocol_sha256: str,
    code_sha256: str,
) -> None:
    training = protocol["training"]
    evaluation_count = training["updates"] // training["eval_every_updates"]
    expected = {
        "status": "complete",
        "mode": "full",
        "arm": arm,
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_sha256,
        "updates": training["updates"],
        "data_tokens": training["nominal_tokens"],
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"invalid {field}: expected {value!r}, got {summary.get(field)!r}")
    commit_sha = summary.get("commit_sha")
    if not isinstance(commit_sha, str) or len(commit_sha) != 40 or any(character not in "0123456789abcdef" for character in commit_sha):
        raise ValueError("invalid commit_sha")
    histories = (
        summary.get("eval_updates", []),
        summary.get("eval_losses", []),
        summary.get("eval_perplexities", []),
        summary.get("eval_input_tokens", []),
        summary.get("eval_target_tokens", []),
        summary.get("eval_data_digests", []),
    )
    if any(len(history) != evaluation_count for history in histories):
        raise ValueError(f"evaluation count must be {evaluation_count}")
    expected_updates = list(range(training["eval_every_updates"], training["updates"] + 1, training["eval_every_updates"]))
    if summary["eval_updates"] != expected_updates:
        raise ValueError(f"invalid eval_updates: expected {expected_updates}")
    if any(tokens < training["eval_tokens"] for tokens in summary["eval_input_tokens"]):
        raise ValueError("evaluation token coverage is below the frozen protocol")
    if any(targets <= 0 or targets > inputs for targets, inputs in zip(summary["eval_target_tokens"], summary["eval_input_tokens"])):
        raise ValueError("evaluation target-token counts are invalid")
    if len(set(summary["eval_data_digests"])) != 1:
        raise ValueError("evaluation data digests differ within a run")
    values = [summary.get("final_eval_loss"), summary.get("final_eval_perplexity"), *summary["eval_losses"], *summary["eval_perplexities"]]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("evaluation values must be finite")
    if not math.isclose(summary["final_eval_perplexity"], summary["eval_perplexities"][-1], rel_tol=1e-12):
        raise ValueError("final_eval_perplexity does not match the final history entry")
    if not math.isclose(summary["final_eval_loss"], summary["eval_losses"][-1], rel_tol=1e-12):
        raise ValueError("final_eval_loss does not match the final history entry")
    if any(not math.isclose(math.exp(loss), perplexity, rel_tol=1e-6) for loss, perplexity in zip(summary["eval_losses"], summary["eval_perplexities"])):
        raise ValueError("evaluation loss and perplexity histories are inconsistent")
    if not math.isclose(math.exp(summary["final_eval_loss"]), summary["final_eval_perplexity"], rel_tol=1e-6):
        raise ValueError("final loss and perplexity are inconsistent")


def compare_summaries(
    no_momentum: dict,
    head_momentum: dict,
    *,
    protocol: dict,
    protocol_sha256: str,
    code_sha256: str,
) -> dict:
    _validate_full_summary(
        no_momentum,
        arm="no_momentum",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        code_sha256=code_sha256,
    )
    _validate_full_summary(
        head_momentum,
        arm="head_momentum",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        code_sha256=code_sha256,
    )
    for field in ("initial_state_sha256", "data_digest", "commit_sha"):
        if no_momentum[field] != head_momentum[field]:
            raise ValueError(f"paired runs have different {field}")
    for field in ("eval_updates", "eval_input_tokens", "eval_target_tokens", "eval_data_digests"):
        if no_momentum[field] != head_momentum[field]:
            raise ValueError(f"paired runs have different {field}")
    delta = no_momentum["final_eval_perplexity"] - head_momentum["final_eval_perplexity"]
    return {
        "outcome": classify_outcome(
            no_momentum["final_eval_perplexity"],
            head_momentum["final_eval_perplexity"],
            head_momentum["eval_perplexities"],
            no_momentum["eval_perplexities"],
            protocol["outcome"],
        ),
        "perplexity_delta": delta,
        "no_momentum_perplexity": no_momentum["final_eval_perplexity"],
        "head_momentum_perplexity": head_momentum["final_eval_perplexity"],
        "initial_state_sha256": no_momentum["initial_state_sha256"],
        "data_digest": no_momentum["data_digest"],
        "eval_data_digest": no_momentum["eval_data_digests"][0],
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_sha256,
        "commit_sha": no_momentum["commit_sha"],
    }


def load_protocol(path: str | Path) -> dict:
    protocol = json.loads(Path(path).read_text())
    if protocol.get("schema_version") != 2:
        raise ValueError("unsupported protocol schema_version")

    def exact_mapping(actual, expected) -> bool:
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items()
        )

    expected_inputs = {
        "dataset": "allenai/c4",
        "dataset_config": "en",
        "dataset_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "tokenizer": "google-t5/t5-base",
        "tokenizer_revision": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
        "tokenizer_pad_id": 0,
        "shuffle_seed": 42,
    }
    if not exact_mapping(protocol.get("inputs"), expected_inputs):
        raise ValueError("inputs differ from the frozen identifiers and revisions")
    expected_model = {
        "vocab_size": 32000,
        "hidden_size": 512,
        "intermediate_size": 1376,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "use_cache": True,
        "tie_word_embeddings": False,
    }
    if not exact_mapping(protocol.get("model"), expected_model):
        raise ValueError("model differs from the frozen configuration")
    expected_arms = {
        "no_momentum": {"head_momentum": 0.0},
        "head_momentum": {"head_momentum": 0.9},
    }
    arms = protocol.get("arms")
    if not isinstance(arms, dict) or arms.keys() != expected_arms.keys() or any(
        not exact_mapping(arms[name], expected) for name, expected in expected_arms.items()
    ):
        raise ValueError("arms differ from the frozen head momenta")
    training = protocol["training"]
    positive_training = (
        "sequence_length",
        "global_batch_size",
        "micro_batch_size",
        "gradient_accumulation",
        "updates",
        "eval_every_updates",
        "eval_tokens",
    )
    if any(not isinstance(training.get(field), int) or training[field] <= 0 for field in positive_training):
        raise ValueError("training counts and intervals must be positive integers")
    learning_rate = training.get("learning_rate")
    min_lr_ratio = training.get("min_lr_ratio")
    weight_decay = training.get("weight_decay")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(min_lr_ratio, (int, float)) or isinstance(min_lr_ratio, bool) or not math.isfinite(min_lr_ratio) or not 0 < min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be finite and inside (0, 1]")
    if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    frozen_optimizer = {
        "seed": 42,
        "dtype": "bfloat16",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "min_lr_ratio": 0.1,
    }
    if any(training.get(field) != value for field, value in frozen_optimizer.items()):
        raise ValueError("training optimizer fields differ from the frozen protocol")
    if not 0 <= training["warmup_updates"] < training["updates"]:
        raise ValueError("warmup_updates must be inside the training run")
    if training["updates"] % training["eval_every_updates"] != 0:
        raise ValueError("evaluation interval must divide training updates")
    if training["micro_batch_size"] * training["gradient_accumulation"] != training["global_batch_size"]:
        raise ValueError("global_batch_size must equal micro_batch_size * gradient_accumulation")
    expected_tokens = training["global_batch_size"] * training["sequence_length"] * training["updates"]
    if training["nominal_tokens"] != expected_tokens:
        raise ValueError("nominal_tokens does not match batch, sequence length, and updates")

    canary = protocol["canary"]
    if canary["updates"] <= training["warmup_updates"]:
        raise ValueError("canary must include post-warmup updates")
    if canary["timing_window_start_update"] != training["warmup_updates"] + 1:
        raise ValueError("canary timing window must start after warmup")
    if canary["timing_window_samples"] != canary["updates"] - training["warmup_updates"]:
        raise ValueError("canary timing sample count is inconsistent")
    for field in ("max_wall_hours", "max_spend_inr", "max_projected_full_hours", "minimum_aggregate_training_tokens_per_second"):
        value = canary.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("canary limits must be positive and finite")

    budget = protocol["budget"]
    for field in ("hard_cap_inr", "a30_on_demand_inr_per_hour_snapshot", "full_projection_cap_inr", "minimum_reserve_inr"):
        value = budget.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("budget values must be positive and finite")
    reserve = budget["hard_cap_inr"] - canary["max_spend_inr"] - budget["full_projection_cap_inr"]
    if not math.isclose(budget["minimum_reserve_inr"], reserve, abs_tol=1e-9):
        raise ValueError("budget reserve is inconsistent")

    timeout = protocol["lifecycle"].get("full_run_timeout_hours_per_arm")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("full-run timeout must be positive and finite")
    reserve_seconds = protocol["lifecycle"].get("recovery_cleanup_reserve_seconds")
    if not isinstance(reserve_seconds, int) or not 0 < reserve_seconds < canary["max_wall_hours"] * 3600:
        raise ValueError("recovery and cleanup reserve must fit inside the paid wall limit")
    outcome = protocol["outcome"]
    evaluation_count = training["updates"] // training["eval_every_updates"]
    direction_count = outcome.get("final_evaluations_for_direction")
    minimum_better = outcome.get("minimum_better_final_evaluations")
    if not isinstance(direction_count, int) or not 0 < direction_count <= evaluation_count:
        raise ValueError("final direction window is impossible")
    if not isinstance(minimum_better, int) or not 0 <= minimum_better <= direction_count:
        raise ValueError("minimum better evaluations is impossible")
    for field in ("strong_reproduction_min_delta", "null_max_absolute_delta"):
        value = outcome.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("outcome thresholds must be finite and non-negative")
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != "e78fac9163b98927f2f4326ec23a9f6f91ecbbf95f4b762fb5932dd2c33891ad":
        raise ValueError("protocol differs from the frozen paid-run protocol")
    return protocol
