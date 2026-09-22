import json
import math
import re
from pathlib import Path

import pytest
import torch
from torch import nn

from scale_repro.core import (
    ScaleOptimizer,
    TokenDigest,
    canary_checkpoint,
    classify_outcome,
    compare_summaries,
    cosine_multiplier,
    finalize_supervised_canary,
    load_protocol,
    partition_scale_parameters,
    project_full_run_hours,
)


ROOT = Path(__file__).parents[1]


def test_frozen_protocol_is_internally_consistent():
    protocol = load_protocol(ROOT / "protocol.json")
    training = protocol["training"]
    canary = protocol["canary"]

    assert training["global_batch_size"] == 512
    assert training["micro_batch_size"] * training["gradient_accumulation"] == 512
    assert training["nominal_tokens"] == 512 * 256 * 11_000
    assert canary["updates"] > training["warmup_updates"]
    assert canary["timing_window_start_update"] == training["warmup_updates"] + 1
    assert canary["timing_window_samples"] == canary["updates"] - training["warmup_updates"]
    assert protocol["budget"]["hard_cap_inr"] == 1_200.0
    assert canary["max_spend_inr"] == 120.0
    assert protocol["budget"]["minimum_reserve_inr"] == pytest.approx(
        protocol["budget"]["hard_cap_inr"]
        - canary["max_spend_inr"]
        - protocol["budget"]["full_projection_cap_inr"]
    )
    assert protocol["lifecycle"]["full_run_timeout_hours_per_arm"] == 10.0
    assert protocol["lifecycle"]["recovery_cleanup_reserve_seconds"] == 900
    assert protocol["lifecycle"]["destroy_after_download"] is True


def test_authoritative_documents_match_amended_canary_scale():
    expected = json.loads((ROOT / "protocol.json").read_text())["canary"]["updates"]
    patterns = {
        "GOAL.md": r"paid ([\d,]+)-update A30 canary",
        "PROTOCOL.md": r"amended canary runs ([\d,]+) optimizer updates",
        "RUNBOOK.md": r"([\d,]+)-update canary",
    }
    for filename, pattern in patterns.items():
        match = re.search(pattern, (ROOT / filename).read_text())
        assert match and int(match.group(1).replace(",", "")) == expected


def test_runbook_enforces_the_canary_derived_full_arm_timeout():
    runbook = (ROOT / "RUNBOOK.md").read_text()

    assert "FULL_TIMEOUT_SECONDS=" in runbook
    assert runbook.count('timeout --signal=TERM --kill-after=5m "$FULL_TIMEOUT_SECONDS"') == 2


def test_protocol_rejects_pre_warmup_canary(tmp_path):
    raw = json.loads((ROOT / "protocol.json").read_text())
    raw["canary"]["updates"] = raw["training"]["warmup_updates"]
    path = tmp_path / "bad-canary.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="canary"):
        load_protocol(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update(schema_version=99),
        lambda raw: raw["training"].update(eval_every_updates=0),
        lambda raw: raw["training"].update(sequence_length=128, nominal_tokens=720_896_000),
        lambda raw: raw["canary"].update(updates=1300, timing_window_samples=200),
        lambda raw: raw["training"].update(learning_rate=-0.001),
        lambda raw: raw["training"].update(min_lr_ratio=2.0),
        lambda raw: raw["training"].update(dtype="float32"),
        lambda raw: raw["training"].update(scheduler="linear"),
        lambda raw: raw["arms"]["head_momentum"].update(head_momentum=9.0),
        lambda raw: raw["model"].update(hidden_size=513),
        lambda raw: raw["model"].update(use_cache=1),
        lambda raw: raw["inputs"].update(dataset_revision="main"),
        lambda raw: raw["inputs"].update(tokenizer_pad_id=False),
        lambda raw: raw["budget"].update(a30_on_demand_inr_per_hour_snapshot=0),
        lambda raw: raw["lifecycle"].update(full_run_timeout_hours_per_arm=0),
        lambda raw: raw["outcome"].update(minimum_better_final_evaluations=5, final_evaluations_for_direction=4),
    ],
)
def test_protocol_rejects_invalid_operational_invariants(tmp_path, mutation):
    raw = json.loads((ROOT / "protocol.json").read_text())
    mutation(raw)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError):
        load_protocol(path)


def test_protocol_rejects_mismatched_nominal_tokens(tmp_path):
    raw = json.loads((ROOT / "protocol.json").read_text())
    raw["training"]["nominal_tokens"] += 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="nominal_tokens"):
        load_protocol(path)


class TinyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.self_attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Linear(4, 4, bias=False)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_parameter_partition_limits_momentum_to_lm_head():
    model = TinyLlama()
    partition = partition_scale_parameters(model)

    assert partition.names[model.lm_head.weight] == "lm_head.weight"
    assert [id(parameter) for parameter in partition.secondary] == [id(model.lm_head.weight)]
    main_ids = {id(parameter) for parameter in partition.main}
    assert id(model.embed_tokens.weight) in main_ids
    assert id(model.self_attn.weight) in main_ids
    assert id(model.mlp.weight) in main_ids
    one_dimensional_ids = {id(parameter) for parameter in partition.one_dimensional}
    assert id(model.norm.weight) in one_dimensional_ids
    assert id(model.norm.bias) in one_dimensional_ids


def test_matrix_update_uses_rms_normalized_gradient():
    parameter = nn.Parameter(torch.zeros(2, 2))
    optimizer = ScaleOptimizer(
        main=[(parameter, "model.mlp.weight")],
        secondary=[],
        one_dimensional=[],
        lr=0.1,
        head_momentum=0.0,
    )
    parameter.grad = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    optimizer.step()

    expected = -0.1 * torch.tensor([[3.0, 4.0], [0.0, 2.0]]) / torch.tensor([[12.5], [2.0]]).sqrt()
    torch.testing.assert_close(parameter, expected)


def test_head_momentum_changes_second_update_only():
    no_momentum = nn.Parameter(torch.zeros(1, 2))
    momentum = nn.Parameter(torch.zeros(1, 2))
    plain = ScaleOptimizer([], [(no_momentum, "lm_head.weight")], [], lr=0.1, head_momentum=0.0)
    headed = ScaleOptimizer([], [(momentum, "lm_head.weight")], [], lr=0.1, head_momentum=0.9)

    for gradient in (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])):
        no_momentum.grad = gradient.clone()
        momentum.grad = gradient.clone()
        plain.step()
        headed.step()

    assert not torch.equal(no_momentum, momentum)
    assert momentum[0, 0] < no_momentum[0, 0]


def test_one_dimensional_update_matches_adam_without_weight_decay():
    scale_parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    adam_parameter = nn.Parameter(scale_parameter.detach().clone())
    scale = ScaleOptimizer([], [], [(scale_parameter, "model.norm.weight")], lr=0.01, head_momentum=0.0)
    adam = torch.optim.Adam([adam_parameter], lr=0.01)

    for gradient in (torch.tensor([0.5, -0.25]), torch.tensor([-0.1, 0.75])):
        scale_parameter.grad = gradient.clone()
        adam_parameter.grad = gradient.clone()
        scale.step()
        adam.step()

    torch.testing.assert_close(scale_parameter, adam_parameter)


def test_optimizer_closure_runs_with_gradients_enabled():
    parameter = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    optimizer = ScaleOptimizer([(parameter, "model.mlp.weight")], [], [], lr=0.1, head_momentum=0.0)

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert float(loss) == 2.0
    assert parameter.grad is not None


def test_token_digest_detects_data_order():
    first = TokenDigest()
    second = TokenDigest()
    for batch in (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])):
        first.update(batch)
    for batch in (torch.tensor([[5, 6]]), torch.tensor([[1, 2], [3, 4]])):
        second.update(batch)

    assert first.hexdigest() != second.hexdigest()
    assert first.tokens == second.tokens == 6


def test_canary_projection_includes_both_arms_and_evaluations():
    projected = project_full_run_hours(
        median_update_seconds=1.0,
        evaluation_seconds=60.0,
        updates=11_000,
        evaluations_per_arm=11,
        arms=2,
    )

    assert projected == pytest.approx(2 * (11_000 + 11 * 60) / 3600)


def test_supervised_canary_gate_uses_verified_run_and_session_cost():
    protocol = load_protocol(ROOT / "protocol.json")
    manifest_hash = "a" * 64
    checkpoint = {
        "status": "complete",
        "mode": "canary",
        "arm": "head_momentum",
        "protocol_sha256": "protocol",
        "code_sha256": "code",
        "commit_sha": "c" * 40,
        "updates": protocol["canary"]["updates"],
        "data_tokens": protocol["canary"]["updates"]
        * protocol["training"]["global_batch_size"]
        * protocol["training"]["sequence_length"],
        "final_eval_loss": math.log(40.0),
        "final_eval_perplexity": 40.0,
        "eval_updates": [protocol["canary"]["updates"]],
        "eval_losses": [math.log(40.0)],
        "eval_perplexities": [40.0],
        "eval_input_tokens": [protocol["training"]["eval_tokens"]],
        "eval_target_tokens": [protocol["training"]["eval_tokens"] - 1],
        "eval_data_digests": ["eval"],
        **canary_checkpoint(
            median_update_seconds=0.5,
            evaluation_seconds=600.0,
            experiment_wall_seconds=3_600.0,
            nominal_tokens_per_update=131_072,
            updates=11_000,
            evaluations_per_arm=11,
        ),
    }

    def finalize(*, wall=1.2, spend=40.0):
        return finalize_supervised_canary(
            checkpoint,
            protocol,
            protocol_sha256="protocol",
            code_sha256="code",
            provider_wall_hours=wall,
            spend_inr=spend,
            canary_manifest_sha256=manifest_hash,
        )

    assert checkpoint["scientific_status"] == "incomplete_canary"
    assert checkpoint["scale_gate"] == "pending_lifecycle"
    decision = finalize()
    assert decision["scale_gate"] == "pass"
    assert decision["supervised_single_instance"] is True
    timed_canary_hours = (
        checkpoint["checkpoint_estimate"]["median_update_seconds"] * protocol["canary"]["updates"]
        + checkpoint["checkpoint_estimate"]["evaluation_seconds"]
    ) / 3600
    expected_full_hours = checkpoint["checkpoint_estimate"]["projected_training_and_evaluation_hours"] + 2 * (
        1.2 - timed_canary_hours
    )
    assert decision["projected_full_hours"] == pytest.approx(expected_full_hours)
    assert decision["projected_total_spend_inr"] < protocol["budget"]["hard_cap_inr"]

    failed = finalize(spend=121.0)
    assert failed["scale_gate"] == "fail"
    assert "canary_spend" in failed["failed_checks"]

    original_projection = checkpoint["checkpoint_estimate"]["projected_training_and_evaluation_hours"]
    checkpoint["checkpoint_estimate"]["projected_training_and_evaluation_hours"] = 19.344444
    timeout_failure = finalize()
    assert timeout_failure["scale_gate"] == "fail"
    assert "full_timeout" in timeout_failure["failed_checks"]
    checkpoint["checkpoint_estimate"]["projected_training_and_evaluation_hours"] = original_projection

    checkpoint["eval_updates"] = [1]
    with pytest.raises(ValueError, match="evaluation update"):
        finalize()
    checkpoint["eval_updates"] = [protocol["canary"]["updates"]]

    checkpoint["protocol_sha256"] = "wrong"
    with pytest.raises(ValueError, match="protocol_sha256"):
        finalize()

    checkpoint["protocol_sha256"] = "protocol"
    with pytest.raises(ValueError, match="provider wall"):
        finalize(wall=0.5)

    checkpoint["checkpoint_estimate"]["median_update_seconds"] = float("nan")
    with pytest.raises(ValueError, match="checkpoint estimate"):
        finalize()


def test_outcome_uses_frozen_thresholds_and_final_direction():
    outcome = load_protocol(ROOT / "protocol.json")["outcome"]
    assert classify_outcome(40.0, 34.0, [36.0, 35.0, 34.5, 34.0], [39.0, 38.0, 37.0, 36.0], outcome) == "strong_reproduction"
    assert classify_outcome(35.0, 34.5, [34.6, 34.5, 34.4, 34.5], [35.1, 35.0, 34.9, 35.0], outcome) == "challenge_or_null"
    assert classify_outcome(36.0, 33.0, [34.0, 33.5, 33.2, 33.0], [35.0, 34.8, 34.7, 34.5], outcome) == "intermediate"


def test_cosine_schedule_matches_frozen_warmup_and_floor():
    assert cosine_multiplier(0, warmup=1_100, total=11_000, minimum=0.1) == 0.0
    assert cosine_multiplier(1_100, warmup=1_100, total=11_000, minimum=0.1) == 1.0
    assert cosine_multiplier(10_999, warmup=1_100, total=11_000, minimum=0.1) == pytest.approx(0.1, abs=1e-6)


def test_compare_rejects_unpaired_runs_and_classifies_paired_runs():
    protocol = load_protocol(ROOT / "protocol.json")
    eval_count = protocol["training"]["updates"] // protocol["training"]["eval_every_updates"]
    plain = {
        "status": "complete",
        "mode": "full",
        "arm": "no_momentum",
        "protocol_sha256": "protocol",
        "code_sha256": "code",
        "commit_sha": "a" * 40,
        "updates": protocol["training"]["updates"],
        "data_tokens": protocol["training"]["nominal_tokens"],
        "initial_state_sha256": "init",
        "data_digest": "data",
        "final_eval_loss": math.log(40.0),
        "final_eval_perplexity": 40.0,
        "eval_updates": list(range(1_000, protocol["training"]["updates"] + 1, 1_000)),
        "eval_losses": [math.log(45.0)] * (eval_count - 4) + [math.log(value) for value in (39.0, 38.0, 37.0, 40.0)],
        "eval_perplexities": [45.0] * (eval_count - 4) + [39.0, 38.0, 37.0, 40.0],
        "eval_input_tokens": [protocol["training"]["eval_tokens"]] * eval_count,
        "eval_target_tokens": [protocol["training"]["eval_tokens"] - 1] * eval_count,
        "eval_data_digests": ["eval-data"] * eval_count,
    }
    headed = {
        **plain,
        "arm": "head_momentum",
        "final_eval_loss": math.log(34.0),
        "final_eval_perplexity": 34.0,
        "eval_losses": [math.log(40.0)] * (eval_count - 4) + [math.log(value) for value in (36.0, 35.0, 34.5, 34.0)],
        "eval_perplexities": [40.0] * (eval_count - 4) + [36.0, 35.0, 34.5, 34.0],
    }

    result = compare_summaries(
        plain,
        headed,
        protocol=protocol,
        protocol_sha256="protocol",
        code_sha256="code",
    )
    assert result["outcome"] == "strong_reproduction"
    assert result["perplexity_delta"] == 6.0

    stricter = json.loads(json.dumps(protocol))
    stricter["outcome"]["strong_reproduction_min_delta"] = 100.0
    assert compare_summaries(
        plain,
        headed,
        protocol=stricter,
        protocol_sha256="protocol",
        code_sha256="code",
    )["outcome"] == "intermediate"

    mismatched_tokens = {**headed, "eval_target_tokens": [1] + headed["eval_target_tokens"][1:]}
    with pytest.raises(ValueError, match="eval_target_tokens"):
        compare_summaries(plain, mismatched_tokens, protocol=protocol, protocol_sha256="protocol", code_sha256="code")

    mismatched_commit = {**headed, "commit_sha": "b" * 40}
    with pytest.raises(ValueError, match="commit_sha"):
        compare_summaries(plain, mismatched_commit, protocol=protocol, protocol_sha256="protocol", code_sha256="code")

    headed["data_digest"] = "different"
    with pytest.raises(ValueError, match="data_digest"):
        compare_summaries(plain, headed, protocol=protocol, protocol_sha256="protocol", code_sha256="code")

    headed["data_digest"] = "data"
    headed["arm"] = "no_momentum"
    with pytest.raises(ValueError, match="arm"):
        compare_summaries(plain, headed, protocol=protocol, protocol_sha256="protocol", code_sha256="code")

    headed["arm"] = "head_momentum"
    headed["protocol_sha256"] = "wrong"
    with pytest.raises(ValueError, match="protocol_sha256"):
        compare_summaries(plain, headed, protocol=protocol, protocol_sha256="protocol", code_sha256="code")

    headed["protocol_sha256"] = "protocol"
    headed["eval_input_tokens"] = headed["eval_input_tokens"][:-1]
    with pytest.raises(ValueError, match="evaluation count"):
        compare_summaries(plain, headed, protocol=protocol, protocol_sha256="protocol", code_sha256="code")

    headed["eval_input_tokens"] = plain["eval_input_tokens"]
    headed["eval_target_tokens"] = headed["eval_target_tokens"][:-1]
    with pytest.raises(ValueError, match="evaluation count"):
        compare_summaries(plain, headed, protocol=protocol, protocol_sha256="protocol", code_sha256="code")
