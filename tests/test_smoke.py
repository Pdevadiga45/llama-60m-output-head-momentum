import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaForCausalLM

import scale_repro.train as train_module
from scale_repro.core import cosine_multiplier
from scale_repro.train import (
    _code_sha256,
    _evaluate,
    _sha256,
    backward_token_sum,
    build_llama_config,
    build_optimizer,
    build_run_summary,
    canary_timing_samples,
    finalize_manifest,
    main,
    normalize_gradients,
    run_experiment,
    run_smoke,
    token_batches,
    verify_artifacts,
    verify_full_canary_evidence,
)


def _write_canary_artifacts(path: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    updates = protocol["canary"]["updates"]
    nominal_per_update = protocol["training"]["global_batch_size"] * protocol["training"]["sequence_length"]
    loss = math.log(40.0)
    evaluation = {
        "update": updates,
        "loss": loss,
        "perplexity": 40.0,
        "tokens": protocol["training"]["eval_tokens"],
        "target_tokens": protocol["training"]["eval_tokens"] - 1,
        "batches": 1,
        "data_digest": "eval",
        "seconds": 600.0,
    }
    state_path = path / "initial_state.pt"
    with torch.device("meta"):
        state = LlamaForCausalLM(build_llama_config(protocol)).state_dict()
    torch.save(state, state_path)
    summary = {
        "status": "complete",
        "mode": "canary",
        "arm": "head_momentum",
        "protocol_sha256": _sha256(protocol_path),
        "code_sha256": _code_sha256(),
        "commit_sha": "a" * 40,
        "updates": updates,
        "data_tokens": updates * nominal_per_update,
        "nonpad_tokens": updates * nominal_per_update,
        "data_digest": "train",
        "initial_state_sha256": _sha256(state_path),
        "final_eval_loss": loss,
        "final_eval_perplexity": 40.0,
        "eval_updates": [updates],
        "eval_losses": [loss],
        "eval_perplexities": [40.0],
        "eval_input_tokens": [evaluation["tokens"]],
        "eval_target_tokens": [evaluation["target_tokens"]],
        "eval_data_digests": [evaluation["data_digest"]],
        "peak_memory_bytes": 1,
        "model_parameter_bytes": 1,
        "optimizer_state_bytes": 1,
        "scientific_status": "incomplete_canary",
        "scale_gate": "pending_lifecycle",
        "checkpoint_estimate": {
            "median_update_seconds": 0.5,
            "evaluation_seconds": 600.0,
            "experiment_wall_hours": 1.0,
            "aggregate_training_tokens_per_second": 262_144.0,
            "projected_training_and_evaluation_hours": 24_200 / 3_600,
        },
    }
    (path / "summary.json").write_text(json.dumps(summary))
    metrics = [
        {
            "update": update,
            "nominal_tokens": update * nominal_per_update,
            "train_loss": 1.0,
            "lr": protocol["training"]["learning_rate"]
            * cosine_multiplier(
                update - 1,
                warmup=protocol["training"]["warmup_updates"],
                total=protocol["training"]["updates"],
                minimum=protocol["training"]["min_lr_ratio"],
            ),
            "update_seconds": 0.5,
            "nominal_tokens_per_second": 262_144.0,
            "nonpad_tokens": update * nominal_per_update,
            "peak_memory_bytes": 1,
        }
        for update in range(1, updates + 1)
    ]
    (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    (path / "evaluations.jsonl").write_text(json.dumps(evaluation) + "\n")
    finalize_manifest(
        path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )
    return summary


def test_code_hash_only_depends_on_installed_package_files(monkeypatch, tmp_path: Path):
    package = tmp_path / "scale_repro"
    package.mkdir()
    (package / "core.py").write_text("core")
    (package / "train.py").write_text("train")
    monkeypatch.setattr(train_module, "__file__", str(package / "train.py"))

    assert len(train_module._code_sha256()) == 64


def test_smoke_run_writes_recoverable_artifacts(tmp_path: Path):
    summary = run_smoke(tmp_path, arm="head_momentum")

    assert summary["status"] == "complete"
    assert summary["arm"] == "head_momentum"
    assert summary["updates"] == 2
    assert summary["data_tokens"] == 64
    assert summary["final_eval_perplexity"] > 0
    assert len(summary["data_digest"]) == 64
    assert (tmp_path / "initial_state.pt").stat().st_size > 0
    assert (tmp_path / "summary.json").stat().st_size > 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["arm"] == "head_momentum"
    assert manifest["secondary_parameters"] == ["lm_head.weight"]
    assert manifest["momentum_parameters"] == ["lm_head.weight"]
    assert len(manifest["code_sha256"]) == 64
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in metrics] == [1, 2]
    assert [row["nominal_tokens"] for row in metrics] == [32, 64]
    assert [row["lr"] for row in metrics] == pytest.approx([0.0, 0.001])
    assert all(row["train_loss"] > 0 for row in metrics)
    assert all("peak_memory_bytes" in row for row in metrics)


def test_token_batches_preserve_row_order_and_partial_final_batch():
    class FakeTokenizer:
        def __call__(self, texts, *, max_length, truncation, padding, return_tensors):
            import torch

            rows = [[len(text)] * max_length for text in texts]
            input_ids = torch.tensor(rows)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    rows = [{"text": "a"}, {"text": "bbb"}, {"text": "cc"}]
    batches = list(token_batches(rows, FakeTokenizer(), batch_size=2, sequence_length=4))

    assert batches[0]["input_ids"].tolist() == [[1, 1, 1, 1], [3, 3, 3, 3]]
    assert batches[1]["input_ids"].tolist() == [[2, 2, 2, 2]]


def test_llama_config_is_built_only_from_frozen_protocol():
    protocol = json.loads((Path(__file__).parents[1] / "protocol.json").read_text())
    config = build_llama_config(protocol)

    assert config.hidden_size == 512
    assert config.num_hidden_layers == 8
    assert config.num_attention_heads == 8
    assert config.intermediate_size == 1376
    assert config.vocab_size == 32_000
    assert config.pad_token_id == 1
    assert config.tie_word_embeddings is False


def test_no_momentum_arm_does_not_allocate_lm_head_state():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(2, 2, bias=False)
            self.lm_head = nn.Linear(2, 3, bias=False)

    plain_model = Tiny()
    headed_model = Tiny()
    plain = build_optimizer(plain_model, lr=0.1, head_momentum=0.0)
    headed = build_optimizer(headed_model, lr=0.1, head_momentum=0.9)
    for model, optimizer in ((plain_model, plain), (headed_model, headed)):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()

    assert "moment1" not in plain.state[plain_model.lm_head.weight]
    assert "moment1" in headed.state[headed_model.lm_head.weight]


def test_accumulated_gradient_is_weighted_by_valid_target_tokens():
    parameter = nn.Parameter(torch.tensor(1.0))
    labels_one = torch.tensor([[9, 8, -100]])
    labels_three = torch.tensor([[9, 8, 7], [9, 6, -100]])

    targets = backward_token_sum((parameter - 2).square(), labels_one)
    targets += backward_token_sum((parameter + 1).square(), labels_three)
    normalize_gradients([parameter], targets)

    expected = nn.Parameter(torch.tensor(1.0))
    (((expected - 2).square() * 1 + (expected + 1).square() * 3) / 4).backward()
    torch.testing.assert_close(parameter.grad, expected.grad)


def test_evaluation_uses_token_weighted_negative_log_likelihood():
    class LossModel:
        def __init__(self):
            self.losses = iter((2.0, 4.0))

        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(loss=torch.tensor(next(self.losses)))

    batches = [
        {"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])},
        {"input_ids": torch.tensor([[1, 2, 3], [1, 4, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])},
    ]
    result = _evaluate(LossModel(), batches, device=torch.device("cpu"), pad_id=0, token_limit=None)

    assert result["loss"] == 3.5
    assert result["target_tokens"] == 4
    assert result["tokens"] == 7
    assert len(result["data_digest"]) == 64


def test_canary_timing_window_uses_exact_post_warmup_samples():
    protocol = json.loads((Path(__file__).parents[1] / "protocol.json").read_text())
    selected = canary_timing_samples(list(range(protocol["canary"]["updates"])), protocol)

    assert selected == list(range(1_100, 1_200))
    with pytest.raises(ValueError, match="timing samples"):
        canary_timing_samples(list(range(1_199)), protocol)


def test_build_run_summary_records_frozen_assay_evidence():
    training_result = {
        "data_tokens": 1_441_792_000,
        "nonpad_tokens": 1_400_000_000,
        "data_digest": "train-digest",
        "peak_memory_bytes": 123,
        "evaluations": [
            {
                "update": 11_000,
                "loss": math.log(40.0),
                "perplexity": 40.0,
                "tokens": 10_000_000,
                "target_tokens": 9_960_937,
                "data_digest": "eval-digest",
            }
        ],
    }
    summary = build_run_summary(
        arm="no_momentum",
        mode="full",
        updates=11_000,
        training_result=training_result,
        initial_state_sha256="initial",
        protocol_sha256="protocol",
        code_sha256="code",
        commit_sha="a" * 40,
        model_parameter_bytes=100,
        optimizer_state_bytes=20,
    )

    assert summary["mode"] == "full"
    assert summary["protocol_sha256"] == "protocol"
    assert summary["code_sha256"] == "code"
    assert summary["commit_sha"] == "a" * 40
    assert summary["eval_updates"] == [11_000]
    assert summary["eval_losses"] == [math.log(40.0)]
    assert summary["eval_input_tokens"] == [10_000_000]
    assert summary["eval_target_tokens"] == [9_960_937]
    assert summary["eval_data_digests"] == ["eval-digest"]
    assert summary["model_parameter_bytes"] == 100
    assert summary["optimizer_state_bytes"] == 20


def test_artifact_manifest_detects_corruption(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    _write_canary_artifacts(tmp_path, protocol_path)

    assert verify_artifacts(tmp_path)["status"] == "complete"
    (tmp_path / "metrics.jsonl").write_text('{"update":2}\n')
    with pytest.raises(ValueError, match="metrics.jsonl"):
        verify_artifacts(tmp_path)


def test_artifact_verifier_rejects_malformed_model_state(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    torch.save({"weight": torch.ones(1)}, tmp_path / "initial_state.pt")
    finalize_manifest(
        tmp_path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="model state"):
        verify_artifacts(tmp_path, protocol_path=protocol_path)


def test_artifact_verifier_rejects_incomplete_inventory(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    _write_canary_artifacts(tmp_path, protocol_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["artifacts"].pop("summary.json")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact inventory"):
        verify_artifacts(tmp_path)


def test_artifact_verifier_rejects_summary_evaluation_mismatch(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    summary["eval_perplexities"] = [41.0]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    finalize_manifest(
        tmp_path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="evaluation evidence"):
        verify_artifacts(tmp_path)


def test_canary_artifact_requires_evaluation_at_final_canary_update(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    summary["eval_updates"] = [1]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    evaluation = json.loads((tmp_path / "evaluations.jsonl").read_text())
    evaluation["update"] = 1
    (tmp_path / "evaluations.jsonl").write_text(json.dumps(evaluation) + "\n")
    finalize_manifest(
        tmp_path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="evaluation update"):
        verify_artifacts(tmp_path, protocol_path=protocol_path)


def test_artifact_verifier_recomputes_canary_checkpoint(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    summary["checkpoint_estimate"]["median_update_seconds"] = 1.0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    finalize_manifest(
        tmp_path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="checkpoint estimate"):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].update(train_loss=float("nan")),
        lambda rows: rows[0].update(lr=99.0),
        lambda rows: rows[0].update(nominal_tokens_per_second=-1.0),
        lambda rows: rows[1].update(nonpad_tokens=rows[0]["nonpad_tokens"] - 1),
        lambda rows: rows[0].update(nonpad_tokens=rows[0]["nominal_tokens"] + 1),
        lambda rows: rows[0].update(peak_memory_bytes=-1),
        lambda rows: rows[0].update(extra="not allowed"),
    ],
)
def test_artifact_verifier_rejects_invalid_metric_evidence(tmp_path: Path, mutation):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    mutation(rows)
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    finalize_manifest(
        tmp_path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="metrics"):
        verify_artifacts(tmp_path, protocol_path=protocol_path)


def test_artifact_verifier_rejects_incomplete_production_status(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    summary["status"] = "failed"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    finalize_manifest(
        tmp_path,
        {
            "arm": summary["arm"],
            "mode": summary["mode"],
            "protocol_sha256": summary["protocol_sha256"],
            "code_sha256": summary["code_sha256"],
            "commit_sha": summary["commit_sha"],
            "initial_state_sha256": summary["initial_state_sha256"],
        },
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="status"):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize("mode", [None, "mystery"])
def test_artifact_verifier_rejects_absent_or_unknown_mode(tmp_path: Path, mode):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    summary = _write_canary_artifacts(tmp_path, protocol_path)
    if mode is None:
        summary.pop("mode")
    else:
        summary["mode"] = mode
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if mode is None:
        manifest.pop("mode")
    else:
        manifest["mode"] = mode
    finalize_manifest(
        tmp_path,
        {key: value for key, value in manifest.items() if key != "artifacts"},
        ["summary.json", "metrics.jsonl", "evaluations.jsonl", "initial_state.pt"],
    )

    with pytest.raises(ValueError, match="mode"):
        verify_artifacts(tmp_path, protocol_path=protocol_path)


def test_finalize_canary_cli_records_supervised_session_gate(tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    run_dir = tmp_path / "canary"
    run_dir.mkdir()
    summary = _write_canary_artifacts(run_dir, protocol_path)
    result_path = tmp_path / "decision.json"

    main(
        [
            "finalize-canary",
            "--canary-run",
            str(run_dir),
            "--protocol",
            str(protocol_path),
            "--provider-wall-hours",
            "1.2",
            "--spend-inr",
            "40",
            "--output",
            str(result_path),
        ]
    )

    decision = json.loads(result_path.read_text())
    assert decision["scale_gate"] == "pass"
    assert decision["supervised_single_instance"] is True
    assert decision["initial_state_sha256"] == summary["initial_state_sha256"]
    assert decision["canary_manifest_sha256"] == _sha256(run_dir / "manifest.json")


@pytest.mark.parametrize("mutation", ["nonhex", "copied-field", "canary-artifact"])
def test_full_canary_evidence_is_bound_to_verified_canary_artifacts(tmp_path: Path, mutation: str):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    run_dir = tmp_path / "canary"
    run_dir.mkdir()
    summary = _write_canary_artifacts(run_dir, protocol_path)
    decision_path = tmp_path / "decision.json"
    main(
        [
            "finalize-canary",
            "--canary-run",
            str(run_dir),
            "--protocol",
            str(protocol_path),
            "--provider-wall-hours",
            "1.2",
            "--spend-inr",
            "40",
            "--output",
            str(decision_path),
        ]
    )

    assert verify_full_canary_evidence(
        decision_path,
        canary_run=run_dir,
        protocol_path=protocol_path,
        initial_state_path=run_dir / "initial_state.pt",
        commit_sha=summary["commit_sha"],
    )["scale_gate"] == "pass"

    if mutation == "canary-artifact":
        (run_dir / "metrics.jsonl").write_text('{"update":1}\n')
    else:
        decision = json.loads(decision_path.read_text())
        if mutation == "nonhex":
            decision["canary_manifest_sha256"] = "g" * 64
        else:
            decision["projected_full_hours"] = 0.1
        decision_path.write_text(json.dumps(decision))

    with pytest.raises(ValueError):
        verify_full_canary_evidence(
            decision_path,
            canary_run=run_dir,
            protocol_path=protocol_path,
            initial_state_path=run_dir / "initial_state.pt",
            commit_sha=summary["commit_sha"],
        )


@pytest.mark.parametrize(
    ("wall", "spend"),
    [(0.0, 40.0), (1.2, -1.0), (float("nan"), 40.0), (1.2, float("nan"))],
)
def test_finalize_canary_rejects_invalid_session_cost(tmp_path: Path, wall: float, spend: float):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    run_dir = tmp_path / "canary"
    run_dir.mkdir()
    _write_canary_artifacts(run_dir, protocol_path)

    with pytest.raises(ValueError):
        main(
            [
                "finalize-canary",
                "--canary-run",
                str(run_dir),
                "--protocol",
                str(protocol_path),
                "--provider-wall-hours",
                str(wall),
                "--spend-inr",
                str(spend),
                "--output",
                str(tmp_path / "decision.json"),
            ]
        )


def test_finalize_canary_never_overwrites_existing_decision(tmp_path: Path):
    output = tmp_path / "decision.json"
    output.write_text('{"scale_gate":"pass"}\n')

    with pytest.raises(FileExistsError):
        main(
            [
                "finalize-canary",
                "--canary-run",
                str(tmp_path / "missing"),
                "--protocol",
                str(Path(__file__).parents[1] / "protocol.json"),
                "--provider-wall-hours",
                "1.2",
                "--spend-inr",
                "40",
                "--output",
                str(output),
            ]
        )

    assert output.read_text() == '{"scale_gate":"pass"}\n'


def test_full_train_requires_canary_decision_before_gpu_or_output(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SCALE_COMMIT_SHA", "a" * 40)
    output = tmp_path / "full"
    initial_state = tmp_path / "initial_state.pt"
    torch.save({"weight": torch.ones(1)}, initial_state)

    with pytest.raises(ValueError, match="canary decision"):
        main(
            [
                "train",
                "--mode",
                "full",
                "--arm",
                "no_momentum",
                "--protocol",
                str(Path(__file__).parents[1] / "protocol.json"),
                "--initial-state",
                str(initial_state),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_direct_full_train_verifies_complete_canary_bundle_before_cuda(monkeypatch, tmp_path: Path):
    protocol_path = Path(__file__).parents[1] / "protocol.json"
    run_dir = tmp_path / "canary"
    run_dir.mkdir()
    summary = _write_canary_artifacts(run_dir, protocol_path)
    decision_path = tmp_path / "decision.json"
    main(
        [
            "finalize-canary",
            "--canary-run",
            str(run_dir),
            "--protocol",
            str(protocol_path),
            "--provider-wall-hours",
            "1.2",
            "--spend-inr",
            "40",
            "--output",
            str(decision_path),
        ]
    )
    (run_dir / "metrics.jsonl").write_text('{"update":1}\n')
    monkeypatch.setenv("SCALE_COMMIT_SHA", summary["commit_sha"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA checked before canary verification"))

    with pytest.raises(ValueError, match="metrics.jsonl"):
        run_experiment(
            tmp_path / "full",
            arm="no_momentum",
            mode="full",
            protocol_path=protocol_path,
            initial_state_path=run_dir / "initial_state.pt",
            canary_decision_path=decision_path,
            canary_run_path=run_dir,
        )


def test_train_cli_never_contaminates_an_existing_output(tmp_path: Path):
    output = tmp_path / "completed"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text('{"status":"complete"}\n')

    with pytest.raises(FileExistsError):
        main(
            [
                "train",
                "--mode",
                "canary",
                "--arm",
                "head_momentum",
                "--protocol",
                str(Path(__file__).parents[1] / "protocol.json"),
                "--output",
                str(output),
            ]
        )

    assert sentinel.read_text() == '{"status":"complete"}\n'
    assert not (output / "failure.json").exists()
