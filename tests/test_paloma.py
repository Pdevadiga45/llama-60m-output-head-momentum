import gzip
import hashlib
import json
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scale_repro.paloma as paloma_module
from scale_repro.paloma import (
    _code_sha256,
    _compare_results,
    evaluate_domain,
    load_extension_protocol,
    main,
    rolling_windows,
    score_windows,
    summarize_pair,
    validate_checkpoint,
    validate_data_manifest,
    verify_bundle,
    write_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PROTOCOL_PATH = ROOT / "scale_repro" / "extension_protocol.json"


def test_frozen_protocols_have_one_packaged_source():
    for name in ("protocol.json", "extension_protocol.json"):
        assert (ROOT / "scale_repro" / name).is_file()
        assert not (ROOT / name).exists()


def _pinned_lm_eval_windows(token_ids: list[int], *, prefix_token: int, max_seq_len: int):
    """Faithful port of d6de816 rolling windows plus make_disjoint_window."""
    if not token_ids:
        return []
    context_len = 1
    pred_len = max_seq_len - context_len + 1
    predicted = 0
    raw_windows = []
    first_seq_len = min(max_seq_len, len(token_ids))
    raw_windows.append(
        ([prefix_token] + token_ids[: first_seq_len - 1], token_ids[:first_seq_len])
    )
    predicted += first_seq_len
    while predicted < len(token_ids):
        window_pred_len = min(len(token_ids) - predicted, pred_len)
        window_end = predicted + window_pred_len
        raw_windows.append(
            (
                token_ids[window_end - max_seq_len - 1 : window_end - 1],
                token_ids[predicted:window_end],
            )
        )
        predicted += window_pred_len
    return [
        (input_tokens[: len(input_tokens) - len(pred_tokens) + 1], pred_tokens)
        for input_tokens, pred_tokens in raw_windows
    ]


def _fake_result(protocol: dict, arm: str, sum_logprob: float) -> dict:
    manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "complete",
        "benchmark": protocol["paloma"]["name"],
        "arm": arm,
        "device_type": protocol["paloma"]["evaluation"]["device_type"],
        "dtype": protocol["paloma"]["evaluation"]["dtype"],
        "checkpoint_sha256": protocol["base"][f"{arm}_state_sha256"],
        "checkpoint_summary_sha256": protocol["base"][f"{arm}_summary_sha256"],
        "data_manifest_sha256": manifest_hash,
        "extension_protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "extension_code_sha256": _code_sha256(),
        "domains": [
            {
                "group": domain["group"],
                "name": domain["name"],
                "path": domain["path"],
                "documents": 1,
                "tokens": 100,
                "bytes": domain["bytes"],
                "data_sha256": domain["sha256"],
                "sum_logprob": sum_logprob,
            }
            for domain in protocol["paloma"]["domains"]
        ],
    }


def test_extension_protocol_is_frozen_to_paloma_b_only():
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)

    assert set(protocol) == {"schema_version", "frozen_at", "base", "paloma"}
    assert protocol["frozen_at"] == "2026-09-24"
    assert protocol["paloma"]["dataset_revision"] == "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6"
    assert protocol["paloma"]["model_max_seq_len"] == 256
    assert protocol["paloma"]["window_policy"] == "lm_eval_rolling_disjoint"
    assert protocol["paloma"]["window_reference"] == {
        "repository": "EleutherAI/lm-evaluation-harness",
        "revision": "d6de81643928d653435c431bae19945d41d32520",
        "context_len": 1,
    }
    assert protocol["paloma"]["evaluation"] == {
        "device_type": "cuda",
        "dtype": "bfloat16",
    }
    base_protocol = json.loads((ROOT / "scale_repro/protocol.json").read_text())
    assert protocol["base"]["source_protocol_canonical_sha256"] == hashlib.sha256(
        json.dumps(base_protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert protocol["paloma"]["tokenizer_revision"] == "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    assert len(protocol["paloma"]["domains"]) == 12
    assert {domain["group"] for domain in protocol["paloma"]["domains"]} == {
        "c4_control",
        "ood_reddit",
        "ood_code",
    }
    assert all(set(domain) == {"group", "name", "path", "bytes", "sha256"} for domain in protocol["paloma"]["domains"])
    assert "coordinate_clip" not in json.dumps(protocol)


def test_data_manifest_binds_exact_revision_inventory_and_local_files():
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())

    assert len(validate_data_manifest(manifest, protocol, ROOT / "data/paloma")) == 64

    changed = json.loads(json.dumps(manifest))
    changed["files"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="manifest"):
        validate_data_manifest(changed, protocol, ROOT / "data/paloma")


@pytest.mark.parametrize("length", [0, 3, 256, 257, 512, 514])
def test_rolling_windows_match_pinned_lm_eval_reference(length: int):
    token_ids = list(range(length))

    assert list(rolling_windows(token_ids, prefix_token=-1, max_seq_len=256)) == (
        _pinned_lm_eval_windows(token_ids, prefix_token=-1, max_seq_len=256)
    )


@pytest.mark.parametrize(
    ("length", "expected_contexts", "expected_continuation_lengths"),
    [
        (257, [[-1], list(range(256))], [256, 1]),
        (514, [[-1], [255], list(range(257, 512))], [256, 256, 2]),
    ],
)
def test_rolling_windows_preserve_exact_partial_tail_contexts(
    length: int,
    expected_contexts: list[list[int]],
    expected_continuation_lengths: list[int],
):
    windows = list(rolling_windows(list(range(length)), prefix_token=-1, max_seq_len=256))

    assert [context for context, _ in windows] == expected_contexts
    assert [len(continuation) for _, continuation in windows] == expected_continuation_lengths


@pytest.mark.parametrize("length", [257, 514])
def test_rolling_windows_score_every_token_once_with_bounded_model_inputs(length: int):
    token_ids = list(range(length))
    windows = list(rolling_windows(token_ids, prefix_token=-1, max_seq_len=256))

    assert [token for _, continuation in windows for token in continuation] == token_ids
    assert all(len((context + continuation)[:-1]) <= 256 for context, continuation in windows)


def test_score_windows_sums_exact_continuation_log_likelihood():
    class NextTokenModel:
        def __call__(self, *, input_ids, attention_mask):
            logits = torch.zeros(*input_ids.shape, 5)
            logits.scatter_(2, ((input_ids + 1) % 5).unsqueeze(-1), 2.0)
            return SimpleNamespace(logits=logits)

    result = score_windows(
        NextTokenModel(),
        [([1], [2, 3]), ([3], [4])],
        device=torch.device("cpu"),
        batch_size=2,
        pad_id=0,
    )

    expected = 3 * (2.0 - math.log(math.exp(2.0) + 4.0))
    assert result == {"sum_logprob": pytest.approx(expected), "tokens": 3}


def test_score_windows_streams_only_one_bounded_batch_ahead():
    inferred = False

    class Model:
        def __call__(self, *, input_ids, attention_mask):
            nonlocal inferred
            inferred = True
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 5))

    def windows():
        for index in range(3):
            if index == 2 and not inferred:
                raise AssertionError("iterator was fully materialized")
            yield [1], [2]

    assert score_windows(
        Model(), windows(), device=torch.device("cpu"), batch_size=2, pad_id=0
    )["tokens"] == 3


def test_score_windows_uses_float32_log_softmax_and_reduction():
    class BFloatModel:
        def __call__(self, *, input_ids, attention_mask):
            logits = torch.zeros(*input_ids.shape, 5, dtype=torch.bfloat16)
            logits[..., 2] = 2
            return SimpleNamespace(logits=logits)

    result = score_windows(
        BFloatModel(),
        [([1], [2] * 256)],
        device=torch.device("cpu"),
        batch_size=1,
        pad_id=0,
    )

    expected = 256 * (2.0 - math.log(math.exp(2.0) + 4.0))
    assert result["sum_logprob"] == pytest.approx(expected, abs=1e-4)


def test_domain_evaluation_streams_complete_gzip_file(tmp_path: Path):
    path = tmp_path / "domain.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "ab"}) + "\n")
        handle.write(json.dumps({"text": "c"}) + "\n")

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [2] * len(text)}

    class UniformModel:
        def __call__(self, *, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 5))

    result = evaluate_domain(
        UniformModel(),
        Tokenizer(),
        path,
        {"name": "tiny", "group": "ood_code", "path": "tiny.jsonl.gz", "bytes": path.stat().st_size, "sha256": "ignored"},
        prefix_token=1,
        max_seq_len=256,
        batch_size=2,
        pad_id=0,
        device=torch.device("cpu"),
    )

    assert result["documents"] == 2
    assert result["tokens"] == 3
    assert result["sum_logprob"] == pytest.approx(-3 * math.log(5))


def test_checkpoint_validation_binds_frozen_state_and_summary_hashes(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    state = ROOT / "artifacts/no_momentum/final_state.pt"

    evidence = validate_checkpoint(state, "no_momentum", protocol)

    assert evidence["checkpoint_sha256"] == protocol["base"]["no_momentum_state_sha256"]
    assert evidence["checkpoint_summary_sha256"] == protocol["base"]["no_momentum_summary_sha256"]

    bad_state = tmp_path / "final_state.pt"
    bad_state.write_bytes(b"wrong")
    (tmp_path / "summary.json").write_text((state.with_name("summary.json")).read_text())
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_checkpoint(bad_state, "no_momentum", protocol)


def test_pair_summary_reports_descriptive_cross_domain_fit_only():
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    plain = []
    headed = []
    for domain in protocol["paloma"]["domains"]:
        evidence = {
            "group": domain["group"],
            "name": domain["name"],
            "path": domain["path"],
            "documents": 1,
            "tokens": 100,
            "bytes": domain["bytes"],
            "data_sha256": domain["sha256"],
        }
        plain.append({**evidence, "sum_logprob": -400.0})
        headed.append({**evidence, "sum_logprob": -390.0})

    result = summarize_pair(plain, headed, protocol)

    assert result["framing"] == "paired_cross_domain_fit"
    assert set(result) == {"benchmark", "framing", "domains", "groups"}
    assert result["groups"]["ood"]["delta_nats_per_token"] == pytest.approx(-0.1)
    assert "gate" not in json.dumps(result)


def test_bundle_has_exact_inventory_recomputed_comparison_and_hash_manifest(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    output = tmp_path / "bundle"

    verified = write_bundle(
        output,
        _fake_result(protocol, "no_momentum", -400.0),
        _fake_result(protocol, "head_momentum", -390.0),
        data_manifest,
        protocol,
    )

    assert {path.name for path in output.iterdir()} == {
        "no_momentum.json",
        "head_momentum.json",
        "comparison.json",
        "data_manifest.json",
        "manifest.json",
    }
    assert verified["comparison"]["framing"] == "paired_cross_domain_fit"
    assert set(verified["manifest"]) == {
        "status",
        "extension_protocol_sha256",
        "extension_code_sha256",
        "data_manifest_sha256",
        "device_type",
        "dtype",
        "no_momentum_checkpoint",
        "head_momentum_checkpoint",
        "no_momentum_checkpoint_summary",
        "head_momentum_checkpoint_summary",
        "artifacts",
    }
    assert verified["manifest"]["extension_code_sha256"] == _code_sha256()
    assert len(_code_sha256()) == 64

    (output / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(output, protocol)


def test_bundle_rejects_extra_raw_result_fields(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    no_momentum = _fake_result(protocol, "no_momentum", -400.0)
    no_momentum["unexpected"] = True

    with pytest.raises(ValueError, match="schema"):
        write_bundle(
            tmp_path / "bundle",
            no_momentum,
            _fake_result(protocol, "head_momentum", -390.0),
            data_manifest,
            protocol,
        )


@pytest.mark.parametrize("arm", ["no_momentum", "head_momentum"])
def test_write_bundle_rejects_extra_domain_result_fields(tmp_path: Path, arm: str):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    results = {
        "no_momentum": _fake_result(protocol, "no_momentum", -400.0),
        "head_momentum": _fake_result(protocol, "head_momentum", -390.0),
    }
    results[arm]["domains"][0]["unexpected"] = True

    with pytest.raises(ValueError, match="domain result schema"):
        write_bundle(
            tmp_path / "bundle",
            results["no_momentum"],
            results["head_momentum"],
            data_manifest,
            protocol,
        )


@pytest.mark.parametrize("arm", ["no_momentum", "head_momentum"])
def test_verify_bundle_rejects_extra_domain_result_fields(tmp_path: Path, arm: str):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    output = tmp_path / "bundle"
    write_bundle(
        output,
        _fake_result(protocol, "no_momentum", -400.0),
        _fake_result(protocol, "head_momentum", -390.0),
        data_manifest,
        protocol,
    )
    result_path = output / f"{arm}.json"
    result = json.loads(result_path.read_text())
    result["domains"][0]["unexpected"] = True
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][result_path.name] = {
        "bytes": result_path.stat().st_size,
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="domain result schema"):
        verify_bundle(output, protocol)


def test_compare_rejects_forged_protocol_dict():
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    forged = {**protocol, "forged": True}

    with pytest.raises(ValueError, match="frozen Paloma protocol"):
        _compare_results(
            _fake_result(protocol, "no_momentum", -400.0),
            _fake_result(protocol, "head_momentum", -390.0),
            forged,
        )


def test_write_bundle_rejects_forged_protocol_dict(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    forged = {**protocol, "forged": True}
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())

    with pytest.raises(ValueError, match="frozen Paloma protocol"):
        write_bundle(
            tmp_path / "bundle",
            _fake_result(protocol, "no_momentum", -400.0),
            _fake_result(protocol, "head_momentum", -390.0),
            data_manifest,
            forged,
        )


def test_verify_bundle_rejects_forged_protocol_dict(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    output = tmp_path / "bundle"
    write_bundle(
        output,
        _fake_result(protocol, "no_momentum", -400.0),
        _fake_result(protocol, "head_momentum", -390.0),
        data_manifest,
        protocol,
    )

    with pytest.raises(ValueError, match="frozen Paloma protocol"):
        verify_bundle(output, {**protocol, "forged": True})


def test_verifier_rejects_internally_consistent_forged_code_hash(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    output = tmp_path / "bundle"
    write_bundle(
        output,
        _fake_result(protocol, "no_momentum", -400.0),
        _fake_result(protocol, "head_momentum", -390.0),
        data_manifest,
        protocol,
    )

    forged = "f" * 64
    for name in ("no_momentum.json", "head_momentum.json", "comparison.json"):
        path = output / name
        payload = json.loads(path.read_text())
        payload["extension_code_sha256"] = forged
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extension_code_sha256"] = forged
    for name in ("no_momentum.json", "head_momentum.json", "comparison.json"):
        path = output / name
        manifest["artifacts"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="executing package"):
        verify_bundle(output, protocol)


def test_verifier_rejects_extra_manifest_fields(tmp_path: Path):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    data_manifest = json.loads((ROOT / "artifacts/extensions/paloma_data_manifest.json").read_text())
    output = tmp_path / "bundle"
    write_bundle(
        output,
        _fake_result(protocol, "no_momentum", -400.0),
        _fake_result(protocol, "head_momentum", -390.0),
        data_manifest,
        protocol,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest schema"):
        verify_bundle(output, protocol)


def test_pair_cli_evaluates_both_frozen_arms_in_one_command(tmp_path: Path, monkeypatch):
    protocol = load_extension_protocol(EXTENSION_PROTOCOL_PATH)
    calls = []

    def fake_evaluate(*, arm, **kwargs):
        calls.append((arm, kwargs["device"].type))
        return _fake_result(protocol, arm, -400.0 if arm == "no_momentum" else -390.0)

    monkeypatch.setattr(paloma_module, "evaluate_checkpoint", fake_evaluate)
    output = tmp_path / "bundle"
    main([
        "--no-momentum-checkpoint", str(ROOT / "artifacts/no_momentum/final_state.pt"),
        "--head-momentum-checkpoint", str(ROOT / "artifacts/head_momentum/final_state.pt"),
        "--data-root", str(ROOT / "data/paloma"),
        "--data-manifest", str(ROOT / "artifacts/extensions/paloma_data_manifest.json"),
        "--protocol", str(EXTENSION_PROTOCOL_PATH),
        "--output", str(output),
    ])

    assert calls == [("no_momentum", "cuda"), ("head_momentum", "cuda")]
    assert verify_bundle(output, protocol)["comparison"]["framing"] == "paired_cross_domain_fit"


def test_wheel_configuration_packages_protocols_and_one_paloma_entrypoint():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert project["project"]["scripts"]["scale-paloma"] == "scale_repro.paloma:main"
    assert project["tool"]["setuptools"]["package-data"]["scale_repro"] == ["*.json"]
