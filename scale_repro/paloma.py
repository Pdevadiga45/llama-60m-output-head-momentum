import argparse
import gzip
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import torch


_PROTOCOL_SHA256 = "b655a96966b46e8f7df4a7d9a217fd7efceaf1dcc8a9f6af5119d2fef51342f4"
_DEFAULT_PROTOCOL = Path(__file__).with_name("extension_protocol.json")
_RESULT_FIELDS = {
    "status",
    "benchmark",
    "arm",
    "device_type",
    "dtype",
    "checkpoint_sha256",
    "checkpoint_summary_sha256",
    "data_manifest_sha256",
    "extension_protocol_sha256",
    "extension_code_sha256",
    "domains",
}
_DOMAIN_RESULT_FIELDS = {
    "group",
    "name",
    "path",
    "documents",
    "tokens",
    "bytes",
    "data_sha256",
    "sum_logprob",
}
_BUNDLE_METADATA = (
    "extension_protocol_sha256",
    "extension_code_sha256",
    "data_manifest_sha256",
    "device_type",
    "dtype",
    "no_momentum_checkpoint",
    "head_momentum_checkpoint",
    "no_momentum_checkpoint_summary",
    "head_momentum_checkpoint_summary",
)
_MANIFEST_FIELDS = {"status", "artifacts", *_BUNDLE_METADATA}


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _authenticated_protocol(protocol: dict | str | Path) -> dict:
    protocol = protocol if isinstance(protocol, dict) else json.loads(Path(protocol).read_text())
    if not isinstance(protocol, dict) or _canonical_sha256(protocol) != _PROTOCOL_SHA256:
        raise ValueError("extension protocol differs from the frozen Paloma protocol")
    return protocol


def load_extension_protocol(path: str | Path = _DEFAULT_PROTOCOL) -> dict:
    return _authenticated_protocol(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("core.py", "train.py", "paloma.py", "protocol.json", "extension_protocol.json"):
        digest.update(f"scale_repro/{name}".encode())
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def _expected_data_manifest(protocol: dict) -> dict:
    paloma = protocol["paloma"]
    return {
        "schema_version": 1,
        "dataset": paloma["dataset"],
        "requested_revision": paloma["dataset_revision"],
        "resolved_revision": paloma["dataset_revision"],
        "files": [
            {key: domain[key] for key in ("path", "bytes", "sha256")}
            for domain in paloma["domains"]
        ],
    }


def validate_data_manifest(manifest: dict, protocol: dict, data_root: str | Path) -> str:
    if manifest != _expected_data_manifest(protocol):
        raise ValueError("Paloma data manifest differs from the frozen inventory")
    root = Path(data_root)
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Paloma data file differs from manifest: {entry['path']}")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def rolling_windows(token_ids: list[int], *, prefix_token: int, max_seq_len: int):
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    predicted = 0
    while predicted < len(token_ids):
        continuation = token_ids[predicted : predicted + max_seq_len]
        if predicted == 0:
            context = [prefix_token]
        else:
            context_len = max_seq_len + 1 - len(continuation)
            context = token_ids[predicted - context_len : predicted]
        yield context, continuation
        predicted += len(continuation)


@torch.no_grad()
def score_windows(model, windows, *, device: torch.device, batch_size: int, pad_id: int) -> dict:
    total = 0.0
    tokens = 0
    iterator = iter(windows)
    while batch := list(itertools.islice(iterator, batch_size)):
        lengths = [len(context) + len(continuation) - 1 for context, continuation in batch]
        width = max(lengths)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for index, ((context, continuation), length) in enumerate(zip(batch, lengths)):
            input_ids[index, :length] = torch.tensor((context + continuation)[:-1], device=device)
            attention_mask[index, :length] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        for index, (context, continuation) in enumerate(batch):
            selected = log_probs[index, len(context) - 1 : len(context) - 1 + len(continuation)]
            targets = torch.tensor(continuation, dtype=torch.long, device=device)
            total += float(selected.gather(1, targets[:, None]).sum())
            tokens += len(continuation)
    return {"sum_logprob": total, "tokens": tokens}


def evaluate_domain(
    model,
    tokenizer,
    path: str | Path,
    domain: dict,
    *,
    prefix_token: int,
    max_seq_len: int,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> dict:
    path = Path(path)
    documents = 0

    def windows():
        nonlocal documents
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                documents += 1
                text = json.loads(line)["text"]
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                yield from rolling_windows(token_ids, prefix_token=prefix_token, max_seq_len=max_seq_len)

    scored = score_windows(model, windows(), device=device, batch_size=batch_size, pad_id=pad_id)
    if scored["tokens"] <= 0:
        raise ValueError(f"Paloma domain produced no tokens: {domain['name']}")
    return {
        **{key: domain[key] for key in ("group", "name", "path")},
        "documents": documents,
        "bytes": path.stat().st_size,
        "data_sha256": _sha256_file(path),
        **scored,
    }


def validate_checkpoint(checkpoint_path: str | Path, arm: str, protocol: dict) -> dict:
    state_key = {
        "no_momentum": "no_momentum_state_sha256",
        "head_momentum": "head_momentum_state_sha256",
    }.get(arm)
    summary_key = {
        "no_momentum": "no_momentum_summary_sha256",
        "head_momentum": "head_momentum_summary_sha256",
    }.get(arm)
    if state_key is None or summary_key is None:
        raise ValueError(f"unknown Paloma arm: {arm}")
    checkpoint = Path(checkpoint_path)
    checkpoint_hash = _sha256_file(checkpoint)
    if checkpoint_hash != protocol["base"][state_key]:
        raise ValueError("checkpoint hash differs from the frozen Paloma protocol")
    summary_path = checkpoint.with_name("summary.json")
    summary_hash = _sha256_file(summary_path)
    if summary_hash != protocol["base"][summary_key]:
        raise ValueError("checkpoint summary hash differs from the frozen Paloma protocol")
    summary = json.loads(summary_path.read_text())
    expected = {
        "status": "complete",
        "mode": "full",
        "arm": arm,
        "commit_sha": protocol["base"]["training_commit_sha"],
        "protocol_sha256": protocol["base"]["artifact_protocol_sha256"],
        "initial_state_sha256": protocol["base"]["initial_state_sha256"],
        "updates": 11_000,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint summary differs from frozen provenance")
    return {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_summary_sha256": summary_hash,
    }


def summarize_pair(no_momentum: list[dict], head_momentum: list[dict], protocol: dict) -> dict:
    domains = protocol["paloma"]["domains"]
    if len(no_momentum) != len(domains) or len(head_momentum) != len(domains):
        raise ValueError("Paloma result count differs from the frozen inventory")
    rows = []
    evidence_fields = ("group", "name", "path", "documents", "tokens", "bytes", "data_sha256")
    for domain, plain, headed in zip(domains, no_momentum, head_momentum):
        for arm, row in (("no_momentum", plain), ("head_momentum", headed)):
            if not isinstance(row, dict) or set(row) != _DOMAIN_RESULT_FIELDS:
                raise ValueError(f"invalid {arm} Paloma domain result schema")
        if any(plain.get(field) != headed.get(field) for field in evidence_fields):
            raise ValueError("paired Paloma evidence differs")
        expected = {
            "group": domain["group"],
            "name": domain["name"],
            "path": domain["path"],
            "bytes": domain["bytes"],
            "data_sha256": domain["sha256"],
        }
        if any(plain.get(key) != value for key, value in expected.items()):
            raise ValueError("Paloma evidence differs from the frozen inventory")
        if any(isinstance(plain.get(key), bool) or not isinstance(plain.get(key), int) or plain[key] <= 0 for key in ("documents", "tokens")):
            raise ValueError("Paloma document and token counts must be positive")
        scores = (plain.get("sum_logprob"), headed.get("sum_logprob"))
        if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score > 0 for score in scores):
            raise ValueError("Paloma log probabilities must be finite and non-positive")
        plain_loss = -plain["sum_logprob"] / plain["tokens"]
        headed_loss = -headed["sum_logprob"] / headed["tokens"]
        rows.append({
            **{field: plain[field] for field in evidence_fields},
            "no_momentum_nats_per_token": plain_loss,
            "head_momentum_nats_per_token": headed_loss,
            "delta_nats_per_token": headed_loss - plain_loss,
            "no_momentum_perplexity": math.exp(plain_loss),
            "head_momentum_perplexity": math.exp(headed_loss),
        })

    def macro(name: str, selected: list[dict]) -> dict:
        plain = sum(row["no_momentum_nats_per_token"] for row in selected) / len(selected)
        headed = sum(row["head_momentum_nats_per_token"] for row in selected) / len(selected)
        return {
            "domains": len(selected),
            "no_momentum_nats_per_token": plain,
            "head_momentum_nats_per_token": headed,
            "delta_nats_per_token": headed - plain,
        }

    groups = {
        name: macro(name, [row for row in rows if row["group"] == name])
        for name in ("c4_control", "ood_reddit", "ood_code")
    }
    groups["ood"] = macro("ood", [row for row in rows if row["group"].startswith("ood_")])
    return {
        "benchmark": protocol["paloma"]["name"],
        "framing": protocol["paloma"]["framing"],
        "domains": rows,
        "groups": groups,
    }


def _compare_results(no_momentum: dict, head_momentum: dict, protocol: dict) -> dict:
    protocol = _authenticated_protocol(protocol)
    expected = {
        "no_momentum": no_momentum,
        "head_momentum": head_momentum,
    }
    code_hash = _code_sha256()
    evaluation = protocol["paloma"]["evaluation"]
    for arm, result in expected.items():
        if set(result) != _RESULT_FIELDS:
            raise ValueError(f"invalid {arm} Paloma result schema")
        if result.get("status") != "complete" or result.get("arm") != arm:
            raise ValueError(f"invalid {arm} Paloma result")
        if result.get("benchmark") != protocol["paloma"]["name"]:
            raise ValueError("Paloma benchmark differs from the frozen protocol")
        if result.get("device_type") != evaluation["device_type"] or result.get("dtype") != evaluation["dtype"]:
            raise ValueError("Paloma evaluation precision differs from the frozen protocol")
        if result.get("checkpoint_sha256") != protocol["base"][f"{arm}_state_sha256"]:
            raise ValueError("Paloma checkpoint differs from the frozen protocol")
        if result.get("checkpoint_summary_sha256") != protocol["base"][f"{arm}_summary_sha256"]:
            raise ValueError("Paloma checkpoint summary differs from the frozen protocol")
        if result.get("extension_code_sha256") != code_hash:
            raise ValueError("Paloma result extension code hash differs from the executing package")
    shared = (
        "data_manifest_sha256",
        "extension_protocol_sha256",
        "extension_code_sha256",
        "device_type",
        "dtype",
    )
    if any(no_momentum.get(field) != head_momentum.get(field) for field in shared):
        raise ValueError("paired Paloma metadata differs")
    if no_momentum.get("extension_protocol_sha256") != _PROTOCOL_SHA256:
        raise ValueError("Paloma result uses a different extension protocol")
    if re.fullmatch(r"[0-9a-f]{64}", no_momentum.get("data_manifest_sha256", "")) is None:
        raise ValueError("invalid data_manifest_sha256")
    comparison = summarize_pair(no_momentum["domains"], head_momentum["domains"], protocol)
    comparison.update({field: no_momentum[field] for field in shared})
    comparison.update({
        "no_momentum_checkpoint": no_momentum["checkpoint_sha256"],
        "head_momentum_checkpoint": head_momentum["checkpoint_sha256"],
        "no_momentum_checkpoint_summary": no_momentum["checkpoint_summary_sha256"],
        "head_momentum_checkpoint_summary": head_momentum["checkpoint_summary_sha256"],
    })
    return comparison


def write_bundle(output: str | Path, no_momentum: dict, head_momentum: dict, data_manifest: dict, protocol: dict) -> dict:
    protocol = _authenticated_protocol(protocol)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if data_manifest != _expected_data_manifest(protocol):
        raise ValueError("bundle data manifest differs from the frozen inventory")
    comparison = _compare_results(no_momentum, head_momentum, protocol)
    if comparison["data_manifest_sha256"] != _canonical_sha256(data_manifest):
        raise ValueError("paired results do not match the data manifest")
    output.mkdir(parents=True)
    payloads = {
        "no_momentum.json": no_momentum,
        "head_momentum.json": head_momentum,
        "comparison.json": comparison,
        "data_manifest.json": data_manifest,
    }
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        **{field: comparison[field] for field in _BUNDLE_METADATA},
        "artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha256_file(output / name)}
            for name in payloads
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_bundle(output, protocol)


def verify_bundle(output: str | Path, protocol: dict | str | Path = _DEFAULT_PROTOCOL) -> dict:
    protocol = _authenticated_protocol(protocol)
    output = Path(output)
    required = {"no_momentum.json", "head_momentum.json", "comparison.json", "data_manifest.json"}
    if {path.name for path in output.iterdir()} != required | {"manifest.json"}:
        raise ValueError("Paloma bundle inventory differs")
    manifest = json.loads((output / "manifest.json").read_text())
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("Paloma bundle manifest schema differs")
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("Paloma bundle manifest inventory differs")
    for name, evidence in manifest["artifacts"].items():
        if not isinstance(evidence, dict) or set(evidence) != {"bytes", "sha256"}:
            raise ValueError(f"Paloma bundle artifact schema differs: {name}")
        path = output / name
        if path.stat().st_size != evidence.get("bytes") or _sha256_file(path) != evidence.get("sha256"):
            raise ValueError(f"Paloma bundle artifact hash differs: {name}")
    no_momentum = json.loads((output / "no_momentum.json").read_text())
    head_momentum = json.loads((output / "head_momentum.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    data_manifest = json.loads((output / "data_manifest.json").read_text())
    if data_manifest != _expected_data_manifest(protocol):
        raise ValueError("Paloma bundle data manifest differs")
    recomputed = _compare_results(no_momentum, head_momentum, protocol)
    if recomputed != comparison or comparison["data_manifest_sha256"] != _canonical_sha256(data_manifest):
        raise ValueError("Paloma bundle comparison differs from raw results")
    if manifest.get("status") != "complete" or any(
        manifest.get(field) != comparison.get(field) for field in _BUNDLE_METADATA
    ):
        raise ValueError("Paloma bundle manifest metadata differs")
    return {
        "manifest": manifest,
        "comparison": comparison,
        "no_momentum": no_momentum,
        "head_momentum": head_momentum,
        "data_manifest": data_manifest,
    }


def evaluate_checkpoint(
    *,
    arm: str,
    checkpoint_path: str | Path,
    data_root: str | Path,
    data_manifest_path: str | Path,
    extension_protocol_path: str | Path = _DEFAULT_PROTOCOL,
    device: torch.device,
) -> dict:
    extension_path = Path(extension_protocol_path)
    protocol = load_extension_protocol(extension_path)
    checkpoint = validate_checkpoint(checkpoint_path, arm, protocol)
    manifest = json.loads(Path(data_manifest_path).read_text())
    manifest_hash = validate_data_manifest(manifest, protocol, data_root)
    base_protocol_path = extension_path.with_name("protocol.json")
    base_protocol_value = json.loads(base_protocol_path.read_text())
    if _canonical_sha256(base_protocol_value) != protocol["base"]["source_protocol_canonical_sha256"]:
        raise ValueError("base protocol hash differs from the frozen Paloma protocol")

    from transformers import AutoTokenizer, LlamaForCausalLM

    from .core import load_protocol
    from .train import build_llama_config

    base_protocol = load_protocol(base_protocol_path)
    evaluation = protocol["paloma"]["evaluation"]
    if device.type != evaluation["device_type"]:
        raise ValueError("Paloma evaluation requires the frozen CUDA device")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Paloma evaluation requires CUDA BF16 support")
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["paloma"]["tokenizer"],
        revision=protocol["paloma"]["tokenizer_revision"],
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id != base_protocol["inputs"]["tokenizer_pad_id"]:
        raise ValueError("tokenizer special tokens differ from the frozen protocol")
    model = LlamaForCausalLM(build_llama_config(base_protocol))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    dtype = torch.bfloat16
    model.to(device=device, dtype=dtype).eval()
    model.config.use_cache = False
    rows = [
        evaluate_domain(
            model,
            tokenizer,
            Path(data_root) / domain["path"],
            domain,
            prefix_token=tokenizer.eos_token_id,
            max_seq_len=protocol["paloma"]["model_max_seq_len"],
            batch_size=protocol["paloma"]["batch_size"],
            pad_id=tokenizer.pad_token_id,
            device=device,
        )
        for domain in protocol["paloma"]["domains"]
    ]
    return {
        "status": "complete",
        "benchmark": protocol["paloma"]["name"],
        "arm": arm,
        "device_type": device.type,
        "dtype": "bfloat16",
        **checkpoint,
        "data_manifest_sha256": manifest_hash,
        "extension_protocol_sha256": _PROTOCOL_SHA256,
        "extension_code_sha256": _code_sha256(),
        "domains": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired Paloma evaluation for the frozen SCALE checkpoints")
    parser.add_argument("--no-momentum-checkpoint", type=Path, required=True)
    parser.add_argument("--head-momentum-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=_DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    protocol = load_extension_protocol(args.protocol)
    data_manifest = json.loads(args.data_manifest.read_text())
    validate_data_manifest(data_manifest, protocol, args.data_root)
    device = torch.device("cuda")
    shared = {
        "data_root": args.data_root,
        "data_manifest_path": args.data_manifest,
        "extension_protocol_path": args.protocol,
        "device": device,
    }
    no_momentum = evaluate_checkpoint(
        arm="no_momentum",
        checkpoint_path=args.no_momentum_checkpoint,
        **shared,
    )
    head_momentum = evaluate_checkpoint(
        arm="head_momentum",
        checkpoint_path=args.head_momentum_checkpoint,
        **shared,
    )
    write_bundle(args.output, no_momentum, head_momentum, data_manifest, protocol)


if __name__ == "__main__":
    main()
