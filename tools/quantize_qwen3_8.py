"""Calibrate GDN GPTQ and replace only selected Qwen3.8 artifact projections.

Run with ``python -m tools.quantize_qwen3_8 --help``. Calibration is offline
checkpoint tooling, not a model-serving route. Conversion preserves every
unselected object byte-for-byte, including vision, target head and proposal head.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec, ResourceSpec, TensorObject
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_5 import build_model
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.quantization.nvfp4_gptq import inverse_factor, quantize_gptq
from tools.convert.quantization.stored_scale import quantize_stored_scale


def source_identity(root):
    root = Path(root).resolve()
    names = ("config.json", "model.safetensors.index.json")
    return {"path": str(root), "files": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }}


def selected(name, mode):
    role = name.rsplit("/", 1)[-1]
    if mode in ("gdn", "both") and name.startswith("text/layers/"):
        return "/gdn/" in name and role in ("query", "key", "value", "z", "output")
    if mode in ("draft", "both") and name.startswith("dflash2/"):
        return (name == "dflash2/feature_projection" or
                "/attention/" in name and role in ("query", "key", "value", "output") or
                "/mlp/" in name and role in ("gate", "up", "down"))
    return False


def calibration_key(name):
    layer = name.split("/")[2]
    return f"layer-{layer}-{'out' if name.endswith('/output') else 'in'}"


def plan_replacements(artifact, model, mode):
    parents = defaultdict(list)
    for name, parameter in model.parameters.items():
        if not selected(name, mode):
            continue
        binding = artifact.directory.bindings[name]
        if "object" in binding:
            parts = [{"object": binding["object"], "range": [0, math.prod(parameter.shape)]}]
        else:
            parts = binding["parts"]
        if len(parts) != 1 or len(parameter.shape) != 2:
            raise ValueError(f"{name}: expected a whole projection in one parent")
        part = parts[0]
        low, high = part["range"]
        obj = artifact.object(part["object"])
        if not isinstance(obj, TensorObject) or len(obj.shape) != 2 or obj.shape[1] != parameter.shape[1]:
            raise ValueError(f"{name}: incompatible parent geometry")
        if high - low != math.prod(parameter.shape) or low % obj.shape[1]:
            raise ValueError(f"{name}: incompatible binding range")
        if parameter.source.encoding() is not None:
            raise ValueError(f"{name}: quantization requires original BF16 source, not encoded weights")
        parents[obj.id].append((low, high, parameter))
    if not parents:
        raise ValueError("no requested projections found")
    for object_id, pieces in parents.items():
        pieces.sort(key=lambda item: item[0])
        cursor = 0
        for low, high, parameter in pieces:
            if low != cursor:
                raise ValueError(f"{object_id}: selected projections must cover the entire parent")
            cursor = high
            # Demand a represented BF16 source; never silently dequantize another quant.
            if parameter.source.values(0, 1).dtype != torch.bfloat16:
                raise ValueError(f"{parameter.name}: expected original BF16 source")
        if cursor != math.prod(artifact.object(object_id).shape):
            raise ValueError(f"{object_id}: partial parent replacement is unsupported")
    return dict(parents)


def _chunks(pieces, k, rows):
    for low, high, parameter in pieces:
        for begin in range(0, high - low, rows * k):
            end = min(high - low, begin + rows * k)
            yield (low + begin) // k, parameter.source.values(begin, end).reshape(-1, k)


def convert(args):
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    with ExitStack() as stack:
        artifact = stack.enter_context(Artifact(args.baseline))
        base = stack.enter_context(SafetensorsSource(args.source))
        companions = {}
        components = ("text",)
        if args.mode in ("draft", "both"):
            if args.draft is None:
                raise ValueError("draft conversion requires --draft with the original BF16 drafter")
            companions["dflash2"] = stack.enter_context(SafetensorsSource(args.draft))
            components += ("dflash2",)
        model = build_model(base, components=components, companions=companions)
        actual = artifact.directory.components["text"]["config"]
        for key in ("hidden_size", "vocab_size", "num_hidden_layers", "layer_types"):
            if model.config[key] != actual[key]:
                raise ValueError(f"source and target {key} differ")
        parents = plan_replacements(artifact, model, args.mode)
        calibration = None
        if args.mode in ("gdn", "both"):
            if args.calibration is None:
                raise ValueError("GDN GPTQ requires --calibration")
            calibration = json.loads((args.calibration / "manifest.json").read_text())
            if calibration["source"] != source_identity(args.source):
                raise ValueError("calibration belongs to a different source checkpoint")
        uses = deepcopy(list(artifact.directory.uses))
        specs = [TensorSpec(o.id, o.shape, "nvfp4" if o.id in parents else o.format,
                            "block_scale_k16_m128x4_v1" if o.id in parents else o.layout)
                 if isinstance(o, TensorObject) else ResourceSpec(o.id, o.bytes, o.encoding)
                 for o in artifact.objects]
        changed_names = {p.name for pieces in parents.values() for _, _, p in pieces}
        # Context K/V alias the same QKV parent, so their permissions must change too.
        changed_names.update(name.replace("/attention/", "/attention/context_")
                             for name in tuple(changed_names)
                             if name.startswith("dflash2/") and name.endswith(("/key", "/value")))
        auxiliary_values = {}
        for use in uses:
            name = use["parameter"]
            if name not in changed_names:
                continue
            use["activation_policy"] = "AllowA4" if name.startswith("text/") else "A16Only"
            use.pop("auxiliaries", None)
            if name.startswith("text/"):
                key = calibration_key(name)
                maximum = calibration["inputs"][key]["amax"]
                if not math.isfinite(maximum) or maximum <= 0:
                    raise ValueError(f"{key}: invalid calibrated activation range")
                object_id = "nvfp4_gdn/activation/" + key
                auxiliary_values[object_id] = struct.pack("<f", 2688.0 / maximum)
                use["auxiliaries"] = {"activation_input_divisor": {"object": object_id}}
        specs.extend(TensorSpec(key, (), "fp32", "contiguous_le_v1") for key in auxiliary_values)
        provenance = deepcopy(artifact.directory.provenance)
        provenance["nvfp4_projection_conversion"] = {
            "baseline": str(args.baseline.resolve()), "baseline_id": artifact.artifact_id.hex(),
            "source": source_identity(args.source), "mode": args.mode,
            "draft": str(args.draft.resolve()) if args.draft else None,
            "gdn_method": "gptq" if calibration else None,
            "calibration": calibration, "draft_method": "stored_scale_mse",
            "damping": args.damping, "changed_objects": sorted(parents),
        }
        report = []
        with ArtifactWriter(args.out, specs, components=artifact.directory.components,
                            bindings=artifact.directory.bindings, uses=uses,
                            metadata=artifact.directory.metadata, provenance=provenance) as writer:
            for obj in artifact.objects:
                if obj.id not in parents:
                    writer.write_object(obj.id, artifact.iter_object(obj.id))
                    continue
                pieces = parents[obj.id]
                k = obj.shape[1]
                maximum = max(float(w.abs().max()) for _, w in _chunks(pieces, k, args.rows))
                if not math.isfinite(maximum):
                    raise ValueError(f"{obj.id}: nonfinite source")
                divisor = struct.pack("<f", 2688.0 / maximum if maximum else 1.0)
                is_gdn = pieces[0][2].name.startswith("text/")
                factor = dead = None
                if is_gdn:
                    keys = {calibration_key(p.name) for _, _, p in pieces}
                    if len(keys) != 1:
                        raise ValueError("fused GDN projections must share the same calibrated input")
                    key = keys.pop()
                    data = torch.load(args.calibration / f"{key}.pt", map_location=args.device,
                                      weights_only=True)
                    if data["tokens"] != calibration["inputs"][key]["tokens"] or data["tokens"] <= 0:
                        raise ValueError(f"{key}: invalid calibration token count")
                    factor, dead = inverse_factor(data["gram"] / data["tokens"], args.damping)
                    del data
                output = TensorOutput(writer, obj.id)
                for row, weight in _chunks(pieces, k, args.rows):
                    if is_gdn:
                        encoded = quantize_gptq(weight.to(args.device), factor, dead, divisor)
                    else:
                        encoded = quantize_stored_scale(weight, "nvfp4", divisor, device=args.device)
                    output.write_codes(row, encoded.codes, encoded.scales, divisor)
                del factor, dead
                report.append({"object": obj.id, "parameters": [p.name for _, _, p in pieces],
                               "old_bytes": obj.bytes, "new_bytes": writer.by_id[obj.id].bytes})
                print(json.dumps(report[-1]), flush=True)
            for key, value in auxiliary_values.items():
                writer.write_object(key, value)
        args.out.with_suffix(args.out.suffix + ".quantization.json").write_text(
            json.dumps({"provenance": provenance["nvfp4_projection_conversion"], "objects": report}, indent=2))


class _CheckpointWeights(Mapping):
    """Read one text parameter on demand; never materialize the 55 GB state dict."""
    def __init__(self, source, names):
        self.source = source
        self.names = tuple(names)

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        tensor = "model.language_model." + name
        info = self.source.describe(tensor)
        if info.dtype != "BF16":
            raise ValueError(f"{tensor}: calibration requires original BF16 weights")
        return self.source.read_flat(tensor).reshape(info.shape)


def calibrate(args):
    from accelerate import cpu_offload, init_empty_weights
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if args.samples <= 0 or args.sequence_length <= 0:
        raise ValueError("calibration samples and sequence length must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    source = source_identity(args.source)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    sequences = []
    with args.data.open() as stream:
        for line in stream:
            item = json.loads(line)
            if "messages" in item:
                ids = tokenizer.apply_chat_template(item["messages"], tools=item.get("tools"),
                                                    tokenize=True, add_generation_prompt=False,
                                                    enable_thinking=True)
            else:
                ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"]
            for begin in range(0, len(ids), args.sequence_length):
                part = ids[begin:begin + args.sequence_length]
                if len(part) >= 16:
                    sequences.append(part)
                if len(sequences) >= args.samples:
                    break
            if len(sequences) >= args.samples:
                break
    if not sequences:
        raise ValueError("calibration data contains no usable sequences")
    config = AutoConfig.from_pretrained(args.source, local_files_only=True).text_config
    with init_empty_weights(include_buffers=False):
        model = AutoModel.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    store = SafetensorsSource(args.source)
    weights = _CheckpointWeights(store, dict(model.named_parameters()))
    cpu_offload(model, execution_device=torch.device(f"cuda:{args.gpu}"), state_dict=weights,
                preload_module_classes=["Qwen3_5DecoderLayer"])
    model.eval()
    statistics = {}
    handles = []

    def capture(key):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f"{key}: nonfinite calibration activation")
            if key not in statistics:
                statistics[key] = {"gram": torch.zeros((x.shape[1], x.shape[1])),
                                   "tokens": 0, "amax": 0.0}
            state = statistics[key]
            state["amax"] = max(state["amax"], float(x.abs().max()))
            state["tokens"] += x.shape[0]
            state["gram"].add_((x.T @ x).cpu())
        return hook

    for name, module in model.named_modules():
        if ".linear_attn." not in name or not name.endswith((".in_proj_qkv", ".out_proj")):
            continue
        layer = name.split("layers.")[1].split(".")[0]
        key = f"layer-{layer}-{'out' if name.endswith('.out_proj') else 'in'}"
        handles.append(module.register_forward_pre_hook(capture(key)))
    if len(handles) != 96:
        raise ValueError(f"expected 96 GDN input sites for Qwen3.8-27B, found {len(handles)}")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for index, sequence in enumerate(sequences):
                ids = torch.tensor([sequence], dtype=torch.long, device=f"cuda:{args.gpu}")
                # No vocabulary-sized logits and no generation/KV cache allocation.
                model(input_ids=ids, use_cache=False, return_dict=True)
                print(f"calibration {index + 1}/{len(sequences)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        store.close()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    args.out.mkdir(parents=True)
    manifest = {"source": source, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "sequences": len(sequences), "sequence_length": args.sequence_length, "inputs": {}}
    for key, state in statistics.items():
        torch.save({"gram": state["gram"], "tokens": state["tokens"]}, args.out / f"{key}.pt")
        manifest["inputs"][key] = {"tokens": state["tokens"], "amax": state["amax"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="collect GDN input Gram matrices from original BF16 source")
    cal.add_argument("--source", type=Path, required=True)
    cal.add_argument("--data", type=Path, required=True, help="JSONL text or messages, separate from evaluation")
    cal.add_argument("--out", type=Path, required=True)
    cal.add_argument("--samples", type=int, default=256)
    cal.add_argument("--sequence-length", type=int, default=2048)
    cal.add_argument("--gpu", type=int, default=0)
    cal.set_defaults(run=calibrate)
    conv = sub.add_parser("convert", help="write a new artifact, preserving all unselected objects")
    conv.add_argument("--baseline", type=Path, required=True)
    conv.add_argument("--source", type=Path, required=True)
    conv.add_argument("--draft", type=Path)
    conv.add_argument("--calibration", type=Path)
    conv.add_argument("--mode", choices=("gdn", "draft", "both"), required=True)
    conv.add_argument("--out", type=Path, required=True)
    conv.add_argument("--device", default="cuda:0")
    conv.add_argument("--rows", type=int, default=128)
    conv.add_argument("--damping", type=float, default=0.01)
    conv.set_defaults(run=convert)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
