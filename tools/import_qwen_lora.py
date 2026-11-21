"""Pack a dense Qwen PEFT adapter into a V3 LoRA companion without changing precision."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open

from .artifact.reader import Artifact
from .artifact.schema import TensorSpec
from .artifact.writer import ArtifactWriter

ROLES = {
    "mlp.gate_proj": "mlp/gate",
    "mlp.up_proj": "mlp/up",
    "mlp.down_proj": "mlp/down",
    "self_attn.q_proj": "attention/query",
    "self_attn.k_proj": "attention/key",
    "self_attn.v_proj": "attention/value",
    "self_attn.o_proj": "attention/output",
}
KEY = re.compile(r"base_model\.model\.(?:model\.language_model|model)\.layers\.(\d+)\.(.+)\.lora_A\.weight")


def convert(source: Path, base: Path, output: Path) -> None:
    peft = json.loads((source / "adapter_config.json").read_text())
    if (peft["peft_type"] != "LORA" or peft["r"] != 32 or peft["bias"] != "none"
            or any(peft.get(key) for key in (
                "fan_in_fan_out", "use_rslora", "use_dora", "use_qalora", "use_bdlora",
                "use_velora", "lora_bias", "alora_invocation_tokens", "rank_pattern",
                "alpha_pattern", "modules_to_save", "target_parameters", "layer_replication",
            ))):
        raise ValueError("expected plain rank-32 PEFT LoRA without bias or per-layer scaling")
    scale = float(peft["lora_alpha"]) / 32
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid LoRA alpha")

    with Artifact(base) as target, safe_open(source / "adapter_model.safetensors", framework="numpy") as src:
        cfg = target.directory.components["text"]["config"]
        d = cfg["head_dim"]
        heads = cfg["num_attention_heads"]
        index = {obj.id: obj for obj in target.objects}
        specs, bindings, writes = [], {}, []
        keys = set(src.keys())
        used = set()

        def shape(name):
            binding = target.directory.bindings[name]
            if "object" in binding:
                return index[binding["object"]].shape
            parts = binding["parts"]
            k = index[parts[0]["object"]].shape[1]
            return (sum(p["range"][1] - p["range"][0] for p in parts) // k, k)

        def add(name, extent, key, head_part=None):
            specs.append(TensorSpec(name, extent, "fp32", "contiguous_le_v1"))
            bindings[name] = {"object": name}
            writes.append((name, key, head_part))

        for key in sorted(keys):
            match = KEY.fullmatch(key)
            if match is None:
                continue
            layer, role = match.groups()
            if role not in ROLES:
                raise ValueError(f"unsupported LoRA target: {role}")
            name = f"text/layers/{int(layer)}/{ROLES[role]}"
            n, k = shape(name)
            bkey = key.replace(".lora_A.", ".lora_B.")
            gated = role == "self_attn.q_proj"
            for tensor, expected in ((key, [32, k]), (bkey, [n * (2 if gated else 1), 32])):
                view = src.get_slice(tensor)
                if view.get_dtype() != "F32" or view.get_shape() != expected:
                    raise ValueError(f"{tensor}: expected FP32 {expected}")
            if name + "/a" in bindings:
                raise ValueError(f"duplicate LoRA target: {name}")
            add(name + "/a", (32, k), key)
            add(name + "/b", (32, n), bkey, 0 if gated else -1)
            if gated:
                gate = f"text/layers/{int(layer)}/attention/gate"
                if n != heads * d or shape(gate) != (n, k):
                    raise ValueError(f"{name}: invalid Q/gate geometry")
                bindings[gate + "/a"] = bindings[name + "/a"]
                add(gate + "/b", (32, n), bkey, 1)
            used.update((key, bkey))
        if not used or used != keys:
            raise ValueError(f"unrecognized adapter tensors: {sorted(keys - used)}")

        with ArtifactWriter(output, specs, bindings=bindings, components={"lora": {
            "config": {"rank": 32, "scale": scale, "base_artifact_id": target.artifact_id.hex()},
        }}, provenance={"trained_base": peft["base_model_name_or_path"]}) as writer:
            for name, key, part in writes:
                weights = src.get_tensor(key)
                if part in (0, 1):
                    weights = weights.reshape(heads, 2, d, 32)[:, part, :, :].reshape(-1, 32)
                if part is not None:
                    weights = weights.T
                writer.write_object(name, memoryview(np.ascontiguousarray(weights, dtype="<f4")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="PEFT adapter directory")
    parser.add_argument("base", type=Path, help="target .ninfer (read-only)")
    parser.add_argument("output", type=Path, help="adapter-only .ninfer")
    args = parser.parse_args()
    convert(args.source, args.base, args.output)


if __name__ == "__main__":
    main()
