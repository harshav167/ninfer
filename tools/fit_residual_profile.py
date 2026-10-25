"""Fit a startup-fixed residual profile from native NInfer activation captures.

This does not load model weights or run inference. Capture with
ninfer-residual-capture first. Labels are comply:N, refuse:N, clean:N and
disclaimer:N; each opening is averaged across the fixed calibration corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def normalize(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm < 1e-10).any():
        raise ValueError('calibration produced a degenerate direction')
    return value / norm


def fit(capture: Path, edit_strength: float = 2.0, disclaimer_strength: float = 0.4) -> dict:
    if not math.isfinite(edit_strength) or not math.isfinite(disclaimer_strength):
        raise ValueError('strengths must be finite')
    manifest = json.loads((capture / 'manifest.json').read_text())
    if (manifest['schema_version'] != 1 or manifest['dtype'] != 'float32'
            or manifest['byteorder'] != 'little'
            or manifest['token_position'] != 'last_prompt_token'
            or manifest['residual_point'] != 'layer_output_after_ffn_residual_add_before_dflash_capture'
            or manifest['reasoning'] != 'enabled'):
        raise ValueError('unsupported capture contract')
    if not re.fullmatch(r'[0-9a-f]{32}', manifest['artifact_id']):
        raise ValueError('invalid artifact identity')
    samples, layers = manifest['samples'], manifest['layers']
    hidden, total_layers = manifest['hidden_size'], manifest['num_hidden_layers']
    if (not samples or hidden <= 0 or total_layers <= 0 or not layers
            or layers != sorted(set(layers)) or layers[0] < 0 or layers[-1] >= total_layers
            or len({s['id'] for s in samples}) != len(samples)):
        raise ValueError('invalid sample or layer geometry')
    selected = list(range(3 * total_layers // 7, 2 * total_layers // 3 + 1))
    if not set(selected).issubset(layers):
        raise ValueError('capture is missing the selected intervention layers')
    payload = capture / 'residuals.f32'
    expected = len(samples) * len(layers) * hidden * 4
    if payload.stat().st_size != expected:
        raise ValueError('activation payload size differs from capture manifest')
    values = np.fromfile(payload, dtype='<f4').reshape(len(samples), len(layers), hidden)
    if not np.isfinite(values).all():
        raise ValueError('non-finite captured activation')
    values = values[:, [layers.index(layer) for layer in selected], :]
    groups: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        if not re.fullmatch(r'(comply|refuse|clean|disclaimer):\d+', sample['label']):
            raise ValueError(f"invalid calibration label: {sample['label']}")
        groups.setdefault(sample['label'], []).append(i)
    if len({len(indices) for indices in groups.values()}) != 1:
        raise ValueError('opening groups must have equal calibration sample counts')

    def anchors(kind: str, minimum: int) -> np.ndarray:
        keys = sorted((k for k in groups if k.startswith(kind + ':')),
                      key=lambda k: int(k.split(':')[1]))
        if len(keys) < minimum:
            raise ValueError(f'missing {kind} calibration openings')
        return np.stack([values[groups[k]].astype(np.float64).mean(axis=0) for k in keys])

    comply, refuse = anchors('comply', 2), anchors('refuse', 2)
    mc, mr = comply.mean(axis=0), refuse.mean(axis=0)
    direction = mr - mc
    # Sequential spread removal matches the original calibration recipe;
    # these spread vectors are not asserted to form an orthogonal basis.
    for opening in comply:
        spread = opening - mc
        spread /= np.linalg.norm(spread, axis=-1, keepdims=True) + 1e-6
        direction -= np.sum(direction * spread, axis=-1, keepdims=True) * spread
    direction = normalize(direction)
    projection = np.sum(comply * direction, axis=-1)
    reference = projection.mean(axis=0) - edit_strength * projection.std(axis=0)
    disc_direction, disc_reference = np.zeros_like(direction), np.zeros_like(reference)
    if disclaimer_strength >= 0:
        clean, disclaimer = anchors('clean', 2), anchors('disclaimer', 2)
        disc_direction = normalize(disclaimer.mean(axis=0) - clean.mean(axis=0))
        clean_level = np.sum(clean * disc_direction, axis=-1).mean(axis=0)
        disclaimer_level = np.sum(disclaimer * disc_direction, axis=-1).mean(axis=0)
        disc_reference = clean_level - disclaimer_strength * (disclaimer_level - clean_level)
    records = []
    for slot, layer in enumerate(selected):
        records.append({'index': layer, 'refusal_direction': direction[slot].astype(np.float32).tolist(),
                        'refusal_reference': float(reference[slot]),
                        'disclaimer_direction': disc_direction[slot].astype(np.float32).tolist(),
                        'disclaimer_reference': float(disc_reference[slot])})
    return {'schema_version': 1, 'artifact_id': manifest['artifact_id'],
            'hidden_size': hidden, 'num_hidden_layers': total_layers, 'layers': records,
            'provenance': {'method': 'fixed-corpus-opening-means-v1',
                'source_recipe': 'heterodoxin/AbliteratedOutTheBox strata dfc8bf24',
                'capture_manifest_sha256': hashlib.sha256((capture / 'manifest.json').read_bytes()).hexdigest(),
                'dataset_sha256': manifest['dataset_sha256'],
                'sample_count': len(samples), 'opening_group_counts': {k: len(v) for k, v in groups.items()},
                'edit_strength': edit_strength, 'disclaimer_strength': disclaimer_strength,
                'reasoning': 'enabled', 'gate': 'none; profile fixed per Engine',
                'limitations': 'A calibration adaptation, not proof of refusal-removal or quality retention.'}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--edit-strength', type=float, default=2.0)
    parser.add_argument('--disclaimer-strength', type=float, default=0.4)
    args = parser.parse_args()
    profile = fit(args.capture, args.edit_strength, args.disclaimer_strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(profile, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'output': str(args.output), 'artifact_id': profile['artifact_id'],
                      'edited_layers': [r['index'] for r in profile['layers']]}))


if __name__ == '__main__':
    main()
