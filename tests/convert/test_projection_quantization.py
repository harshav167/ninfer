from argparse import Namespace
import json

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec, ResourceSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_5 import build_model
from tools.quantize_qwen3_8 import convert, selected, source_identity
from .test_qwen3_5 import _config, _checkpoint


def test_draft_conversion_preserves_target_and_shared_context_bindings(tmp_path):
    cfg = _config()
    cfg['text_config'].update(hidden_size=128)
    draft_cfg = {
        'architectures': ['DFlash2DraftModel'], 'hidden_size': 128, 'vocab_size': 8,
        'num_hidden_layers': 1, 'intermediate_size': 128, 'num_attention_heads': 2,
        'num_key_value_heads': 2, 'head_dim': 64, 'max_position_embeddings': 128,
        'layer_types': ['sliding_attention'], 'sliding_window': 32,
        'dflash_config': {'target_layer_ids': [0], 'mask_token_id': 7,
                         'conv_kernel_size': 2, 'conv_group_size': 16,
                         'selector_rank': 4, 'selector_top_k': 2}}
    names = ['fc.weight'] + [f'layers.0.self_attn.{p}_proj.weight' for p in ('q', 'k', 'v', 'o')]
    names += [f'layers.0.mlp.{p}_proj.weight' for p in ('gate', 'up', 'down')]
    generator = torch.Generator().manual_seed(91)
    tensors = {name: torch.randn(128, 128, generator=generator).bfloat16() for name in names}
    with _checkpoint(tmp_path/'source', cfg, {'unused': torch.ones(1)}) as base, \
         _checkpoint(tmp_path/'draft', draft_cfg, tensors) as draft:
        (tmp_path/'source/model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'unused': 'model.safetensors'}}))
        model = build_model(base, components=('text', 'dflash2'), companions={'dflash2': draft})
        parameters = [p for name, p in model.parameters.items() if selected(name, 'draft')]
        specs = [TensorSpec(p.name, p.shape, 'bf16', 'contiguous_le_v1') for p in parameters]
        specs += [ResourceSpec(key, len(value)) for key, value in model.resources.items()]
        specs += [TensorSpec('target-weight', (128, 128), 'bf16', 'contiguous_le_v1')]
        bindings = {p.name: {'object': p.name} for p in parameters}
        for role in ('key', 'value'):
            bindings[f'dflash2/layers/0/attention/context_{role}'] = \
                bindings[f'dflash2/layers/0/attention/{role}']
        bindings['text/output_head'] = {'object': 'target-weight'}
        uses = [{'parameter': name, 'input': model.parameters[name].inputs[0],
                 'activation_policy': 'A16Only'} for name in bindings if name != 'text/output_head']
        with ArtifactWriter(tmp_path/'baseline.ninfer', specs, components=model.components,
                            bindings=bindings, uses=uses) as writer:
            for p in parameters:
                TensorOutput(writer, p.name).write_values(0, p.source.values())
            TensorOutput(writer, 'target-weight').write_values(0, torch.ones(128, 128).bfloat16())
            for key, value in model.resources.items():
                writer.write_object(key, value)
    args = Namespace(baseline=tmp_path/'baseline.ninfer', source=tmp_path/'source',
                     draft=tmp_path/'draft', mode='draft', calibration=None, rows=128,
                     device='cpu', damping=.01, out=tmp_path/'candidate.ninfer')
    convert(args)
    with Artifact(args.baseline) as baseline, Artifact(args.out) as candidate:
        assert candidate.directory.bindings == baseline.directory.bindings
        assert candidate.read_object('target-weight') == baseline.read_object('target-weight')
        assert candidate.directory.components == baseline.directory.components
        for key in model.resources:
            assert candidate.read_object(key) == baseline.read_object(key)
        assert all(candidate.object(p.name).format == 'nvfp4' for p in parameters)
        assert all(u['activation_policy'] == 'A16Only' for u in candidate.directory.uses)
        assert candidate.file_bytes < baseline.file_bytes


@pytest.mark.parametrize('name', [
    'text/output_head', 'text/token_embedding', 'vision/layers/0/attention/query',
    'text/layers/0/gdn/a_projection', 'text/layers/0/gdn/b_projection',
    'text/layers/3/attention/query', 'dflash2/candidate_selector/hidden_projection',
    'dflash2/layers/0/attention_conv/kernel_projection', 'dflash2/layers/0/mlp_conv/kernel_projection',
])
def test_sensitive_or_unrelated_parameters_are_not_selected(name):
    assert not selected(name, 'both')
