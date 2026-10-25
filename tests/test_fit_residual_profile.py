"""Independent small-geometry checks for reusable residual profile fitting."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from tools.fit_residual_profile import fit

ROOT = Path(__file__).resolve().parents[1]


class ResidualProfileFitTest(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '.agent-tmp' / 'residual-fit-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        anchors = {'comply': [[0, 1, 0], [0, -1, 0]],
                   'refuse': [[4, 0.5, 0], [4, -0.5, 0]],
                   'clean': [[0, 1, 0], [0, -1, 0]],
                   'disclaimer': [[0, 1, 3], [0, -1, 3]]}
        self.samples = []
        vectors = []
        for kind, values in anchors.items():
            for i, value in enumerate(values):
                label = f'{kind}:{i}'
                self.samples.append({'id': label, 'label': label})
                vectors.append([value] * 3)
        self.values = np.asarray(vectors, dtype='<f4')
        self.values.tofile(self.path / 'residuals.f32')
        self.manifest = {'schema_version': 1, 'artifact_id': 'a' * 32,
                         'hidden_size': 3, 'num_hidden_layers': 7,
                         'layers': [2, 3, 4], 'dtype': 'float32', 'byteorder': 'little',
                         'residual_point': 'layer_output_after_ffn_residual_add_before_dflash_capture',
                         'token_position': 'last_prompt_token', 'reasoning': 'enabled',
                         'dataset_sha256': 'b' * 64, 'samples': self.samples}
        self.save_manifest()

    def save_manifest(self):
        (self.path / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_orthogonal_fixture_has_known_directions_and_bounds(self):
        profile = fit(self.path)
        self.assertEqual([x['index'] for x in profile['layers']], [3, 4])
        for layer in profile['layers']:
            np.testing.assert_allclose(layer['refusal_direction'], [1, 0, 0], atol=1e-6)
            self.assertAlmostEqual(layer['refusal_reference'], 0)
            np.testing.assert_allclose(layer['disclaimer_direction'], [0, 0, 1], atol=1e-6)
            self.assertAlmostEqual(layer['disclaimer_reference'], -1.2)

    def test_degenerate_unused_layers_do_not_veto_the_edit_band(self):
        self.values[:, 0, :] = 0
        self.values.tofile(self.path / 'residuals.f32')
        for layer in fit(self.path)['layers']:
            np.testing.assert_allclose(layer['refusal_direction'], [1, 0, 0], atol=1e-6)
            self.assertAlmostEqual(layer['refusal_reference'], 0)
            np.testing.assert_allclose(layer['disclaimer_direction'], [0, 0, 1], atol=1e-6)
            self.assertAlmostEqual(layer['disclaimer_reference'], -1.2)

    def test_disclaimer_can_be_disabled_without_extra_anchor_groups(self):
        self.manifest['samples'] = self.samples[:4]
        self.values[:4].tofile(self.path / 'residuals.f32')
        self.save_manifest()
        for layer in fit(self.path, disclaimer_strength=-1)['layers']:
            self.assertEqual(layer['disclaimer_direction'], [0, 0, 0])
            self.assertEqual(layer['disclaimer_reference'], 0)

    def test_truncated_capture_is_rejected(self):
        (self.path / 'residuals.f32').write_bytes(b'\0' * 4)
        with self.assertRaises(ValueError):
            fit(self.path)

    def test_nonfinite_capture_is_rejected(self):
        self.values[0, 0, 0] = np.nan
        self.values.tofile(self.path / 'residuals.f32')
        with self.assertRaises(ValueError):
            fit(self.path)

    def test_degenerate_direction_is_rejected(self):
        self.values[2:4] = self.values[:2]
        self.values.tofile(self.path / 'residuals.f32')
        with self.assertRaises(ValueError):
            fit(self.path)

    def test_wrong_observation_position_is_rejected(self):
        self.manifest['token_position'] = 'last_generated_token'
        self.save_manifest()
        with self.assertRaises(ValueError):
            fit(self.path)


if __name__ == '__main__':
    unittest.main()
