"""Task 4 TTA views and OOF helpers (CPU only, no data needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Task4'))
import oof  # noqa: E402
import stress  # noqa: E402
import train as t4  # noqa: E402


class TTAViewTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = torch.rand(3, 3, 16, 32)

    def test_views_keep_shape(self):
        for view in t4.KNOWN_VIEWS[1:]:
            if view.startswith('size'):
                continue  # size views change the input size inside Preprocess, see the next test
            self.assertEqual(t4.apply_view(self.batch, view).shape, self.batch.shape, view)

    def test_size_views_resize_in_preprocess(self):
        preprocess = t4.Preprocess(torch.device('cpu'), 128, 256, 0.0)
        images = (torch.rand(2, 3, 120, 230) * 255).to(torch.uint8)
        self.assertEqual(tuple(preprocess(images, False, view='size1.25').shape[2:]), (160, 320))
        self.assertEqual(tuple(preprocess(images, False, view='size0.875').shape[2:]), (112, 224))
        self.assertEqual(tuple(preprocess(images, False, view='id').shape[2:]), (128, 256))

    def test_hflip_twice_is_identity(self):
        self.assertTrue(torch.equal(t4.apply_view(t4.apply_view(self.batch, 'hflip'), 'hflip'), self.batch))

    def test_zero_rotation_and_unit_zoom_are_identity(self):
        self.assertTrue(torch.allclose(t4.affine(self.batch, 0.0, 1.0), self.batch, atol=1e-5))

    def test_rotation_blackens_corners(self):
        turned = t4.apply_view(torch.ones(1, 3, 16, 32), 'rot+16')
        self.assertLess(float(turned[0, :, 0, 0].max()), 0.5)
        self.assertGreater(float(turned[0, :, 8, 16].min()), 0.99)

    def test_autocontrast_brightens_dark_frames_only(self):
        grey = torch.rand(1, 1, 16, 32).expand(1, 3, 16, 32)  # brightness spans the full range
        dark = grey * 0.2
        self.assertGreater(float(t4.apply_view(dark, 'auto').mean()), float(dark.mean()) * 2)
        full = grey
        self.assertLess(float((t4.apply_view(full, 'auto') - full).abs().mean()), 0.03)

    def test_unknown_view_raises(self):
        with self.assertRaises(ValueError):
            t4.apply_view(self.batch, 'spin')


class OOFHelperTest(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        probs = oof.softmax(np.array([[[1.0, 2.0, 3.0, 0.0, -1.0]]]))
        self.assertTrue(np.allclose(probs.sum(-1), 1.0))
        self.assertEqual(int(probs.argmax(-1)[0, 0]), 2)

    def test_presets_use_known_views(self):
        for views in oof.PRESET_VIEW_SETS.values():
            self.assertTrue(set(views) <= set(t4.KNOWN_VIEWS))
        self.assertTrue(set(t4.TTA_VIEWS) <= set(t4.KNOWN_VIEWS))

    def test_parse_group_with_and_without_own_folds(self):
        self.assertEqual(oof.parse_group('cnt=cnt_sc_f{fold}', [0, 1, 2]), ('cnt', 'cnt_sc_f{fold}', [0, 1, 2]))
        self.assertEqual(oof.parse_group('ev=ev2sRS_sc_f{fold}@0,1', [0, 1, 2]), ('ev', 'ev2sRS_sc_f{fold}', [0, 1]))
        with self.assertRaises(SystemExit):
            oof.parse_group('cnt=cnt_sc_f0', [0])

    def test_blend_averages_only_groups_that_cover_a_row(self):
        full = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
        partial = np.array([[0.2, 0.8], [np.nan, np.nan]], dtype=np.float32)
        mixed = oof.blend([full, partial], [1.0, 1.0])
        self.assertTrue(np.allclose(mixed, [[0.5, 0.5], [0.6, 0.4]]))
        self.assertTrue(np.allclose(oof.blend([full, full * 0.5], [3.0, 1.0]), full * 0.875))


class StressDamageTest(unittest.TestCase):
    def test_every_damage_keeps_shape_range_and_repeats(self):
        batch = torch.rand(2, 3, 32, 64)
        for kind in stress.DAMAGES:
            first = stress.damage(batch.clone(), kind, torch.Generator().manual_seed(1))
            again = stress.damage(batch.clone(), kind, torch.Generator().manual_seed(1))
            self.assertEqual(first.shape, batch.shape, kind)
            self.assertGreaterEqual(float(first.min()), 0.0, kind)
            self.assertLessEqual(float(first.max()), 1.0 + 1e-6, kind)
            self.assertTrue(torch.allclose(first, again), kind)

    def test_clean_is_untouched(self):
        batch = torch.rand(1, 3, 8, 16)
        self.assertTrue(torch.equal(stress.damage(batch, 'clean', torch.Generator()), batch))


if __name__ == '__main__':
    unittest.main()
