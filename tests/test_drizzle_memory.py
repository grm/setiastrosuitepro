"""Drizzle rejection storage and light-intermediate purge."""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from setiastro.saspro.stacking_suite import (  # noqa: E402
    build_rejection_layers,
    light_intermediate_files,
    paint_rejection_items,
    prune_empty_intermediate_dirs,
    record_rejection_pixels,
)


class RejectionPixelTests(unittest.TestCase):
    def test_sparse_band_stores_coordinates_not_the_tile(self):
        mask = np.zeros((16, 9576), dtype=bool)
        mask[3, 10] = True
        mask[3, 4000] = True
        mask[15, 9575] = True
        bucket = []
        record_rejection_pixels(bucket, 0, 32, mask)
        self.assertEqual(len(bucket), 1)
        kind, ys, xs = bucket[0]
        self.assertEqual(kind, "px")
        self.assertEqual(ys.dtype, np.uint32)
        self.assertLess(ys.nbytes + xs.nbytes, mask.nbytes)
        dest = np.zeros((64, 9576), dtype=bool)
        self.assertEqual(paint_rejection_items(bucket, dest), 3)
        self.assertTrue(dest[35, 10] and dest[35, 4000] and dest[47, 9575])
        self.assertEqual(int(dest.sum()), 3)

    def test_empty_mask_records_nothing(self):
        bucket = []
        record_rejection_pixels(bucket, 0, 0, np.zeros((16, 100), dtype=bool))
        self.assertEqual(bucket, [])

    def test_legacy_tile_and_point_still_paint(self):
        tile = np.zeros((2, 3), dtype=bool)
        tile[0, 1] = True
        dest = np.zeros((5, 5), dtype=bool)
        n = paint_rejection_items([(1, 2, tile), (4, 4)], dest)
        self.assertEqual(n, 2)
        self.assertTrue(dest[2, 2] and dest[4, 4])
        self.assertEqual(int(dest.sum()), 2)

    def test_layers_count_compact_pixels(self):
        bucket = []
        mask = np.zeros((2, 4), dtype=bool)
        mask[0, 1] = True
        record_rejection_pixels(bucket, 0, 0, mask)
        rej_cnt, _frac, _any = build_rejection_layers({"a.fit": bucket}, 4, 4, n_frames=2)
        self.assertEqual(int(rej_cnt[0, 1]), 1)


class LightIntermediateTests(unittest.TestCase):
    def test_only_files_inside_intermediate_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            cal = os.path.join(root, "Calibrated")
            raw = os.path.join(root, "raw")
            master = os.path.join(root, "Master_Light")
            os.makedirs(cal)
            os.makedirs(raw)
            os.makedirs(master)
            inside = os.path.join(cal, "abc_frame_c.fit")
            outside = os.path.join(raw, "frame.fit")
            master_path = os.path.join(master, "MasterLight.fit")
            sat = os.path.join(cal, "abc_frame_c_satmask.npy")
            for path in (inside, outside, master_path, sat):
                open(path, "wb").close()
            found = light_intermediate_files(root, [inside, outside, master_path])
            self.assertEqual(set(found), {os.path.realpath(inside), os.path.realpath(sat)})

    def test_symlink_escape_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            cal = os.path.join(root, "Calibrated")
            os.makedirs(cal)
            outside_dir = tempfile.mkdtemp()
            try:
                secret = os.path.join(outside_dir, "secret.fit")
                open(secret, "wb").close()
                link = os.path.join(cal, "linked.fit")
                os.symlink(secret, link)
                self.assertEqual(light_intermediate_files(root, [link]), [])
            finally:
                os.remove(secret)
                os.rmdir(outside_dir)

    def test_cfa_sidecar_and_empty_dir_prune(self):
        with tempfile.TemporaryDirectory() as root:
            norm = os.path.join(root, "Normalized_Images", "set")
            os.makedirs(norm)
            frame = os.path.join(norm, "abc_frame_n.fit")
            cfa = os.path.join(norm, "abc_frame_n_cfa.fit")
            open(frame, "wb").close()
            open(cfa, "wb").close()
            found = light_intermediate_files(root, [frame])
            self.assertEqual(
                set(found), {os.path.realpath(frame), os.path.realpath(cfa)}
            )
            os.remove(frame)
            os.remove(cfa)
            prune_empty_intermediate_dirs(root)
            self.assertFalse(os.path.isdir(os.path.join(root, "Normalized_Images")))
            self.assertTrue(os.path.isdir(root))


if __name__ == "__main__":
    unittest.main()
