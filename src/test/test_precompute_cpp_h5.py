import unittest
from unittest import mock

import numpy as np

from src.scripts import precompute_cpp_h5


class PrecomputeCppH5Test(unittest.TestCase):
    def test_compute_cpp_pair_passes_torch_device_options(self):
        f_axis = np.array([-0.5, 0.5], dtype=np.float32)
        alpha_axis = np.array([-1.0, 1.0], dtype=np.float32)
        image0 = np.ones((2, 2), dtype=np.float32)
        image1 = np.full((2, 2), 2.0, dtype=np.float32)

        with mock.patch.object(
            precompute_cpp_h5,
            "compute_fam_grid_segmented",
            side_effect=[
                (image0, f_axis, alpha_axis),
                (image1, f_axis, alpha_axis),
            ],
        ) as compute_fam_grid_segmented:
            cpp, returned_f_axis, returned_alpha_axis = precompute_cpp_h5.compute_cpp_pair(
                np.array([1 + 0j], dtype=np.complex64),
                np.array([2 + 0j], dtype=np.complex64),
                segment_samples=8,
                segment_hop_samples=4,
                fam_merge="mean",
                f_bins=2,
                alpha_bins=2,
                device="mps",
                pair_chunk_size=4096,
            )

        self.assertEqual(compute_fam_grid_segmented.call_count, 2)
        for call in compute_fam_grid_segmented.call_args_list:
            self.assertEqual(call.kwargs["device"], "mps")
            self.assertEqual(call.kwargs["pair_chunk_size"], 4096)

        np.testing.assert_array_equal(cpp, np.stack([image0, image1], axis=0))
        self.assertEqual(cpp.dtype, np.float32)
        np.testing.assert_array_equal(returned_f_axis, f_axis)
        np.testing.assert_array_equal(returned_alpha_axis, alpha_axis)


if __name__ == "__main__":
    unittest.main()
