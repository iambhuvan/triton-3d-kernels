"""Tests for K4: Batched Chamfer Distance kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.chamfer_ref import chamfer_distance_reference
from kernels.chamfer import batched_chamfer_distance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_point_clouds(B=2, N=256, M=256, device='cuda', dtype=torch.float32):
    pc1 = torch.randn(B, N, 3, device=device, dtype=dtype)
    pc2 = torch.randn(B, M, 3, device=device, dtype=dtype)
    return pc1, pc2


class TestChamferCorrectness:
    def test_basic_shapes(self):
        pc1, pc2 = make_point_clouds()
        loss, dist1, dist2 = batched_chamfer_distance(pc1, pc2)
        assert loss.shape == (2,)
        assert dist1.shape == (2, 256)
        assert dist2.shape == (2, 256)

    def test_matches_reference(self):
        pc1, pc2 = make_point_clouds()
        ref_loss, ref_d1, ref_d2 = chamfer_distance_reference(pc1, pc2)
        tri_loss, tri_d1, tri_d2 = batched_chamfer_distance(pc1, pc2)
        torch.testing.assert_close(tri_loss, ref_loss, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(tri_d1, ref_d1, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(tri_d2, ref_d2, atol=5e-3, rtol=5e-3)

    def test_identical_clouds_zero_distance(self):
        """Chamfer distance between identical point clouds should be 0."""
        pc = torch.randn(2, 64, 3, device='cuda')
        loss, dist1, dist2 = batched_chamfer_distance(pc, pc)
        torch.testing.assert_close(loss, torch.zeros(2, device='cuda'), atol=1e-6, rtol=1e-6)

    def test_symmetric(self):
        """CD(A, B) should equal CD(B, A) when N=M."""
        pc1, pc2 = make_point_clouds(N=128, M=128)
        loss_ab, _, _ = batched_chamfer_distance(pc1, pc2)
        loss_ba, _, _ = batched_chamfer_distance(pc2, pc1)
        torch.testing.assert_close(loss_ab, loss_ba, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("N,M", [
        (32, 64),
        (128, 256),
        (512, 512),
        (1024, 512),
    ])
    def test_asymmetric_shapes(self, N, M):
        pc1, pc2 = make_point_clouds(N=N, M=M)
        ref_loss, _, _ = chamfer_distance_reference(pc1, pc2)
        tri_loss, _, _ = batched_chamfer_distance(pc1, pc2)
        torch.testing.assert_close(tri_loss, ref_loss, atol=5e-3, rtol=5e-3)

    def test_non_negative(self):
        """Chamfer distance should always be non-negative."""
        pc1, pc2 = make_point_clouds()
        loss, dist1, dist2 = batched_chamfer_distance(pc1, pc2)
        assert torch.all(loss >= 0)
        assert torch.all(dist1 >= 0)
        assert torch.all(dist2 >= 0)

    def test_single_point(self):
        """CD of single points = squared Euclidean distance."""
        p1 = torch.tensor([[[1.0, 0.0, 0.0]]], device='cuda')  # (1, 1, 3)
        p2 = torch.tensor([[[0.0, 1.0, 0.0]]], device='cuda')   # (1, 1, 3)
        loss, _, _ = batched_chamfer_distance(p1, p2)
        # ||[1,0,0] - [0,1,0]||^2 = 2.0, both directions same
        expected = torch.tensor([4.0], device='cuda')  # 2.0 + 2.0
        torch.testing.assert_close(loss, expected, atol=5e-3, rtol=5e-3)
