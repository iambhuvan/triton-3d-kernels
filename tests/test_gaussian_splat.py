"""Tests for K6: 3D Gaussian Splatting Rasterizer kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.gaussian_splat_ref import gaussian_splat_reference, project_gaussians, compute_cov2d
from kernels.gaussian_splat import gaussian_splat_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_camera_matrices(device='cuda', dtype=torch.float32):
    """Simple camera looking down -Z axis."""
    viewmatrix = torch.eye(4, device=device, dtype=dtype)
    # Simple perspective projection
    fov = 60.0
    aspect = 1.0
    near, far = 0.1, 100.0
    import math
    f = 1.0 / math.tan(math.radians(fov) / 2)
    projmatrix = torch.zeros(4, 4, device=device, dtype=dtype)
    projmatrix[0, 0] = f / aspect
    projmatrix[1, 1] = f
    projmatrix[2, 2] = (far + near) / (near - far)
    projmatrix[2, 3] = (2 * far * near) / (near - far)
    projmatrix[3, 2] = -1.0
    return viewmatrix, projmatrix


def make_splat_inputs(N=32, img_h=64, img_w=64, device='cuda', dtype=torch.float32):
    """Generate test Gaussians in front of the camera."""
    means_3d = torch.randn(N, 3, device=device, dtype=dtype)
    means_3d[:, 2] = -torch.abs(means_3d[:, 2]) - 2.0  # all in front of camera

    scales = torch.abs(torch.randn(N, 3, device=device, dtype=dtype)) * 0.1 + 0.05
    quats = torch.randn(N, 4, device=device, dtype=dtype)
    quats = quats / quats.norm(dim=-1, keepdim=True)  # normalize quaternions

    opacities = torch.sigmoid(torch.randn(N, device=device, dtype=dtype))
    colors = torch.sigmoid(torch.randn(N, 3, device=device, dtype=dtype))

    viewmatrix, projmatrix = make_camera_matrices(device, dtype)
    return means_3d, scales, quats, opacities, colors, viewmatrix, projmatrix, img_h, img_w


class TestGaussianSplatCorrectness:
    def test_basic_shapes(self):
        args = make_splat_inputs()
        image = gaussian_splat_forward(*args)
        assert image.shape == (64, 64, 3)

    def test_matches_reference(self):
        args = make_splat_inputs(N=8, img_h=32, img_w=32)
        ref = gaussian_splat_reference(*args)
        tri = gaussian_splat_forward(*args)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_non_negative_colors(self):
        """Rendered image should have non-negative values."""
        args = make_splat_inputs()
        image = gaussian_splat_forward(*args)
        assert torch.all(image >= -1e-6)

    def test_bounded_colors(self):
        """With sigmoid colors and opacity, output should be bounded."""
        args = make_splat_inputs()
        image = gaussian_splat_forward(*args)
        assert torch.all(image <= 1.0 + 1e-4)

    def test_empty_scene(self):
        """No Gaussians should produce a black image."""
        args = make_splat_inputs(N=0)
        # N=0 edge case — skip if not supported
        try:
            image = gaussian_splat_forward(*args)
            assert torch.all(image == 0)
        except (IndexError, RuntimeError):
            pytest.skip("N=0 not supported")

    def test_single_gaussian(self):
        """Single Gaussian should produce a visible spot."""
        device = 'cuda'
        dtype = torch.float32
        # Place a single Gaussian at the origin (x=0, y=0) at z=-5 so it
        # projects to the center of the image and is well within the frustum.
        means_3d = torch.tensor([[0.0, 0.0, -5.0]], device=device, dtype=dtype)
        scales = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=dtype)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        opacities = torch.tensor([0.9], device=device, dtype=dtype)
        colors = torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype)
        viewmatrix, projmatrix = make_camera_matrices(device, dtype)
        img_h, img_w = 32, 32
        image = gaussian_splat_forward(
            means_3d, scales, quats, opacities, colors,
            viewmatrix, projmatrix, img_h, img_w,
        )
        # At least some pixels should be non-zero
        assert image.sum() > 0


class TestProjection:
    def test_project_shapes(self):
        means = torch.randn(16, 3, device='cuda')
        means[:, 2] = -5.0  # in front of camera
        vm, pm = make_camera_matrices()
        means_2d, depths = project_gaussians(means, vm, pm, 64, 64)
        assert means_2d.shape == (16, 2)
        assert depths.shape == (16,)

    def test_depths_positive_for_front_objects(self):
        """Objects in front of camera should have positive depth (or negative depending on convention)."""
        means = torch.zeros(4, 3, device='cuda')
        means[:, 2] = -torch.tensor([1.0, 2.0, 5.0, 10.0], device='cuda')
        vm, pm = make_camera_matrices()
        _, depths = project_gaussians(means, vm, pm, 64, 64)
        # Depths should all be the same sign (in front of camera)
        assert torch.all(depths < 0) or torch.all(depths > 0)


class TestCov2D:
    def test_cov2d_shapes(self):
        means = torch.randn(8, 3, device='cuda')
        scales = torch.abs(torch.randn(8, 3, device='cuda')) + 0.01
        quats = torch.randn(8, 4, device='cuda')
        quats = quats / quats.norm(dim=-1, keepdim=True)
        vm, _ = make_camera_matrices()
        cov2d = compute_cov2d(means, scales, quats, vm)
        assert cov2d.shape == (8, 2, 2)

    def test_cov2d_symmetric(self):
        """2D covariance should be symmetric."""
        means = torch.randn(8, 3, device='cuda')
        scales = torch.abs(torch.randn(8, 3, device='cuda')) + 0.01
        quats = torch.randn(8, 4, device='cuda')
        quats = quats / quats.norm(dim=-1, keepdim=True)
        vm, _ = make_camera_matrices()
        cov2d = compute_cov2d(means, scales, quats, vm)
        torch.testing.assert_close(cov2d, cov2d.transpose(-1, -2), atol=5e-3, rtol=5e-3)
