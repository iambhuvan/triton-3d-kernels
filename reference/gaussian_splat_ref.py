"""
K6: 3D Gaussian Splatting Forward Rasterizer — Reference Implementation

3D Gaussian Splatting (3DGS) represents a 3D scene as a set of 3D Gaussians,
each with a position (mean), covariance, opacity, and color (spherical harmonics).
Rendering works by "splatting" each Gaussian onto the image plane.

The forward rasterization pipeline:
  1. PROJECT: Transform 3D Gaussian means to 2D screen space
  2. COVARIANCE: Project 3D covariance to 2D covariance on image plane
  3. SORT: Sort Gaussians front-to-back by depth (for correct alpha blending)
  4. RASTERIZE: For each pixel, blend all overlapping Gaussians front-to-back

Math for each step:

Step 1 — Projection:
  mean_2d = project(mean_3d, viewmatrix, projmatrix)
  depth = (viewmatrix @ mean_3d).z

Step 2 — 2D Covariance from 3D:
  Given 3D covariance Σ (3×3 symmetric positive definite):
    Σ_2d = J @ W @ Σ @ W^T @ J^T
  where W is the view rotation (3×3) and J is the Jacobian of the
  projection (2×3) — essentially how 3D perturbations map to 2D.

  In practice, covariance is stored as scale (3,) + rotation quaternion (4,)
  and reconstructed as: Σ = R @ S @ S^T @ R^T

Step 3 — Sorting:
  Sort all Gaussians by depth. This ensures correct front-to-back
  alpha compositing.

Step 4 — Alpha blending (front-to-back):
  For each pixel p:
    C_p = 0, T = 1  (accumulated color, transmittance)
    For each Gaussian g in depth order:
      alpha_g = opacity_g * exp(-0.5 * (p - mean_2d_g)^T @ Σ_2d_g^{-1} @ (p - mean_2d_g))
      C_p += T * alpha_g * color_g
      T *= (1 - alpha_g)
      if T < 0.001: break  (early termination)

Why this kernel matters:
  The rasterization loop (step 4) is embarrassingly parallel over pixels
  but requires careful memory management — each pixel may overlap with
  hundreds of Gaussians. The current CUDA implementation is a black box.
  A Triton version would be more readable, hackable, and potentially
  competitive in performance with careful tiling over screen-space tiles.
"""

import torch
import torch.nn.functional as F


def project_gaussians(
    means_3d: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    img_h: int,
    img_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D Gaussian centers to 2D screen coordinates.

    Args:
        means_3d:   (N, 3) 3D positions
        viewmatrix: (4, 4) world-to-camera transform
        projmatrix: (4, 4) full projection matrix (view @ projection)
        img_h, img_w: image dimensions

    Returns:
        means_2d: (N, 2) pixel coordinates
        depths:   (N,) z-depth in camera space
    """
    N = means_3d.shape[0]

    # Homogeneous coordinates
    ones = torch.ones(N, 1, device=means_3d.device, dtype=means_3d.dtype)
    pts_h = torch.cat([means_3d, ones], dim=-1)  # (N, 4)

    # Camera space
    pts_cam = (viewmatrix @ pts_h.T).T  # (N, 4)
    depths = pts_cam[:, 2]  # z-depth

    # Clip space
    pts_clip = (projmatrix @ pts_h.T).T  # (N, 4)

    # NDC: perspective divide
    pts_ndc = pts_clip[:, :2] / (pts_clip[:, 3:4] + 1e-8)

    # Screen space: NDC [-1,1] -> pixel [0, W/H]
    means_2d = torch.zeros(N, 2, device=means_3d.device, dtype=means_3d.dtype)
    means_2d[:, 0] = (pts_ndc[:, 0] + 1.0) * 0.5 * img_w
    means_2d[:, 1] = (pts_ndc[:, 1] + 1.0) * 0.5 * img_h

    return means_2d, depths


def compute_cov2d(
    means_3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    viewmatrix: torch.Tensor,
    focal_x: float = None,
    focal_y: float = None,
) -> torch.Tensor:
    """Compute 2D covariance from 3D Gaussian parameters.

    Uses the standard 3DGS formula: Σ_2d = J @ W @ Σ_3d @ W^T @ J^T
    where J is the Jacobian of perspective projection (includes focal/depth scaling).

    Args:
        means_3d:   (N, 3) positions
        scales:     (N, 3) scale factors
        quats:      (N, 4) rotation quaternions (w, x, y, z)
        viewmatrix: (4, 4) world-to-camera
        focal_x:    focal length in pixels (x-axis). If None, skip Jacobian (legacy).
        focal_y:    focal length in pixels (y-axis). If None, skip Jacobian (legacy).

    Returns:
        cov2d: (N, 2, 2) 2D covariance matrices in pixel coordinates
    """
    N = means_3d.shape[0]
    device = means_3d.device
    dtype = means_3d.dtype

    # Build rotation matrices from quaternions
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    # Build 3D covariance: Σ = R @ diag(s^2) @ R^T
    S_mat = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    S_mat[:, 0, 0] = scales[:, 0] ** 2
    S_mat[:, 1, 1] = scales[:, 1] ** 2
    S_mat[:, 2, 2] = scales[:, 2] ** 2
    cov3d = R @ S_mat @ R.transpose(-1, -2)  # (N, 3, 3)

    # View rotation (upper-left 3x3 of viewmatrix)
    W = viewmatrix[:3, :3]  # (3, 3)

    # Project 3D covariance to camera space
    cov_cam = W @ cov3d @ W.T  # (N, 3, 3) via broadcasting

    if focal_x is not None and focal_y is not None:
        # Apply Jacobian of perspective projection (standard 3DGS formula)
        # J = [[fx/tz, 0, -fx*tx/tz²],
        #      [0, fy/tz, -fy*ty/tz²]]
        #
        # Σ_2d = J @ Σ_cam @ J^T (in pixel coordinates)

        # Get camera-space positions for per-Gaussian depth
        ones = torch.ones(N, 1, device=device, dtype=dtype)
        pts_h = torch.cat([means_3d, ones], dim=-1)
        pts_cam = (viewmatrix @ pts_h.T).T  # (N, 4)
        tx = pts_cam[:, 0]
        ty = pts_cam[:, 1]
        # Use absolute depth (OpenGL convention has negative z in front of camera)
        tz = pts_cam[:, 2].abs().clamp(min=0.1)

        # Build per-Gaussian Jacobian (N, 2, 3)
        J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
        J[:, 0, 0] = focal_x / tz
        J[:, 0, 2] = -focal_x * tx / (tz * tz)
        J[:, 1, 1] = focal_y / tz
        J[:, 1, 2] = -focal_y * ty / (tz * tz)

        # Σ_2d = J @ Σ_cam @ J^T  (N, 2, 2)
        cov2d = J @ cov_cam @ J.transpose(-1, -2)
    else:
        # Legacy: no Jacobian (camera-space covariance, NOT pixel-space)
        cov2d = cov_cam[:, :2, :2]

    return cov2d


def gaussian_splat_reference(
    means_3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    img_h: int,
    img_w: int,
    focal_x: float = None,
    focal_y: float = None,
) -> torch.Tensor:
    """Render image by splatting 3D Gaussians.

    Args:
        means_3d:   (N, 3) Gaussian centers
        scales:     (N, 3) Gaussian scales
        quats:      (N, 4) Gaussian rotations (wxyz)
        opacities:  (N,) per-Gaussian opacity [0, 1]
        colors:     (N, 3) per-Gaussian RGB color
        viewmatrix: (4, 4) world-to-camera
        projmatrix: (4, 4) full projection matrix
        img_h, img_w: output image size
        focal_x:    focal length in pixels (x). If None, skip Jacobian.
        focal_y:    focal length in pixels (y). If None, skip Jacobian.

    Returns:
        image: (img_h, img_w, 3) rendered RGB image
    """
    N = means_3d.shape[0]
    device = means_3d.device
    dtype = means_3d.dtype

    # Step 1: Project to 2D
    means_2d, depths = project_gaussians(means_3d, viewmatrix, projmatrix, img_h, img_w)

    # Step 2: Compute 2D covariance
    cov2d = compute_cov2d(means_3d, scales, quats, viewmatrix,
                          focal_x=focal_x, focal_y=focal_y)  # (N, 2, 2)

    # Step 3: Sort by depth (front-to-back)
    sorted_indices = torch.argsort(depths)

    means_2d = means_2d[sorted_indices]
    cov2d = cov2d[sorted_indices]
    opacities = opacities[sorted_indices]
    colors = colors[sorted_indices]

    # Precompute inverse covariance
    cov2d_inv = torch.linalg.inv(cov2d + 1e-6 * torch.eye(2, device=device, dtype=dtype))

    # Step 4: Rasterize — for each pixel, blend Gaussians front-to-back
    image = torch.zeros(img_h, img_w, 3, device=device, dtype=dtype)

    # Create pixel grid (use pixel centers: +0.5)
    py, px = torch.meshgrid(
        torch.arange(img_h, device=device, dtype=dtype) + 0.5,
        torch.arange(img_w, device=device, dtype=dtype) + 0.5,
        indexing='ij'
    )
    pixels = torch.stack([px, py], dim=-1).reshape(-1, 2)  # (H*W, 2)

    # For each Gaussian, compute contribution to all pixels
    transmittance = torch.ones(img_h * img_w, device=device, dtype=dtype)
    accum_color = torch.zeros(img_h * img_w, 3, device=device, dtype=dtype)

    for i in range(N):
        if transmittance.max() < 0.001:
            break

        # Distance from this Gaussian's center to all pixels
        delta = pixels - means_2d[i:i+1]  # (H*W, 2)

        # Mahalanobis distance: delta^T @ Σ^{-1} @ delta
        mahal = (delta @ cov2d_inv[i]) * delta  # (H*W, 2)
        mahal = mahal.sum(dim=-1)  # (H*W,)

        # Gaussian weight * opacity
        alpha = opacities[i] * torch.exp(-0.5 * mahal)
        alpha = alpha.clamp(max=0.99)

        # Accumulate: C += T * alpha * color
        weight = transmittance * alpha  # (H*W,)
        accum_color += weight.unsqueeze(-1) * colors[i:i+1]  # (H*W, 3)

        # Update transmittance
        transmittance *= (1 - alpha)

    image = accum_color.reshape(img_h, img_w, 3)
    return image
