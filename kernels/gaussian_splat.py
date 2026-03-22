"""
K6: 3D Gaussian Splatting Forward Rasterizer — Triton Kernel

Architecture:
  Tile-based rasterization parallelized over screen tiles.

  Preprocessing (Python):
    1. Project 3D Gaussians to 2D screen space
    2. Compute 2D covariance matrices
    3. Sort by depth (front-to-back)
    4. Assign Gaussians to overlapping screen tiles
    5. Pack per-tile Gaussian lists into contiguous arrays

  Triton kernel — one program per tile:
    Grid: (tiles_x * tiles_y,)
    Each program:
      1. Determine tile pixel bounds
      2. Load list of Gaussians assigned to this tile
      3. For each pixel in tile:
         - T = 1.0, C = (0, 0, 0)
         - For each Gaussian (front-to-back):
           a. delta = pixel - mean_2d
           b. mahal = delta^T @ cov2d_inv @ delta
           c. alpha = opacity * exp(-0.5 * mahal), clamp to [0, 0.99]
           d. C += T * alpha * color
           e. T *= (1 - alpha)
           f. if T < 0.001: break
      4. Store pixel colors

  Why tile-based: adjacent pixels in a tile access the same Gaussians,
  enabling coalesced loads and shared computation.
"""

import torch
import triton
import triton.language as tl
from reference.gaussian_splat_ref import project_gaussians, compute_cov2d


# ============================================================
# Preprocessing
# ============================================================

def preprocess_gaussians(
    means_3d, scales, quats, opacities, colors,
    viewmatrix, projmatrix, img_h, img_w, tile_size=16,
    focal_x=None, focal_y=None,
):
    """Project, sort, and assign Gaussians to tiles.

    Returns all data structures needed by the Triton kernel.
    """
    N = means_3d.shape[0]
    device = means_3d.device
    dtype = means_3d.dtype

    if N == 0:
        return None  # empty scene

    # 1. Project to 2D
    means_2d, depths = project_gaussians(means_3d, viewmatrix, projmatrix, img_h, img_w)

    # 2. Compute 2D covariance (with Jacobian if focal lengths provided)
    cov2d = compute_cov2d(means_3d, scales, quats, viewmatrix,
                          focal_x=focal_x, focal_y=focal_y)

    # 3. Sort by depth (front-to-back)
    sorted_idx = torch.argsort(depths)
    means_2d = means_2d[sorted_idx]
    cov2d = cov2d[sorted_idx]
    opacities = opacities[sorted_idx]
    colors = colors[sorted_idx]

    # Precompute inverse covariance
    eye = 1e-4 * torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
    cov2d_inv = torch.linalg.inv(cov2d + eye)

    # 4. Compute bounding radius per Gaussian from eigenvalues
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    trace = a + c
    det = a * c - b * b
    disc = torch.sqrt(torch.clamp((trace / 2)**2 - det, min=0))
    max_eval = trace / 2 + disc
    radius = 3.0 * torch.sqrt(max_eval + 1e-8)

    # 5. Assign to tiles
    tiles_x = (img_w + tile_size - 1) // tile_size
    tiles_y = (img_h + tile_size - 1) // tile_size
    n_tiles = tiles_x * tiles_y

    # Build per-tile lists
    tile_gaussian_lists = [[] for _ in range(n_tiles)]
    cx, cy = means_2d[:, 0], means_2d[:, 1]

    for i in range(N):
        x_min = max(0, int((cx[i] - radius[i]) / tile_size))
        x_max = min(tiles_x - 1, int((cx[i] + radius[i]) / tile_size))
        y_min = max(0, int((cy[i] - radius[i]) / tile_size))
        y_max = min(tiles_y - 1, int((cy[i] + radius[i]) / tile_size))
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                tile_gaussian_lists[ty * tiles_x + tx].append(i)

    # 6. Pack into contiguous arrays
    max_per_tile = max((len(lst) for lst in tile_gaussian_lists), default=0)
    max_per_tile = max(max_per_tile, 1)

    tile_gauss_idx = torch.full((n_tiles, max_per_tile), -1, dtype=torch.int32, device=device)
    tile_gauss_count = torch.zeros(n_tiles, dtype=torch.int32, device=device)

    for t in range(n_tiles):
        count = len(tile_gaussian_lists[t])
        tile_gauss_count[t] = count
        if count > 0:
            tile_gauss_idx[t, :count] = torch.tensor(tile_gaussian_lists[t], dtype=torch.int32)

    return {
        'means_2d': means_2d.contiguous(),
        'cov2d_inv': cov2d_inv.contiguous(),
        'opacities': opacities.contiguous(),
        'colors': colors.contiguous(),
        'tile_gauss_idx': tile_gauss_idx.contiguous(),
        'tile_gauss_count': tile_gauss_count.contiguous(),
        'tiles_x': tiles_x,
        'tiles_y': tiles_y,
        'max_per_tile': max_per_tile,
    }


# ============================================================
# Triton Kernel
# ============================================================

@triton.jit
def _gaussian_splat_kernel(
    # Per-Gaussian data (sorted by depth)
    MEANS2D_ptr,    # (N, 2) float
    COV2D_INV_ptr,  # (N, 2, 2) float — stored as (N, 4) flattened
    OPACITIES_ptr,  # (N,) float
    COLORS_ptr,     # (N, 3) float
    # Per-tile data
    TILE_IDX_ptr,   # (n_tiles, max_per_tile) int32 — Gaussian indices per tile
    TILE_COUNT_ptr, # (n_tiles,) int32 — count per tile
    # Output
    IMAGE_ptr,      # (H, W, 3) float
    # Dims
    IMG_H, IMG_W,
    tiles_x,
    max_per_tile,
    # Tile config
    TILE_SIZE: tl.constexpr,
    PIXELS_PER_TILE: tl.constexpr,  # = TILE_SIZE * TILE_SIZE
):
    """Rasterize one screen tile.

    Grid: (n_tiles,)
    Each program processes TILE_SIZE x TILE_SIZE pixels.
    """
    tile_id = tl.program_id(0)

    # Tile coordinates
    tile_y = tile_id // tiles_x
    tile_x = tile_id % tiles_x

    # Pixel bounds for this tile
    px_start = tile_x * TILE_SIZE
    py_start = tile_y * TILE_SIZE

    # Number of Gaussians for this tile
    n_gauss = tl.load(TILE_COUNT_ptr + tile_id)

    # Process each pixel in the tile
    # Flatten pixel loop: iterate over TILE_SIZE * TILE_SIZE pixels
    for local_idx in range(0, PIXELS_PER_TILE):
        local_y = local_idx // TILE_SIZE
        local_x = local_idx % TILE_SIZE

        px = px_start + local_x
        py = py_start + local_y

        # Bounds check — wrap entire pixel body in condition (no continue/break in Triton)
        if px < IMG_W and py < IMG_H:
            # Initialize alpha blending state
            T = 1.0  # transmittance
            r_acc = 0.0
            g_acc = 0.0
            b_acc = 0.0

            # Alpha blend Gaussians front-to-back
            # Use a "done" flag instead of break
            done = False
            for g_idx in range(0, max_per_tile):
                if not done:
                    if g_idx >= n_gauss:
                        done = True
                    if not done and T < 0.001:
                        done = True

                    if not done:
                        # Load Gaussian index for this tile
                        gauss_id = tl.load(TILE_IDX_ptr + tile_id * max_per_tile + g_idx)
                        if gauss_id < 0:
                            done = True

                        if not done:
                            # Load Gaussian 2D mean
                            mean_x = tl.load(MEANS2D_ptr + gauss_id * 2 + 0)
                            mean_y = tl.load(MEANS2D_ptr + gauss_id * 2 + 1)

                            # Delta from pixel to Gaussian center
                            dx = px + 0.5 - mean_x  # pixel center
                            dy = py + 0.5 - mean_y

                            # Load inverse covariance (stored as 4 floats: a, b, c, d for [[a,b],[c,d]])
                            inv_a = tl.load(COV2D_INV_ptr + gauss_id * 4 + 0)
                            inv_b = tl.load(COV2D_INV_ptr + gauss_id * 4 + 1)
                            inv_c = tl.load(COV2D_INV_ptr + gauss_id * 4 + 2)
                            inv_d = tl.load(COV2D_INV_ptr + gauss_id * 4 + 3)

                            # Mahalanobis distance: delta^T @ Σ^{-1} @ delta
                            mahal = dx * (inv_a * dx + inv_b * dy) + dy * (inv_c * dx + inv_d * dy)

                            # Gaussian weight * opacity
                            opacity = tl.load(OPACITIES_ptr + gauss_id)
                            alpha = opacity * tl.exp(-0.5 * mahal)
                            # Clamp alpha
                            alpha = tl.minimum(alpha, 0.99)
                            # Skip if alpha too small (instead of continue, use conditional)
                            if alpha >= 1.0 / 255.0:
                                # Accumulate color
                                weight = T * alpha
                                cr = tl.load(COLORS_ptr + gauss_id * 3 + 0)
                                cg = tl.load(COLORS_ptr + gauss_id * 3 + 1)
                                cb = tl.load(COLORS_ptr + gauss_id * 3 + 2)

                                r_acc += weight * cr
                                g_acc += weight * cg
                                b_acc += weight * cb

                                # Update transmittance
                                T *= (1.0 - alpha)

            # Store pixel color
            pixel_offset = (py * IMG_W + px) * 3
            tl.store(IMAGE_ptr + pixel_offset + 0, r_acc)
            tl.store(IMAGE_ptr + pixel_offset + 1, g_acc)
            tl.store(IMAGE_ptr + pixel_offset + 2, b_acc)


# ============================================================
# Python wrapper
# ============================================================

def gaussian_splat_forward(
    means_3d, scales, quats, opacities, colors,
    viewmatrix, projmatrix, img_h, img_w,
    focal_x=None, focal_y=None,
) -> torch.Tensor:
    """Render image via 3D Gaussian Splatting.

    Args:
        means_3d:   (N, 3) Gaussian centers
        scales:     (N, 3) scales
        quats:      (N, 4) rotation quaternions
        opacities:  (N,) opacities
        colors:     (N, 3) RGB colors
        viewmatrix: (4, 4) world-to-camera
        projmatrix: (4, 4) projection matrix
        img_h, img_w: output image size
        focal_x:    focal length in pixels (x). If None, estimated from projmatrix.
        focal_y:    focal length in pixels (y). If None, estimated from projmatrix.

    Returns:
        image: (H, W, 3) rendered RGB image
    """
    device = means_3d.device
    dtype = means_3d.dtype
    TILE_SIZE = 16

    image = torch.zeros(img_h, img_w, 3, device=device, dtype=dtype)

    if means_3d.shape[0] == 0:
        return image

    # Preprocess: project, sort, assign to tiles
    data = preprocess_gaussians(
        means_3d, scales, quats, opacities, colors,
        viewmatrix, projmatrix, img_h, img_w, tile_size=TILE_SIZE,
        focal_x=focal_x, focal_y=focal_y,
    )

    if data is None:
        return image

    # Flatten cov2d_inv from (N, 2, 2) to (N, 4) for kernel
    cov2d_inv_flat = data['cov2d_inv'].reshape(-1, 4).contiguous()

    n_tiles = data['tiles_x'] * data['tiles_y']

    _gaussian_splat_kernel[(n_tiles,)](
        data['means_2d'], cov2d_inv_flat,
        data['opacities'], data['colors'],
        data['tile_gauss_idx'], data['tile_gauss_count'],
        image,
        img_h, img_w,
        data['tiles_x'],
        data['max_per_tile'],
        TILE_SIZE=TILE_SIZE,
        PIXELS_PER_TILE=TILE_SIZE * TILE_SIZE,
    )

    return image
