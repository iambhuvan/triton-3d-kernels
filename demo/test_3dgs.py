"""Quick test for 3DGS rendering on GPU."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from PIL import Image

# Use the same functions from gradio_app
from gradio_app import create_scene, make_camera_matrices

from kernels.gaussian_splat import gaussian_splat_forward

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create scene
means, scales, quats, opacities, colors = create_scene("Colored Cube", 2000)
viewmatrix, projmatrix = make_camera_matrices(30, 20, 5.0, 45, 256, 256)

# Move to device
means = means.to(device)
scales = scales.to(device)
quats = quats.to(device)
opacities = opacities.to(device)
colors = colors.to(device)
viewmatrix = viewmatrix.to(device)
projmatrix = projmatrix.to(device)

print(f"Means shape: {means.shape}, range: [{means.min():.2f}, {means.max():.2f}]")
print(f"Viewmatrix:\n{viewmatrix}")
print(f"Projmatrix:\n{projmatrix}")

# Compute focal lengths for proper Jacobian
fov_rad = np.radians(45)
focal_x = 256 / (2 * np.tan(fov_rad / 2))
focal_y = focal_x
print(f"Focal lengths: fx={focal_x:.1f}, fy={focal_y:.1f}")

# Render
image = gaussian_splat_forward(
    means, scales, quats, opacities, colors,
    viewmatrix, projmatrix, 256, 256,
    focal_x=focal_x, focal_y=focal_y,
)

print(f"Image shape: {image.shape}")
print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
print(f"Nonzero pixels: {(image.sum(dim=-1) > 0).sum().item()} / {256*256}")

if image.max() > 0:
    img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    pil.save("/tmp/test_3dgs.png")
    print("Saved to /tmp/test_3dgs.png")
    print("SUCCESS: Image has content!")
else:
    print("FAIL: Image is all black!")

    # Debug: check projection
    from reference.gaussian_splat_ref import project_gaussians
    means_2d, depths = project_gaussians(means, viewmatrix, projmatrix, 256, 256)
    print(f"Projected means_2d range: x=[{means_2d[:,0].min():.1f}, {means_2d[:,0].max():.1f}], "
          f"y=[{means_2d[:,1].min():.1f}, {means_2d[:,1].max():.1f}]")
    print(f"Depths range: [{depths.min():.2f}, {depths.max():.2f}]")
    in_frame = ((means_2d[:,0] >= 0) & (means_2d[:,0] < 256) &
                (means_2d[:,1] >= 0) & (means_2d[:,1] < 256)).sum()
    print(f"Points in frame: {in_frame} / {means.shape[0]}")
