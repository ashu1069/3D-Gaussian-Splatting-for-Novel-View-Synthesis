"""
3D Gaussian Splatting rendering implementation.

3D Gaussian Splatting renders a scene by projecting 3D Gaussians onto the 2D image plane
and compositing them using alpha blending. The rendering pipeline consists of:

1. Projection: Transform 3D Gaussians to camera space and project to 2D
2. Covariance Projection: Project 3D covariance matrices to 2D image space
3. Culling: Remove Gaussians outside view frustum or too small to see
4. Depth Sorting: Sort Gaussians by depth for correct alpha blending
5. Tiling: Divide image into tiles for efficient parallel processing
6. Rasterization: Evaluate Gaussian contributions per pixel
7. Alpha Blending: Composite Gaussians front-to-back

Mathematical Foundation:
-----------------------
A 3D Gaussian is defined by:
    G(x) = exp(-0.5 * (x-μ)^T Σ^(-1) (x-μ))

where μ is the mean (position) and Σ is the covariance matrix.

When projected to 2D image space, the Gaussian becomes:
    G_2D(u,v) = exp(-0.5 * [u-μ_u, v-μ_v] Σ_2D^(-1) [u-μ_u, v-μ_v]^T)

The 2D covariance Σ_2D is computed via:
    Σ_2D = J @ Σ_3D_camera @ J^T

where J is the Jacobian of the projection function (pinhole camera model).

Alpha Blending:
--------------
Gaussians are composited using front-to-back alpha blending:
    C_final = Σ_i α_i * c_i * Π_{j<i} (1 - α_j)

where:
- α_i is the opacity of Gaussian i (Gaussian value * base opacity)
- c_i is the color of Gaussian i
- The product term ensures Gaussians behind others are occluded

This is equivalent to volume rendering with discrete samples.
"""

import torch
import numpy as np

try:
    from .utils import project_points, inv2x2, transform_to_camera_space, check_frustum_camera_space
except ImportError:
    from utils import project_points, inv2x2, transform_to_camera_space, check_frustum_camera_space

# Try to compile render function for speed (PyTorch 2.0+)
# This can provide 2-5x speedup, but still won't match custom CUDA kernels
_USE_COMPILE = hasattr(torch, 'compile')
if _USE_COMPILE:
    try:
        # Compile render function with optimizations
        _render_compiled = None
    except:
        _USE_COMPILE = False


def render(pos, color, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy,
           near=0.01, far=100.0, pix_guard=32, T=16, min_conis=1e-6,
           chi_square_clip=6.25, alpha_max=0.99, alpha_cutoff=1/128.):
    """
    Render a scene using 3D Gaussian Splatting.
    
    Rendering Pipeline:
    -------------------
    1. Project 3D points to 2D image coordinates
    2. Cull Gaussians outside view frustum (with guard band)
    3. Project 3D covariance matrices to 2D using Jacobian
    4. Sort Gaussians by depth (front-to-back for alpha blending)
    5. Tile-based rasterization: Divide image into T×T tiles
    6. For each tile, evaluate Gaussian contributions to pixels
    7. Alpha blend Gaussians to produce final image
    
    Key Optimizations:
    ------------------
    - Tiling: Only evaluate Gaussians that intersect each tile
    - Chi-square clipping: Ignore Gaussians beyond 3σ (99.7% of mass)
    - Alpha cutoff: Skip Gaussians with opacity < 1/255 (invisible)
    - Early termination: Stop blending when accumulated opacity ≈ 1
    
    Args:
        pos: 3D positions of Gaussians. Shape: [N, 3]
        color: RGB colors per Gaussian (from SH evaluation). Shape: [N, 3]
        opacity_raw: Raw opacity values (will be sigmoided). Shape: [N]
        sigma: 3D covariance matrices. Shape: [N, 3, 3]
        c2w: Camera-to-world transform matrix. Shape: [4, 4]
        H, W: Image height and width
        fx, fy, cx, cy: Camera intrinsics
        near, far: Near and far clipping planes (depth bounds)
        pix_guard: Pixel guard band for conservative culling (default: 32, reduced from 64 for speed)
        T: Tile size (typically 16×16 pixels)
        min_conis: Minimum value for inverse covariance diagonal (numerical stability)
        chi_square_clip: Chi-square clipping threshold (default 6.25 ≈ 2.5σ for 2D, reduced from 9.21 for speed)
        alpha_max: Maximum alpha value (prevents over-saturation)
        alpha_cutoff: Alpha cutoff threshold (default: 1/128, increased from 1/255 for speed)
        
    Returns:
        Rendered image. Shape: [H, W, 3], values in [0,1]
    """
    # Step 1: Pre-filter low-opacity Gaussians (optimization)
    # Filter out Gaussians with very low opacity before expensive operations
    opacity_pre = torch.sigmoid(opacity_raw).clamp(0, 0.999)
    opacity_mask = opacity_pre >= alpha_cutoff * 0.5  # Pre-filter threshold (half of final cutoff)
    
    if not opacity_mask.any():
        # Return zeros but preserve gradients
        dummy = (color.sum() * 0.0).expand(H * W * 3).reshape(H, W, 3)
        return dummy
    
    pos = pos[opacity_mask]
    color = color[opacity_mask]
    opacity_raw = opacity_raw[opacity_mask]
    sigma = sigma[opacity_mask]
    
    # Step 2: Transform to camera space and check frustum BEFORE projection
    # This optimization avoids expensive projection for Gaussians outside the view frustum
    # We transform to camera space first, then check bounds, then only project visible ones
    x, y, z = transform_to_camera_space(pos, c2w)
    
    # Step 3: Cull Gaussians outside view frustum (in camera space, before projection)
    # pix_guard adds a safety margin to account for Gaussian extent (they're not point samples)
    # We keep Gaussians that might contribute to visible pixels
    in_guard = check_frustum_camera_space(x, y, z, fx, fy, cx, cy, H, W, near, far, pix_guard)
    
    # Filter all arrays before projection
    pos = pos[in_guard]
    color = color[in_guard]
    opacity_raw = opacity_raw[in_guard]
    sigma = sigma[in_guard]
    x = x[in_guard]
    y = y[in_guard]
    z = z[in_guard]
    
    # Early return if no Gaussians are visible
    if pos.shape[0] == 0:
        # Return zeros but ensure it's connected to the computation graph
        dummy = (color.sum() * 0.0).expand(H * W * 3).reshape(H, W, 3)
        return dummy
    
    # Step 4: Project remaining Gaussians to 2D image coordinates
    # Now we only project the Gaussians that passed frustum culling
    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)

    opacity = torch.sigmoid(opacity_raw).clamp(0, 0.999)
    idx = torch.nonzero(in_guard, as_tuple=False).squeeze(1)

    # Step 5: Project 3D covariance matrices to 2D image space
    # This is the key mathematical operation: Σ_2D = J @ Σ_3D_camera @ J^T
    # where J is the Jacobian of the projection function
    
    # First, transform covariance from world space to camera space
    Rcw = c2w[:3, :3]  # Camera-to-world rotation
    Rwc = Rcw.t()      # World-to-camera rotation (transpose = inverse for rotation)
    # Transform: Σ_camera = R_wc @ Σ_world @ R_wc^T
    tmp = Rwc.unsqueeze(0) @ sigma @ Rwc.t().unsqueeze(0)  # Eq. 5 from paper
    
    # Build Jacobian J of projection function [u,v] = project([X,Y,Z])
    # For pinhole: u = fx * X/Z + cx, v = fy * Y/Z + cy
    # Jacobian J = [[∂u/∂X, ∂u/∂Y, ∂u/∂Z],
    #               [∂v/∂X, ∂v/∂Y, ∂v/∂Z]]
    invz = 1 / z.clamp_min(1e-6)  # 1/Z (avoid division by zero)
    invz2 = invz * invz            # 1/Z²
    J = torch.zeros((pos.shape[0], 2, 3), device=pos.device, dtype=pos.dtype)
    J[:, 0, 0] = fx * invz         # ∂u/∂X = fx/Z
    J[:, 1, 1] = fy * invz         # ∂v/∂Y = fy/Z
    J[:, 0, 2] = -fx * x * invz2   # ∂u/∂Z = -fx*X/Z²
    J[:, 1, 2] = -fy * y * invz2   # ∂v/∂Z = -fy*Y/Z²
    
    # Project covariance: Σ_2D = J @ Σ_3D_camera @ J^T
    sigma_camera = J @ tmp @ J.transpose(1, 2)
    sigma_camera = 0.5 * (sigma_camera + sigma_camera.transpose(1, 2))  # Enforce symmetry
    # Ensure positive definiteness
    evals, evecs = torch.linalg.eigh(sigma_camera)
    evals = torch.clamp(evals, min=1e-6, max=1e4)
    sigma_camera = evecs @ torch.diag_embed(evals) @ evecs.transpose(1, 2)

    # Check for finite values in covariance matrices
    if sigma_camera.shape[0] == 0:
        # Return zeros but preserve gradients
        dummy = (color.sum() * 0.0).expand(H * W * 3).reshape(H, W, 3)
        return dummy
    
    keep = torch.isfinite(
        sigma_camera.reshape(sigma_camera.shape[0], -1)).all(dim=-1)
    
    if not keep.any():
        # Return zeros but preserve gradients
        dummy = (color.sum() * 0.0).expand(H * W * 3).reshape(H, W, 3)
        return dummy
    
    uv = uv[keep]
    color = color[keep]
    opacity = opacity[keep]
    z = z[keep]
    sigma_camera = sigma_camera[keep]
    idx = idx[keep]
    evals = evals[keep]

    # Step 6: Sort Gaussians by depth (front-to-back)
    # Alpha blending requires front-to-back order for correct occlusion
    # We sort by z (depth in camera space), smallest z = closest to camera
    if z.shape[0] == 0:
        # Return zeros but preserve gradients
        dummy = (color.sum() * 0.0).expand(H * W * 3).reshape(H, W, 3)
        return dummy
    
    order = torch.argsort(z, descending=False)
    uv = uv[order]
    u = uv[:, 0]
    v = uv[:, 1]
    color = color[order]
    opacity = opacity[order]
    sigma_camera = sigma_camera[order]
    evals = evals[order]
    idx = idx[order]

    # Step 7: Tiling - Compute which tiles each Gaussian intersects
    # Tiling enables parallel processing: each tile can be rendered independently
    # We compute a bounding box (AABB) for each Gaussian based on its 2D extent
    
    # Use the larger eigenvalue (major variance) to estimate Gaussian radius
    # Use 2.5σ instead of 3σ for tighter bounds (faster, matches chi_square_clip=6.25)
    major_variance = evals[:, 1].clamp_min(1e-12).clamp_max(1e4)  # [N]
    radius = torch.ceil(2.5 * torch.sqrt(major_variance)).to(torch.int64)  # 2.5σ radius in pixels (matches chi_square_clip)
    umin = torch.floor(u - radius).to(torch.int64)
    umax = torch.floor(u + radius).to(torch.int64)
    vmin = torch.floor(v - radius).to(torch.int64)
    vmax = torch.floor(v + radius).to(torch.int64)

    on_screen = (umax >= 0) & (umin < W) & (vmax >= 0) & (vmin < H)
    if not on_screen.any():
        raise Exception("All projected points are off-screen")
    u, v = u[on_screen], v[on_screen]
    color = color[on_screen]
    opacity = opacity[on_screen]
    sigma_camera = sigma_camera[on_screen]
    umin, umax = umin[on_screen], umax[on_screen]
    vmin, vmax = vmin[on_screen], vmax[on_screen]
    idx = idx[on_screen]
    umin = umin.clamp(0, W - 1)
    umax = umax.clamp(0, W - 1)
    vmin = vmin.clamp(0, H - 1)
    vmax = vmax.clamp(0, H - 1)

    # Step 8: Compute tile indices for each Gaussian's bounding box
    # Divide pixel coordinates by tile size to get tile coordinates
    umin_tile = (umin // T).to(torch.int64)  # [N] - leftmost tile
    umax_tile = (umax // T).to(torch.int64)  # [N] - rightmost tile
    vmin_tile = (vmin // T).to(torch.int64)  # [N] - topmost tile
    vmax_tile = (vmax // T).to(torch.int64)  # [N] - bottommost tile

    # Count how many tiles each Gaussian spans (for creating tile-Gaussian associations)
    n_u = umax_tile - umin_tile + 1  # [N] - tiles in u direction
    n_v = vmax_tile - vmin_tile + 1  # [N] - tiles in v direction

    # Max number of tiles
    max_u = int(n_u.max().item())
    max_v = int(n_v.max().item())

    nb_gaussians = umin_tile.shape[0]
    span_indices_u = torch.arange(max_u, device=pos.device, dtype=torch.int64)  # [max_u]
    span_indices_v = torch.arange(max_v, device=pos.device, dtype=torch.int64)  # [max_v]
    tile_u = (umin_tile[:, None, None] + span_indices_u[None, :, None]
              ).expand(nb_gaussians, max_u, max_v)  # [N, max_u, max_v]
    tile_v = (vmin_tile[:, None, None] + span_indices_v[None, None, :]
              ).expand(nb_gaussians, max_u, max_v)  # [N, max_u, max_v]
    mask = (span_indices_u[None, :, None] < n_u[:, None, None]
            ) & (span_indices_v[None, None, :] < n_v[:, None, None])  # [N, max_u, max_v]
    flat_tile_u = tile_u[mask]  # [0, 0, 1, 1, 2, ...]
    flat_tile_v = tile_v[mask]  # [0, 1, 0, 1, 2]

    nb_tiles_per_gaussian = n_u * n_v  # [N]
    gaussian_ids = torch.repeat_interleave(
        torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64),
        nb_tiles_per_gaussian)  # [0, 0, 0, 0, 1 ...]
    nb_tiles_u = (W + T - 1) // T
    flat_tile_id = flat_tile_v * nb_tiles_u + flat_tile_u  # [0, 0, 0, 0, 1 ...]

    # Step 9: Sort Gaussians by (tile_id, depth) for efficient tile-based rendering
    # This creates a list where Gaussians are grouped by tile, and within each tile,
    # they're sorted front-to-back for alpha blending
    
    # Create composite key: tile_id * M + depth_order
    # This ensures Gaussians are sorted first by tile, then by depth within tile
    idx_z_order = torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64)
    M = nb_gaussians + 1  # Large enough multiplier to separate tile groups
    comp = flat_tile_id * M + idx_z_order[gaussian_ids]
    comp_sorted, perm = torch.sort(comp)
    gaussian_ids = gaussian_ids[perm]
    tile_ids_1d = torch.div(comp_sorted, M, rounding_mode='floor')  # Extract tile_id from composite key

    # tile_ids_1d [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # nb_gaussian_per_tile [3, 2, 4]
    # start [0, 3, 5]
    # end [3, 5, 9]
    unique_tile_ids, nb_gaussian_per_tile = torch.unique_consecutive(tile_ids_1d, return_counts=True)
    start = torch.zeros_like(unique_tile_ids)
    start[1:] = torch.cumsum(nb_gaussian_per_tile[:-1], dim=0)
    end = start + nb_gaussian_per_tile

    # Step 10: Compute inverse covariance matrices for Gaussian evaluation
    # We need Σ^(-1) to evaluate: G(u,v) = exp(-0.5 * (p-μ)^T Σ^(-1) (p-μ))
    inverse_covariance = inv2x2(sigma_camera)
    # Clamp diagonal to prevent numerical issues (very small eigenvalues → very large inverse)
    # Use out-of-place operations to preserve gradients
    diag_00 = torch.clamp(inverse_covariance[:, 0, 0], min=min_conis)
    diag_11 = torch.clamp(inverse_covariance[:, 1, 1], min=min_conis)
    # Create new tensor instead of modifying in-place
    inverse_covariance = inverse_covariance.clone()
    inverse_covariance[:, 0, 0] = diag_00
    inverse_covariance[:, 1, 1] = diag_11

    # Initialize final image with zeros (will accumulate contributions)
    final_image = torch.zeros((H * W, 3), device=pos.device, dtype=pos.dtype)
    
    # Collect all tile contributions
    tile_contributions = []
    tile_indices = []
    
    # Iterate over tiles
    for tile_id, s0, s1 in zip(unique_tile_ids.tolist(), start.tolist(), end.tolist()):

        current_gaussian_ids = gaussian_ids[s0:s1]

        txi = tile_id % nb_tiles_u
        tyi = tile_id // nb_tiles_u
        x0, y0 = txi * T, tyi * T
        x1, y1 = min((txi + 1) * T, W), min((tyi + 1) * T, H)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = torch.arange(x0, x1, device=pos.device, dtype=pos.dtype)
        ys = torch.arange(y0, y1, device=pos.device, dtype=pos.dtype)
        pu, pv = torch.meshgrid(xs, ys, indexing='xy')
        px_u = pu.reshape(-1)  # [T * T]
        px_v = pv.reshape(-1)
        pixel_idx_1d = (px_v * W + px_u).to(torch.int64)

        gaussian_i_u = u[current_gaussian_ids]  # [N]
        gaussian_i_v = v[current_gaussian_ids]  # [N]
        gaussian_i_color = color[current_gaussian_ids]  # [N, 3]
        gaussian_i_opacity = opacity[current_gaussian_ids]  # [N]
        gaussian_i_inverse_covariance = inverse_covariance[current_gaussian_ids]  # [N, 2, 2]

        # Step 11: Evaluate Gaussian contributions for each pixel in this tile
        # Compute distance from each pixel to each Gaussian center
        du = px_u.unsqueeze(0) - gaussian_i_u.unsqueeze(-1)  # [N, T * T] - u distance
        dv = px_v.unsqueeze(0) - gaussian_i_v.unsqueeze(-1)  # [N, T * T] - v distance
        
        # Extract inverse covariance matrix elements
        # For 2D Gaussian: G(u,v) = exp(-0.5 * [du, dv] @ Σ^(-1) @ [du, dv]^T)
        A11 = gaussian_i_inverse_covariance[:, 0, 0].unsqueeze(-1)  # [N, 1] - Σ^(-1)[0,0]
        A12 = gaussian_i_inverse_covariance[:, 0, 1].unsqueeze(-1)  # [N, 1] - Σ^(-1)[0,1]
        A22 = gaussian_i_inverse_covariance[:, 1, 1].unsqueeze(-1)  # [N, 1] - Σ^(-1)[1,1]
        
        # Compute Mahalanobis distance squared: q = [du, dv] @ Σ^(-1) @ [du, dv]^T
        # Expanded: q = A11*du² + 2*A12*du*dv + A22*dv²
        q = A11 * du * du + 2 * A12 * du * dv + A22 * dv * dv   # [N, T * T]
        
        # Chi-square clipping: Only evaluate Gaussians within 3σ (q <= 9.21 ≈ 3²)
        # This culls Gaussians that contribute negligibly to pixels
        inside = q <= chi_square_clip
        g = torch.exp(-0.5 * torch.clamp(q, max=chi_square_clip))  # [N, T * T] - Gaussian value
        g = torch.where(inside, g, torch.zeros_like(g))  # Zero out contributions outside clipping threshold
        
        # Step 12: Alpha blending
        # Combine Gaussian value with base opacity: α = opacity_base * G(u,v)
        alpha_i = (gaussian_i_opacity.unsqueeze(-1) * g).clamp_max(alpha_max)  # [N, T * T]
        # Alpha cutoff: Skip Gaussians with opacity < 1/255 (invisible)
        alpha_i = torch.where(alpha_i >= alpha_cutoff, alpha_i, torch.zeros_like(alpha_i))
        one_minus_alpha_i = 1 - alpha_i  # [N, T * T]
        
        # Compute transmittance: T_i = Π_{j<i} (1 - α_j)
        # This represents how much light passes through Gaussians in front
        T_i = torch.cumprod(one_minus_alpha_i, dim=0)  # Cumulative product along depth
        # Prepend 1.0 (no occlusion for first Gaussian)
        T_i = torch.concatenate([
            torch.ones((1, alpha_i.shape[-1]), device=pos.device, dtype=pos.dtype),
            T_i[:-1]], dim=0)
        
        # Early termination: Stop blending when transmittance is very low (opaque)
        # More aggressive early termination for speed
        alive = (T_i > 5e-5).float()  # More aggressive threshold
        
        # Final weight: w_i = α_i * T_i
        # This is the contribution of Gaussian i, accounting for occlusion
        w = alpha_i * T_i * alive  # [N, T * T]
        
        # Step 13: Accumulate color contributions
        # Final color: C = Σ_i w_i * c_i (weighted sum of Gaussian colors)
        color_contributions = (w.unsqueeze(-1) * gaussian_i_color.unsqueeze(1)).sum(dim=0)  # [T*T, 3]
        
        # Store contributions and indices for later accumulation
        tile_contributions.append(color_contributions)
        tile_indices.append(pixel_idx_1d)
    
    # Accumulate all contributions using scatter_add (out-of-place to preserve gradients)
    if len(tile_contributions) > 0:
        # Concatenate all contributions and indices
        all_contributions = torch.cat(tile_contributions, dim=0)  # [total_pixels, 3]
        all_indices = torch.cat(tile_indices, dim=0)  # [total_pixels]
        
        # Use scatter_add (out-of-place) to accumulate into final_image
        final_image = final_image.scatter_add(0, all_indices.unsqueeze(-1).expand(-1, 3), all_contributions)
    
    return final_image.reshape((H, W, 3)).clamp(0, 1)

