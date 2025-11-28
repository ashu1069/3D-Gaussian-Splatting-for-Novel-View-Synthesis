"""
Utility functions for 3D Gaussian Splatting.

This module contains fundamental geometric operations used throughout the rendering pipeline.
"""

import torch


def transform_to_camera_space(pc, c2w):
    """
    Transform 3D points from world coordinates to camera space.
    
    This is the first step of projection - we transform points to camera space
    without doing the actual projection. This allows us to check frustum bounds
    before performing the expensive projection operation.
    
    Args:
        pc: 3D points in world coordinates. Shape: [N, 3]
        c2w: Camera-to-world transform matrix. Shape: [4, 4]
        
    Returns:
        x, y, z: Points in camera space. Shape: [N] each
    """
    w2c = torch.eye(4, device=pc.device, dtype=pc.dtype)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.t()
    w2c[:3, 3] = -R.t() @ t

    PC = ((w2c @ torch.concatenate(
        [pc, torch.ones_like(pc[:, :1])], dim=1).t()).t())[:, :3]
    x, y, z = PC[:, 0], PC[:, 1], PC[:, 2]  # Camera space
    return x, y, z


def check_frustum_camera_space(x, y, z, fx, fy, cx, cy, H, W, near, far, pix_guard):
    """
    Check if points in camera space fall within the viewing frustum.
    
    This function checks frustum bounds BEFORE projection, which is more efficient
    than projecting first and then checking. We check:
    1. Points must be in front of camera: z > 0 (critical to avoid projection errors)
    2. Depth bounds: z must be between near and far planes
    3. Image bounds: The projected point would fall within image bounds (with guard band)
    
    For a point to project within bounds:
        -pix_guard < u < W + pix_guard  where u = fx * x/z + cx
        -pix_guard < v < H + pix_guard  where v = fy * y/z + cy
    
    Rearranging these inequalities (assuming z > 0):
        z * (-pix_guard - cx) < fx * x < z * (W + pix_guard - cx)
        z * (-pix_guard - cy) < fy * y < z * (H + pix_guard - cy)
    
    Note: These inequalities only hold when z > 0. Points behind the camera (z <= 0)
    would cause division by zero or negative z, leading to incorrect projections.
    
    Args:
        x, y, z: Points in camera space. Shape: [N] each
        fx, fy: Focal lengths (in pixels)
        cx, cy: Principal point coordinates (in pixels)
        H, W: Image height and width
        near, far: Near and far clipping planes
        pix_guard: Pixel guard band for conservative culling
        
    Returns:
        mask: Boolean mask indicating which points are in frustum. Shape: [N]
    """
    # CRITICAL: Filter out points behind the camera (z <= 0)
    # Points behind camera would cause division by zero or negative z in projection,
    # leading to incorrect or invalid projected coordinates
    in_front = z > 0
    
    # Check depth bounds (z must be between near and far planes)
    depth_ok = (z > near) & (z < far)
    
    # Check image bounds in camera space (avoiding division by z)
    # These inequalities only make sense when z > 0, which we've already checked
    # For u = fx * x/z + cx to be in [-pix_guard, W + pix_guard]:
    #   -pix_guard - cx < fx * x/z < W + pix_guard - cx
    #   z * (-pix_guard - cx) < fx * x < z * (W + pix_guard - cx)
    u_min_bound = z * (-pix_guard - cx)
    u_max_bound = z * (W + pix_guard - cx)
    fx_x = fx * x
    u_ok = (fx_x > u_min_bound) & (fx_x < u_max_bound)
    
    # For v = fy * y/z + cy to be in [-pix_guard, H + pix_guard]:
    #   -pix_guard - cy < fy * y/z < H + pix_guard - cy
    #   z * (-pix_guard - cy) < fy * y < z * (H + pix_guard - cy)
    v_min_bound = z * (-pix_guard - cy)
    v_max_bound = z * (H + pix_guard - cy)
    fy_y = fy * y
    v_ok = (fy_y > v_min_bound) & (fy_y < v_max_bound)
    
    # All conditions must be satisfied: in front of camera, within depth bounds, and within image bounds
    return in_front & depth_ok & u_ok & v_ok


def project_points(pc, c2w, fx, fy, cx, cy):
    """
    Project 3D points from world coordinates to camera space and image coordinates.
    
    This function implements the pinhole camera model, which projects 3D world points onto
    a 2D image plane. The process involves two main transformations:
    
    1. World-to-Camera Transformation:
       We need to transform points from world coordinates to camera coordinates. Given a
       camera-to-world matrix c2w = [R | t] (where R is rotation and t is translation),
       the world-to-camera transform is w2c = [R^T | -R^T @ t].
       
       In homogeneous coordinates, a point P_world = [x, y, z, 1]^T is transformed to
       camera space as: P_camera = w2c @ P_world
    
    2. Perspective Projection:
       The pinhole camera model projects 3D points onto a 2D image plane using:
       u = fx * (X_cam / Z_cam) + cx
       v = fy * (Y_cam / Z_cam) + cy
       
       where:
       - fx, fy are focal lengths (in pixels) representing the camera's field of view
       - cx, cy are the principal point (image center) coordinates
       - The division by Z_cam implements perspective projection (farther objects appear smaller)
    
    The focal length determines how "zoomed in" the camera is - larger fx/fy means narrower
    field of view. The principal point accounts for any offset of the optical axis from the
    image center.
    
    Args:
        pc: 3D points in world coordinates. Shape: [N, 3]
        c2w: Camera-to-world transform matrix. Shape: [4, 4]
        fx, fy: Focal lengths (in pixels)
        cx, cy: Principal point coordinates (in pixels)
        
    Returns:
        uv: Image coordinates. Shape: [N, 2]
        x, y, z: Points in camera space. Shape: [N] each
    """
    w2c = torch.eye(4, device=pc.device, dtype=pc.dtype)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.t()
    w2c[:3, 3] = -R.t() @ t

    PC = ((w2c @ torch.concatenate(
        [pc, torch.ones_like(pc[:, :1])], dim=1).t()).t())[:, :3]
    x, y, z = PC[:, 0], PC[:, 1], PC[:, 2]  # Camera space

    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    return uv, x, y, z


def inv2x2(M, eps=1e-12):
    """
    Compute the inverse of a batch of 2x2 matrices.
    
    For a 2x2 matrix M = [[a, b], [c, d]], the inverse is given by:
    
        M^(-1) = (1/det(M)) * [[d, -b], [-c, a]]
    
    where det(M) = ad - bc is the determinant.
    
    This formula is derived from Cramer's rule and is more efficient than general matrix
    inversion methods (like Gaussian elimination) for 2x2 matrices. We use this because
    during rendering, we need to invert 2D covariance matrices (projected from 3D) for
    thousands of Gaussians, so efficiency matters.
    
    Numerical Stability:
    The determinant can be very small or zero, leading to numerical issues. We clamp it
    to a minimum value (eps) to prevent division by zero and ensure numerical stability.
    This is safe because in rendering, very small determinants correspond to Gaussians
    that are essentially invisible (degenerate cases).
    
    Args:
        M: Batch of 2x2 matrices. Shape: [N, 2, 2]
        eps: Small epsilon for numerical stability (prevents division by zero)
        
    Returns:
        Inverse matrices. Shape: [N, 2, 2]
    """
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    safe_det = torch.clamp(det, min=eps)
    inv = torch.empty_like(M)
    inv[:, 0, 0] = d / safe_det
    inv[:, 0, 1] = -b / safe_det
    inv[:, 1, 0] = -c / safe_det
    inv[:, 1, 1] = a / safe_det
    return inv


def scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy):
    """
    Scale camera intrinsics to match a different image resolution.
    
    When rendering at a different resolution than the original camera calibration, we need
    to scale the intrinsic parameters proportionally. This preserves the camera's field of
    view and relative geometry.
    
    The scaling factors are:
        scale_x = W / W_src  (horizontal scaling)
        scale_y = H / H_src  (vertical scaling)
    
    The focal lengths scale linearly with image size:
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
    
    This makes sense because focal length in pixels is proportional to image size - if you
    double the image width, you need to double fx to maintain the same field of view.
    
    The principal point also scales:
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
    
    This ensures the principal point remains at the same relative position in the image
    (e.g., if it was at the center before, it stays at the center after scaling).
    
    Example:
    If rendering at half resolution (scale_factor=2), all intrinsics are halved, preserving
    the camera geometry but reducing computational cost.
    
    Args:
        H, W: Target image height and width
        H_src, W_src: Source image height and width
        fx, fy, cx, cy: Source camera intrinsics
        
    Returns:
        Scaled camera intrinsics (fx_scaled, fy_scaled, cx_scaled, cy_scaled)
    """
    scale_x = W / W_src
    scale_y = H / H_src
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    return fx_scaled, fy_scaled, cx_scaled, cy_scaled

