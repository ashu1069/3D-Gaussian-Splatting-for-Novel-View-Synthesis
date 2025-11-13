"""
Gaussian parameterization and transformation utilities.

Theory:
------
In 3D Gaussian Splatting, each Gaussian is parameterized by:
1. Position (μ): 3D center point
2. Covariance matrix (Σ): 3x3 symmetric positive-definite matrix describing the shape/orientation
3. Opacity (α): Controls transparency
4. Color: View-dependent appearance via spherical harmonics

The covariance matrix Σ determines how the Gaussian is oriented and scaled in 3D space.
It's decomposed as: Σ = R S S^T R^T, where:
- R is a rotation matrix (3x3) representing orientation
- S is a diagonal scale matrix (3x3) representing size along principal axes

We store rotations as quaternions (4 parameters) instead of rotation matrices (9 parameters)
because quaternions are more compact, avoid gimbal lock, and interpolate smoothly.
"""

import torch


def quat_to_rotmat(quat):
    """
    Convert quaternions to rotation matrices.
    
    Theory:
    ------
    Quaternions are a 4D representation of 3D rotations, more efficient than Euler angles
    (no gimbal lock) and more compact than rotation matrices (4 vs 9 parameters).
    
    A quaternion q = (x, y, z, w) represents a rotation by angle θ around axis (x, y, z):
        q = (sin(θ/2) * axis, cos(θ/2))
    
    The rotation matrix R corresponding to quaternion q = (x, y, z, w) is:
    
        R = [[1-2(y²+z²),  2(xy-wz),     2(xz+wy)    ],
             [2(xy+wz),     1-2(x²+z²),  2(yz-wx)    ],
             [2(xz-wy),     2(yz+wx),     1-2(x²+y²)  ]]
    
    This formula is derived from Rodrigues' rotation formula. The quaternion must be
    normalized (unit quaternion) to represent a pure rotation.
    
    Why quaternions in 3DGS?
    - Compact: 4 parameters vs 9 for rotation matrix
    - Smooth optimization: gradients flow better than Euler angles
    - No gimbal lock: unlike Euler angles, all orientations are representable
    - Efficient conversion: this formula is faster than matrix exponentiation
    
    Args:
        quat: Quaternions in (x, y, z, w) format. Shape: [..., 4]
              Should be normalized (unit quaternions) for proper rotations.
        
    Returns:
        Rotation matrices. Shape: [..., 3, 3]
    """
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))
    return R


def build_sigma_from_params(scale_raw, q_raw):
    """
    Build 3D covariance matrix from scale and rotation parameters.
    
    Theory:
    ------
    The covariance matrix Σ describes the shape and orientation of a 3D Gaussian. It's
    constructed via eigendecomposition:
    
        Σ = R S S^T R^T
    
    where:
    - R is a rotation matrix (from quaternion) representing orientation
    - S is a diagonal matrix with scale factors [sx, sy, sz] along principal axes
    
    Why S @ S instead of just S?
    The covariance matrix must be positive semi-definite. If we have scale factors s,
    the variance along each axis is s². So we use S @ S = diag(s²) to ensure proper
    scaling. This is equivalent to Σ = R diag(s²) R^T.
    
    Parameterization choices:
    1. scale_raw: We store log(scale) and exponentiate it. This ensures scale > 0
       (covariance must be positive definite) and makes optimization easier (unconstrained).
    2. q_raw: Raw quaternion that gets normalized. Normalization ensures it represents
       a pure rotation (unit quaternion).
    
    The covariance matrix determines:
    - Shape: How elongated/flat the Gaussian is (via scale factors)
    - Orientation: Which direction it's pointing (via rotation)
    - Size: How large the Gaussian appears (via magnitude of scale)
    
    During rendering, this 3D covariance is projected to 2D image space using the
    Jacobian of the projection function (see render.py).
    
    Args:
        scale_raw: Raw scale parameters (log-space, will be exponentiated). Shape: [N, 3]
                   Stored in log-space to ensure positivity and improve optimization.
        q_raw: Raw quaternion parameters (will be normalized to unit quaternion). Shape: [N, 4]
        
    Returns:
        Covariance matrices Σ. Shape: [N, 3, 3]
        Each matrix is symmetric positive-definite, representing a 3D Gaussian's shape.
    """
    # Exponentiate to get positive scale factors (covariance must be positive definite)
    scale = torch.exp(scale_raw).clamp_min(1e-6)
    
    # Normalize quaternion to unit quaternion (ensures pure rotation)
    q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + 1e-9)
    
    # Convert quaternion to rotation matrix
    R = quat_to_rotmat(q)
    
    # Build diagonal scale matrix S = diag(sx, sy, sz)
    S = torch.diag_embed(scale)
    
    # Construct covariance: Σ = R S S^T R^T = R diag(s²) R^T
    return R @ S @ S @ R.transpose(1, 2)

