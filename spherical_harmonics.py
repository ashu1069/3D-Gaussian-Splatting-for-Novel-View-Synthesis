"""
Spherical harmonics evaluation for view-dependent appearance in 3D Gaussian Splatting.

Spherical Harmonics (SH) are a set of basis functions defined on the sphere, analogous to
Fourier series but for spherical functions. They're used to represent view-dependent
appearance - how color changes with viewing angle.

Mathematical Foundation:
-----------------------
Real spherical harmonics Y_l^m(θ, φ) form an orthonormal basis for functions on the sphere,
where:
- l (degree): 0, 1, 2, ... determines the "frequency" of variation
- m (order): -l, ..., +l determines the pattern orientation
- θ, φ: Spherical coordinates (elevation, azimuth)

For degree l, there are (2l+1) basis functions. We use degrees 0-3, giving:
- l=0: 1 function (constant, view-independent)
- l=1: 3 functions (linear variation)
- l=2: 5 functions (quadratic variation)
- l=3: 7 functions (cubic variation)
Total: 16 basis functions

In 3DGS:
--------
Each Gaussian stores SH coefficients f = [f_dc, f_rest] where:
- f_dc: DC term (l=0) - view-independent base color [3 channels]
- f_rest: Higher-order terms (l=1,2,3) - view-dependent variations [45 coefficients = 15 terms × 3 channels]

To compute color for a view direction d (normalized vector from camera to point):
    color(d) = sigmoid(Σ_k f_k * Y_k(d))

where Y_k are the 16 SH basis functions evaluated at direction d, and f_k are the learned
coefficients. The sigmoid ensures colors stay in [0,1].

Why SH?
-------
1. Compact: 16 coefficients capture complex view-dependent effects
2. Smooth: Low-frequency basis ensures smooth color transitions
3. Efficient: Evaluation is just polynomial evaluation (fast)
4. Learnable: Coefficients can be optimized via gradient descent

The constants HARMONICS are normalization factors (C_l^m) that ensure orthonormality.
They're precomputed because they involve factorials and square roots of π, which are
expensive to compute at runtime.
"""

import torch

# Precomputed normalization factors for real spherical harmonics basis functions up to degree 3
HARMONICS = {
    'SH_C0': 0.28209479177387814,
    'SH_C1_x': 0.4886025119029199,
    'SH_C1_y': 0.4886025119029199,
    'SH_C1_z': 0.4886025119029199,
    'SH_C2_xy': 1.0925484305920792,
    'SH_C2_xz': 1.0925484305920792,
    'SH_C2_yz': 1.0925484305920792,
    'SH_C2_zz': 0.31539156525252005,
    'SH_C2_xx_yy': 0.5462742152960396,
    'SH_C3_yxx_yyy': 0.5900435899266435,
    'SH_C3_xyz': 2.890611442640554,
    'SH_C3_yzz_yxx_yyy': 0.4570457994644658,
    'SH_C3_zzz_zxx_zyy': 0.3731763325901154,
    'SH_C3_xzz_xxx_xyy': 0.4570457994644658,
    'SH_C3_zxx_zyy': 1.445305721320277,
    'SH_C3_xxx_xyy': 0.5900435899266435,
}


def evaluate_sh(f_dc, f_rest, points, c2w):
    """
    Evaluate spherical harmonics to compute view-dependent colors.
    
    This function implements the SH color evaluation formula:
    
        color(view_dir) = sigmoid(Σ_{k=0}^{15} f_k * Y_k(view_dir))
    
    Steps:
    1. Pack coefficients: Combine f_dc (DC term) and f_rest (higher-order terms) into
       a [N, 16, 3] tensor where each of the 16 SH basis functions has RGB coefficients.
    
    2. Compute view direction: For each Gaussian at position p, compute the normalized
       vector from camera position c to p: d = normalize(p - c)
       
       This direction vector d = (x, y, z) is what we evaluate the SH basis at.
    
    3. Evaluate SH basis: Compute all 16 real SH basis functions Y_0...Y_15 at direction d.
       These are polynomials in x, y, z. For example:
       - Y_0 = constant (view-independent)
       - Y_1, Y_2, Y_3 = linear terms (x, y, z)
       - Y_4...Y_8 = quadratic terms (xy, xz, yz, z², x²-y²)
       - Y_9...Y_15 = cubic terms (more complex combinations)
    
    4. Weighted sum: Multiply each basis value by its corresponding coefficient and sum:
       color = Σ_k (f_k * Y_k(d))
    
    5. Sigmoid: Apply sigmoid to map raw RGB values to [0,1] range for display.
    
    The result is a view-dependent color that changes smoothly as the viewing angle changes,
    enabling realistic rendering of materials with specular highlights, reflections, etc.

    Args:
        f_dc: Tensor with constant SH terms (l=0) per point. Shape: [N, 3]
              This is the view-independent base color.
        f_rest: Learned SH coefficients for l=1,2,3. Shape: [N, 45]
                Organized as [15 SH terms × 3 RGB channels] = 45 total
        points: 3D world positions of the Gaussians. Shape: [N, 3]
        c2w: Camera-to-world transform matrix. Shape: [4, 4]
             Used to extract camera position: c = c2w[:3, 3]
        
    Returns:
        RGB colors per point, view-dependent. Shape: [N, 3]
        Values are in [0,1] range after sigmoid activation.
    """
    # Create an uninitialized buffer for SH coefficients per point. Shape: [N, 16, 3]
    sh = torch.empty((points.shape[0], 16, 3),
                    device=points.device, dtype=points.dtype)

    # Index 0: DC constant (Y00). f_dc is used directly as the RGB coefficient for that basis
    sh[:, 0] = f_dc

    # f_rest stores 15 SH coefficients per channel
    sh[:, 1:, 0] = f_rest[:, :15]  # R
    sh[:, 1:, 1] = f_rest[:, 15:30]  # G
    sh[:, 1:, 2] = f_rest[:, 30:45]  # B
    # => sh[n,k,c] holds the coefficient for basis k and color channel c for point n.

    # Compute view direction: c2w[:3,3] -> camera position in world coordinates (translation column)
    # c2w = [[R,t], [0,1]] where R is 3x3
    view_dir = points - c2w[:3, 3].unsqueeze(0)  # Shape: [N,3] => yields vector from camera->point
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)  # normalization
    x, y, z = view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]  # components of the normalized view direction for each point
    # => Please note that sign convention matters, (point - camera) view vector point from camera to the point.
    # If elsewhere you expected a view vector pointing toward the camera, you'll need to flip the sign.

    # Precompute powers/cross terms
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    # => reused heavily when computing Cartesian forms of real SH basis functions (faster than recomputing)

    # Build the 16 real SH basis values Y0..Y15, below is the mapping.
    # These expressions are the real SH basis, expressed in x,y,z - exactly the basis used by 3DGS
    Y0 = torch.full_like(x, HARMONICS['SH_C0'])  # [N]: l=0 (constant)
    Y1 = -HARMONICS['SH_C1_y'] * y  # l=1, m = -1
    Y2 = HARMONICS['SH_C1_z'] * z  # l=1, m = 0
    Y3 = -HARMONICS['SH_C1_x'] * x  # l=1, m = +1
    Y4 = HARMONICS['SH_C2_xy'] * xy  # l=2 cross-term xy
    Y5 = HARMONICS['SH_C2_yz'] * yz  # l=2 cross-term yz
    Y6 = HARMONICS['SH_C2_zz'] * (3 * zz - 1)  # l=2 zonal
    Y7 = HARMONICS['SH_C2_xz'] * xz
    Y8 = HARMONICS['SH_C2_xx_yy'] * (xx - yy)
    Y9 = HARMONICS['SH_C3_yxx_yyy'] * y * (3 * xx - yy)
    Y10 = HARMONICS['SH_C3_xyz'] * x * y * z
    Y11 = HARMONICS['SH_C3_yzz_yxx_yyy'] * y * (4 * zz - xx - yy)
    Y12 = HARMONICS['SH_C3_zzz_zxx_zyy'] * z * (2 * zz - 3 * xx - 3 * yy)
    Y13 = HARMONICS['SH_C3_xzz_xxx_xyy'] * x * (4 * zz - xx - yy)
    Y14 = HARMONICS['SH_C3_zxx_zyy'] * z * (xx - yy)
    Y15 = HARMONICS['SH_C3_xxx_xyy'] * x * (xx - 3 * yy)

    Y = torch.stack([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15],
                    dim=1)  # [N, 16]
    # => broadcasting multiplies each SH coefficient for each color by the scalar basis value for that point.
    # sigmoid maps the raw RGB to (0,1) per channel -> learned colored output.
    return torch.sigmoid((sh * Y.unsqueeze(2)).sum(dim=1))

