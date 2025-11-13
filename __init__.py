"""
Fast Gaussian Splatting for Novel View Synthesis

A clean implementation of 3D Gaussian Splatting from scratch.
"""

from .gaussian import build_sigma_from_params, quat_to_rotmat
from .spherical_harmonics import evaluate_sh, HARMONICS
from .render import render
from .utils import project_points, inv2x2, scale_intrinsics

__all__ = [
    'build_sigma_from_params',
    'quat_to_rotmat',
    'evaluate_sh',
    'HARMONICS',
    'render',
    'project_points',
    'inv2x2',
    'scale_intrinsics',
]

