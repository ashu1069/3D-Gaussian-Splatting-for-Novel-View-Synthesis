"""
Loss functions for training 3D Gaussian Splatting.

Training 3D Gaussians requires a combination of reconstruction losses to ensure
the rendered images match the ground truth. The standard approach uses:

1. L1 Loss: Measures absolute pixel-wise differences
   L_L1 = |I_rendered - I_gt|

2. SSIM Loss: Structural Similarity Index - measures perceptual quality
   SSIM considers luminance, contrast, and structure between images
   L_SSIM = 1 - SSIM(I_rendered, I_gt)

The total loss is typically:
   L_total = λ_L1 * L_L1 + λ_SSIM * L_SSIM

where λ_L1 ≈ 0.8 and λ_SSIM ≈ 0.2 are weighting factors.

SSIM is particularly important because it captures perceptual quality better than
pixel-wise metrics - it's more sensitive to structural differences that humans notice.
"""

import torch
import torch.nn.functional as F


def l1_loss(pred, target):
    """
    Compute L1 (mean absolute error) loss.
    
    L1 loss measures the average absolute difference between predicted and target pixels.
    It's more robust to outliers than L2 loss and encourages sparsity.
    
    Args:
        pred: Predicted image. Shape: [B, H, W, 3] or [H, W, 3]
        target: Target image. Shape: [B, H, W, 3] or [H, W, 3]
        
    Returns:
        L1 loss (scalar)
    """
    return F.l1_loss(pred, target)


def ssim_loss(pred, target, window_size=11, size_average=True):
    """
    Compute SSIM (Structural Similarity Index) loss.
    
    Theory:
    ------
    SSIM measures perceptual similarity between two images by comparing:
    1. Luminance: μ_pred vs μ_target (mean intensity)
    2. Contrast: σ_pred vs σ_target (standard deviation)
    3. Structure: Covariance between images
    
    SSIM(x, y) = [2μ_x μ_y + c1][2σ_xy + c2] / [(μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2)]
    
    where c1, c2 are small constants for numerical stability.
    
    SSIM loss = 1 - SSIM, so minimizing loss maximizes similarity.
    
    Args:
        pred: Predicted image. Shape: [B, H, W, 3] or [H, W, 3]
        target: Target image. Shape: [B, H, W, 3] or [H, W, 3]
        window_size: Size of Gaussian window for local SSIM computation
        size_average: Whether to average over spatial dimensions
        
    Returns:
        SSIM loss (scalar)
    """
    # Ensure batch dimension
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Convert to [B, C, H, W] format
    pred = pred.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    
    # Compute SSIM per channel and average
    ssim_per_channel = []
    for c in range(pred.shape[1]):
        ssim_val = _ssim_single_channel(
            pred[:, c:c+1], target[:, c:c+1], window_size
        )
        ssim_per_channel.append(ssim_val)
    
    ssim = torch.stack(ssim_per_channel).mean()
    return 1 - ssim


def _ssim_single_channel(pred, target, window_size=11):
    """
    Compute SSIM for a single channel.
    
    Args:
        pred: Predicted image. Shape: [B, 1, H, W]
        target: Target image. Shape: [B, 1, H, W]
        window_size: Size of Gaussian window
        
    Returns:
        SSIM value (scalar)
    """
    # Constants for numerical stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    gauss = _create_gaussian_window(window_size, 1.5, pred.device, pred.dtype)
    gauss = gauss.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    
    # Compute local means using convolution
    mu1 = F.conv2d(pred, gauss, padding=window_size//2)
    mu2 = F.conv2d(target, gauss, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(pred * pred, gauss, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(target * target, gauss, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred * target, gauss, padding=window_size//2) - mu1_mu2
    
    # Compute SSIM
    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2
    
    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    
    return ssim_map.mean()


def _create_gaussian_window(window_size, sigma, device, dtype):
    """
    Create a 2D Gaussian window for SSIM computation.
    
    Args:
        window_size: Size of the window
        sigma: Standard deviation of Gaussian
        device: Device to create tensor on
        dtype: Data type
        
    Returns:
        Gaussian window. Shape: [window_size, window_size]
    """
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D window
    gauss = g.unsqueeze(1) * g.unsqueeze(0)
    return gauss


def compute_loss(pred, target, lambda_l1=0.8, lambda_ssim=0.2):
    """
    Compute combined L1 + SSIM loss for training.
    
    This is the standard loss function used in 3D Gaussian Splatting training.
    The combination of L1 (pixel accuracy) and SSIM (perceptual quality) ensures
    both accurate reconstruction and visually pleasing results.
    
    Args:
        pred: Predicted image. Shape: [H, W, 3] or [B, H, W, 3]
        target: Target image. Shape: [H, W, 3] or [B, H, W, 3]
        lambda_l1: Weight for L1 loss (default: 0.8)
        lambda_ssim: Weight for SSIM loss (default: 0.2)
        
    Returns:
        Total loss (scalar)
        Dictionary with individual loss components
    """
    l1 = l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    
    total_loss = lambda_l1 * l1 + lambda_ssim * ssim
    
    return total_loss, {
        'l1': l1.item(),
        'ssim': ssim.item(),
        'total': total_loss.item()
    }

