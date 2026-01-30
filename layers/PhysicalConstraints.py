"""
PhysicalConstraints.py: 

This module provides:
1. ReconLoss - Additive reconstruction loss: MSE(X̂_t + X̂_s, X)
2. SmoothLoss - Trend self-smoothness constraint
3. PeriodicityLoss - Seasonal self-periodicity constraint (autocorrelation regularity)
4. MutualInfoLoss - CLUB upper bound for disentanglement

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft


# ============================================================================
# Helper Functions
# ============================================================================

_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """Returns the next largest number n >= size whose prime factors are all 2, 3, or 5."""
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass
    
    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.
    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    """
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(range(N, 0, -1), dtype=input.dtype, device=input.device)
    autocorr = autocorr / (autocorr[..., :1] + 1e-8)
    return autocorr.transpose(dim, -1)


# ============================================================================
# Loss Functions
# ============================================================================

class ReconLoss(nn.Module):
    """
    Additive reconstruction loss: X̂ = X̂_t + X̂_s should equal X
    
    L_recon = MSE(X̂_t + X̂_s, X)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, x_recon_trend, x_recon_seasonal, x_true):
        """
        Args:
            x_recon_trend: [B, T, C] reconstructed trend
            x_recon_seasonal: [B, T, C] reconstructed seasonal
            x_true: [B, T, C] original signal
        Returns:
            Reconstruction loss (scalar)
        """
        x_recon = x_recon_trend + x_recon_seasonal
        return self.mse(x_recon, x_true)


class SmoothLoss(nn.Module):
    """
    Trend self-smoothness constraint.
    
    Trend should be smooth (small first-order differences).
    L_smooth = mean((X̂_t[1:] - X̂_t[:-1])^2)
    
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x_recon_trend):
        """
        Args:
            x_recon_trend: [B, T, C] reconstructed trend
        Returns:
            Smoothness loss (scalar)
        """
        # First-order difference
        trend_diff = x_recon_trend[:, 1:, :] - x_recon_trend[:, :-1, :]
        
        # Penalize non-smoothness
        loss = (trend_diff ** 2).mean()
        
        return loss


class PeriodicityLoss(nn.Module):
    """
    Seasonal self-periodicity constraint.
    
    Seasonal component should have regular autocorrelation pattern.
    We measure this by checking if autocorrelation has clear peaks.
    
    L_periodic = 1 - normalized_ac_var
    
    High variance in autocorr = clear periodic peaks = low loss
    Low variance in autocorr = no periodic pattern = high loss
    """
    def __init__(self, min_lag=1, max_lag_ratio=0.5):
        super().__init__()
        self.min_lag = min_lag
        self.max_lag_ratio = max_lag_ratio
    
    def forward(self, x_recon_seasonal):
        """
        Args:
            x_recon_seasonal: [B, T, C] reconstructed seasonal
        Returns:
            Periodicity loss (scalar, non-negative)
        """
        B, T, C = x_recon_seasonal.shape
        
        # Flatten to [B*C, T]
        x = x_recon_seasonal.reshape(-1, T)
        
        # Compute autocorrelation
        ac = autocorrelation(x, dim=1)  # [B*C, T]
        
        # Only consider lags from min_lag to max_lag
        max_lag = int(T * self.max_lag_ratio)
        if max_lag <= self.min_lag:
            max_lag = self.min_lag + 1
        ac_valid = ac[:, self.min_lag:max_lag]
        
        # Periodicity measure: variance of autocorr (should be high for periodic)
        ac_var = ac_valid.var(dim=1).mean()
        
        # Normalize variance to [0, 1] range (autocorr is bounded [-1, 1])
        # Max variance for [-1, 1] range is 1.0
        normalized_var = torch.clamp(ac_var, 0, 1)
        
        # Loss: 1 - normalized_var (high variance = low loss, low variance = high loss)
        loss = 1.0 - normalized_var
        
        return loss


class CriticNetwork(nn.Module):
    """
    Critic function for mutual information estimation (CLUB upper bound).
    Improved: Larger hidden dim, no Sigmoid to avoid gradient saturation.
    """
    def __init__(self, x_dim, y_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        cat_dim = x_dim + y_dim
        hidden_dim = hidden_dim or max(cat_dim, 64)  # Increased from cat_dim // 4
        
        self.critic = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyReLU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
            # No Sigmoid - output unbounded for better gradients
        )

    def forward(self, x, y):
        """
        Args:
            x: [B, T, D] or [B, D]
            y: [B, T, D] or [B, D]
        Returns:
            Critic score [B, T, 1] or [B, 1]
        """
        cat = torch.cat((x, y), dim=-1)
        return self.critic(cat)


class MutualInfoLoss(nn.Module):
    """
    Mutual Information upper bound loss for disentanglement.
    Uses CLUB (Contrastive Log-ratio Upper Bound) estimation.
    
    Goal: Minimize I(z_trend, z_seasonal) to ensure independence.
    """
    def __init__(self, z_dim, dropout=0.1):
        super().__init__()
        self.critic = CriticNetwork(z_dim, z_dim, dropout=dropout)
    
    def forward(self, z_trend, z_seasonal):
        """
        Args:
            z_trend: [B, T, D] or [B, D]
            z_seasonal: [B, T, D] or [B, D]
        Returns:
            MI upper bound (to be minimized)
        """
        # Shuffle z_seasonal to break pairing
        idx = torch.randperm(z_seasonal.shape[0])
        z_seasonal_shuffled = z_seasonal[idx]
        
        # CLUB upper bound: E[f(z_t, z_s)] - E[f(z_t, z_s')]
        f_joint = self.critic(z_trend, z_seasonal)
        f_marginal = self.critic(z_trend, z_seasonal_shuffled)
        
        mubo = f_joint - f_marginal
        
        # Regularization for negative values
        pos_mask = (mubo < 0).float()
        reg = ((mubo * pos_mask) ** 2).mean()
        
        return mubo.mean() + reg


# ============================================================================
# Combined Loss Class 
# ============================================================================

class PhysicalConstraintLossV3(nn.Module):
    def __init__(self, z_dim, lambda_smooth=0.1, lambda_periodic=0.1, lambda_mi=0.1, verbose=False):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_periodic = lambda_periodic
        self.lambda_mi = lambda_mi
        self.verbose = verbose
        
        # Loss components
        self.recon_loss = ReconLoss()
        self.smooth_loss = SmoothLoss()
        self.periodic_loss = PeriodicityLoss()
        # Note: L_mi requires high-dim latent representations (z_dim >> C)
        # Currently disabled by default (lambda_mi=0) as model outputs signal-space
        self.mi_loss = MutualInfoLoss(z_dim) if lambda_mi > 0 else None
        
        # For logging
        self.last_losses = {}
    
    def forward(self, x_recon_trend, x_recon_seasonal, x_true, z_trend, z_seasonal, stage=1):
        """
        Args:
            x_recon_trend: Reconstructed trend signal [B, T, C]
            x_recon_seasonal: Reconstructed seasonal signal [B, T, C]
            x_true: Original input signal [B, T, C]
            z_trend: Trend representation [B, T, D] or [B, D]
            z_seasonal: Seasonal representation [B, T, D] or [B, D]
            stage: 1 for pretraining, 2 for training
        
        Returns:
            Total physical constraint loss (scalar)
        """
        # 1. Reconstruction loss (main loss for Stage 1)
        loss_recon = self.recon_loss(x_recon_trend, x_recon_seasonal, x_true)
        
        # 2. Smoothness loss (trend should be smooth)
        loss_smooth = self.smooth_loss(x_recon_trend)
        
        # 3. Periodicity loss (seasonal should have regular pattern)
        loss_periodic = self.periodic_loss(x_recon_seasonal)
        
        # 4. Mutual information loss (disentanglement) - only if enabled
        if self.mi_loss is not None and self.lambda_mi > 0:
            # Pool z to [B, D] if needed
            if z_trend.dim() == 3:
                z_t_pooled = z_trend.mean(dim=1)
                z_s_pooled = z_seasonal.mean(dim=1)
            else:
                z_t_pooled = z_trend
                z_s_pooled = z_seasonal
            loss_mi = self.mi_loss(z_t_pooled, z_s_pooled)
            loss_mi = torch.clamp(loss_mi, min=0)
        else:
            loss_mi = torch.tensor(0.0, device=x_true.device)
        
        # Combine losses
        total_loss = (loss_recon + 
                      self.lambda_smooth * loss_smooth + 
                      self.lambda_periodic * loss_periodic + 
                      self.lambda_mi * loss_mi)
        
        # Store for logging
        self.last_losses = {
            'recon': loss_recon.item(),
            'smooth': loss_smooth.item(),
            'periodic': loss_periodic.item(),
            'mi': loss_mi.item() if isinstance(loss_mi, torch.Tensor) else 0.0,
            'total': total_loss.item()
        }
        
        return total_loss
    
    def get_loss_string(self):
        """Return a formatted string of the last computed losses."""
        if not self.last_losses:
            return ""
        return (f"L_rec={self.last_losses['recon']:.3f} "
                f"L_smo={self.last_losses['smooth']:.3f} "
                f"L_per={self.last_losses['periodic']:.3f} "
                f"L_mi={self.last_losses['mi']:.3f}")


# ============================================================================
# Legacy V2 Loss (Kept for Compatibility)
# ============================================================================

def period_similarity(x, y):
    """Legacy: Periodicity constraint using autocorrelation similarity."""
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[1])
        y = y.reshape(-1, y.shape[1])
    
    x_ac = autocorrelation(x, dim=1)[:, 1:]
    y_ac = autocorrelation(y, dim=1)[:, 1:]
    
    distance = ((x_ac - y_ac) ** 2).mean(dim=1).mean()
    return -distance


def shape_extract(x, mode="value"):
    """Extract first-order difference (rate of change)."""
    x_diff = x[:, 1:] - x[:, :-1]
    if mode == "binary":
        x_diff[x_diff > 0] = 1.0
        x_diff[x_diff <= 0] = 0.0
        x_diff = x_diff.type(torch.LongTensor).to(x.device)
    return x_diff


def trend_similarity(x, y):
    """Legacy: Trend constraint using CORT."""
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[1])
        y = y.reshape(-1, y.shape[1])
    
    x_t = shape_extract(x)
    y_t = shape_extract(y)
    
    denominator = torch.sqrt((x_t ** 2).sum(dim=1)) * torch.sqrt((y_t ** 2).sum(dim=1))
    numerator = (x_t * y_t).sum(dim=1)
    cort = numerator / (denominator + 1e-8)
    
    return cort.mean()


class PhysicalConstraintLoss(nn.Module):
    """
    Legacy V2 Combined physical constraint loss.
    Kept for backward compatibility.
    """
    def __init__(self, z_dim, lambda_trend=1.0, lambda_seasonal=1.0, lambda_mi=0.1, verbose=False):
        super().__init__()
        self.lambda_trend = lambda_trend
        self.lambda_seasonal = lambda_seasonal
        self.lambda_mi = lambda_mi
        self.verbose = verbose
        
        self.mi_loss = MutualInfoLoss(z_dim)
        self.last_losses = {}
    
    def forward(self, x_recon_trend, x_recon_seasonal, x_true, z_trend, z_seasonal):
        cort_score = trend_similarity(x_recon_trend, x_true)
        loss_trend = 1.0 - cort_score
        
        period_score = period_similarity(x_recon_seasonal, x_true)
        loss_seasonal = -period_score
        
        loss_mi = self.mi_loss(z_trend, z_seasonal)
        loss_mi = torch.clamp(loss_mi, min=0)
        
        total_loss = (self.lambda_trend * loss_trend + 
                      self.lambda_seasonal * loss_seasonal + 
                      self.lambda_mi * loss_mi)
        
        self.last_losses = {
            'cort': cort_score.item(),
            'loss_t': loss_trend.item(),
            'loss_s': loss_seasonal.item(),
            'loss_mi': loss_mi.item(),
            'total': total_loss.item()
        }
        
        return total_loss
    
    def get_loss_string(self):
        if not self.last_losses:
            return ""
        return (f"CORT={self.last_losses['cort']:.3f} "
                f"L_t={self.last_losses['loss_t']:.3f} "
                f"L_s={self.last_losses['loss_s']:.3f} "
                f"L_mi={self.last_losses['loss_mi']:.3f}")
