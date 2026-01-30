# SemanticMoE.py: Semantic Mixture-of-Experts for PrismGTS.


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft


# ============================================================================
# Physical Feature Sensors 
# ============================================================================

class MultiScaleTrendSensor(nn.Module):
    """
    CoST-style multi-scale convolutional feature extractor for trend.
    Uses multiple kernel sizes to capture trends at different scales.
    """
    def __init__(self, d_model, kernels=[1, 2, 4, 8, 16, 32]):
        super().__init__()
        self.kernels = kernels
        self.d_out = d_model // 2
        
        # Multi-scale 1D convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, self.d_out, k, padding=k-1)
            for k in kernels
        ])
        
        # Projection to match d_model
        self.proj = nn.Linear(self.d_out, d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features
        Returns:
            trend_feat: [B, T, D] multi-scale trend features
        """
        # [B, T, D] -> [B, D, T]
        x_t = x.transpose(1, 2)
        
        outputs = []
        for i, conv in enumerate(self.convs):
            out = conv(x_t)
            # Trim to original length
            if self.kernels[i] != 1:
                out = out[..., :-(self.kernels[i] - 1)]
            outputs.append(out)
        
        # Average across scales: [B, D//2, T]
        trend_raw = torch.stack(outputs, dim=0).mean(dim=0)
        
        # [B, D//2, T] -> [B, T, D//2] -> [B, T, D]
        trend_feat = self.proj(trend_raw.transpose(1, 2))
        
        return trend_feat


class FrequencySeasonalSensor(nn.Module):
    def __init__(self, d_model, top_k=None): # top_k kept for compat but unused
        super().__init__()
        self.d_model = d_model
        
        # Simplified robust implementation:
        # 1. Project in time domain
        # 2. FFT
        # 3. Learnable Complex Weight (per feature dim)
        # 4. iFFT
        
        # We use a simple parameter that scales spectrum amplitude/phase.
        # Since freq_len varies, we can't learn per-frequency weight easily without fixed length.
        # Here we stick to a channel-wise mixing in freq domain to be length-agnostic.
        
        self.freq_proj_real = nn.Linear(d_model, d_model)
        self.freq_proj_imag = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features
        Returns:
            seasonal_feat: [B, T, D] frequency-domain seasonal features
        """
        B, T, D = x.shape
        
        # FFT along time dimension
        x_fft = rfft(x, dim=1)  # [B, freq_len, D] complex
        
        # Apply learnable complex interaction
        # (Real + jIm) * (W_r + jW_i) = (Real*W_r - Im*W_i) + j(Real*W_i + Im*W_r)
        
        real = x_fft.real
        imag = x_fft.imag
        
        out_real = self.freq_proj_real(real) - self.freq_proj_imag(imag)
        out_imag = self.freq_proj_real(imag) + self.freq_proj_imag(real)
        
        x_fft_out = torch.complex(out_real, out_imag)
        
        # iFFT back to time domain
        seasonal_feat = irfft(x_fft_out, n=T, dim=1)  # [B, T, D]
        
        return seasonal_feat


# ============================================================================
# Semantic Router
# ============================================================================

class SemanticRouter(nn.Module):
    """
    Routes based on physical properties of the input.
    
    - High CORT score → more weight to trend experts
    - High periodicity score → more weight to seasonal experts
    """
    def __init__(self, d_model, num_experts=8):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Feature extractors for routing decision
        self.trend_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_experts)
        )
        
        self.seasonal_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_experts)
        )
    
    def forward(self, trend_feat, seasonal_feat):
        """
        Args:
            trend_feat: [B, T, D] trend features from sensor
            seasonal_feat: [B, T, D] seasonal features from sensor
        Returns:
            gate_trend: [B, T, num_experts] routing weights for trend
            gate_seasonal: [B, T, num_experts] routing weights for seasonal
        """
        # Analyze physical properties
        gate_trend = F.softmax(self.trend_analyzer(trend_feat), dim=-1)
        gate_seasonal = F.softmax(self.seasonal_analyzer(seasonal_feat), dim=-1)
        
        return gate_trend, gate_seasonal


# ============================================================================
# Semantic MoE Layer (SEMPO-compatible)
# ============================================================================

class SemanticMoELayer(nn.Module):
    """
    Semantic Mixture-of-Experts layer for prefix generation.
    
    Generates K/V prefixes conditioned on physical semantics.
    """
    def __init__(self, d_model, num_experts, prompt_len, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.prompt_len = prompt_len
        self.n_layers = n_layers
        
        # Physical Sensors
        self.trend_sensor = MultiScaleTrendSensor(d_model)
        self.seasonal_sensor = FrequencySeasonalSensor(d_model)
        
        # Semantic Router
        self.router = SemanticRouter(d_model, num_experts)
        
        # Expert embeddings (like SEMPO)
        self.trend_experts = nn.Embedding(num_experts, d_model)
        self.seasonal_experts = nn.Embedding(num_experts, d_model)
        
        # Prefix transform (like SEMPO)
        self.trend_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_layers * 2 * d_model)  # K and V for each layer
        )
        
        self.seasonal_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_layers * 2 * d_model)
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, T, D = x.shape
        device = x.device
        
        # 1. Extract physical features
        trend_feat = self.trend_sensor(x)      # [B, T, D]
        seasonal_feat = self.seasonal_sensor(x) # [B, T, D]
        
        # 2. Get routing weights
        gate_trend, gate_seasonal = self.router(trend_feat, seasonal_feat)
        # gate_*: [B, T, num_experts]
        
        # 3. Get expert embeddings
        expert_idx = torch.arange(self.num_experts, device=device)
        trend_emb = self.trend_experts(expert_idx)      # [num_experts, D]
        seasonal_emb = self.seasonal_experts(expert_idx) # [num_experts, D]
        
        # 4. Weighted combination of experts
        # [B, T, num_experts] @ [num_experts, D] -> [B, T, D]
        trend_ctx = torch.einsum('bte,ed->btd', gate_trend, trend_emb)
        seasonal_ctx = torch.einsum('bte,ed->btd', gate_seasonal, seasonal_emb)
        
        # 5. Transform to prefix
        trend_prefix = self.trend_transform(trend_ctx)     # [B, T, n_layers*2*D]
        seasonal_prefix = self.seasonal_transform(seasonal_ctx) # [B, T, n_layers*2*D]
        
        # 6. Fuse trend and seasonal prefixes
        fusion_input = torch.cat([trend_ctx, seasonal_ctx], dim=-1)
        fusion_weight = self.fusion_gate(fusion_input)  # [B, T, D]
        
        # Combine prefixes with learned fusion
        prefix_raw = fusion_weight.unsqueeze(-1) * trend_prefix.view(B, T, self.n_layers, 2, D) + \
                     (1 - fusion_weight.unsqueeze(-1)) * seasonal_prefix.view(B, T, self.n_layers, 2, D)
        # [B, T, n_layers, 2, D]
        
        # 7. Pool to prompt_len (take first prompt_len tokens or average)
        if T >= self.prompt_len:
            prefix = prefix_raw[:, :self.prompt_len]  # [B, prompt_len, n_layers, 2, D]
        else:
            # Pad if needed
            pad_len = self.prompt_len - T
            prefix = F.pad(prefix_raw, (0, 0, 0, 0, 0, 0, 0, pad_len))
        
        # Rearrange to [n_layers, 2, B, prompt_len, D]
        prefix = prefix.permute(2, 3, 0, 1, 4)
        
        return prefix, trend_feat, seasonal_feat


class DualStreamSemanticMoE(nn.Module):
    """
    Dual-stream Semantic MoE for separate trend and seasonal processing.
    Generates independent prefixes for each stream.
    """
    def __init__(self, d_model, num_experts, prompt_len, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.prompt_len = prompt_len
        self.n_layers = n_layers
        
        # Physical Sensors (shared)
        self.trend_sensor = MultiScaleTrendSensor(d_model)
        self.seasonal_sensor = FrequencySeasonalSensor(d_model)
        
        # Separate MoE for each stream (like SEMPO's domain_EnMoE)
        self.trend_moe = nn.ModuleDict({
            'gate': nn.Linear(d_model, num_experts),
            'experts': nn.Embedding(num_experts, d_model),
            'transform': nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_layers * 2 * d_model)
            )
        })
        
        self.seasonal_moe = nn.ModuleDict({
            'gate': nn.Linear(d_model, num_experts),
            'experts': nn.Embedding(num_experts, d_model),
            'transform': nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_layers * 2 * d_model)
            )
        })
    
    def _moe_forward(self, x, moe_dict):
        """Helper for MoE forward pass."""
        B, T, D = x.shape
        device = x.device
        
        # Gate
        gate_logits = moe_dict['gate'](x)  # [B, T, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Expert combination
        expert_idx = torch.arange(self.num_experts, device=device)
        expert_emb = moe_dict['experts'](expert_idx)  # [num_experts, D]
        
        ctx = torch.einsum('bte,ed->btd', gate_weights, expert_emb)  # [B, T, D]
        
        # Transform to prefix
        prefix = moe_dict['transform'](ctx)  # [B, T, n_layers*2*D]
        
        return prefix
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features
        Returns:
            trend_prefix: [n_layers, 2, B, T, D] 
            seasonal_prefix: [n_layers, 2, B, T, D]
            trend_feat: [B, T, D]
            seasonal_feat: [B, T, D]
        """
        B, T, D = x.shape
        
        # Extract physical features
        trend_feat = self.trend_sensor(x)
        seasonal_feat = self.seasonal_sensor(x)
        
        # Generate prefixes based on physical features
        trend_prefix_raw = self._moe_forward(trend_feat, self.trend_moe)      # [B, T, n_layers*2*D]
        seasonal_prefix_raw = self._moe_forward(seasonal_feat, self.seasonal_moe)  # [B, T, n_layers*2*D]
        
        # Reshape: [B, T, n_layers*2*D] -> [n_layers, 2, B, T, D]
        trend_prefix = trend_prefix_raw.view(B, T, self.n_layers, 2, D).permute(2, 3, 0, 1, 4)
        seasonal_prefix = seasonal_prefix_raw.view(B, T, self.n_layers, 2, D).permute(2, 3, 0, 1, 4)
        
        return trend_prefix, seasonal_prefix, trend_feat, seasonal_feat
