"""
PrismGTS.py

Core Architecture:
1. Patch Embedding → Tokenize time series into patches
2. DualStreamSemanticMoE → Extract physical features + generate prefixes
3. Dual-Stream Transformer Encoder → Trend-guided and Seasonal-guided encoding
4. Split Decoder  → Independent decoders for trend and seasonal
5. CrossStreamAttention  → Light interaction before prediction
6. Additive Output → X̂ = X̂_t + X̂_s, Y = Y_t + Y_s

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.PrismGTS_EncDec import TSTEncoder, TowerEncoder, PredictionHead, PretrainHead
from layers.SemanticMoE import DualStreamSemanticMoE, MultiScaleTrendSensor, FrequencySeasonalSensor
from layers.PhysicalConstraints import PhysicalConstraintLossV3
from layers.pos_encoding import positional_encoding


# ============================================================================
# CrossStreamAttention
# ============================================================================

class CrossStreamAttention(nn.Module):
    """
    Light interaction layer between trend and seasonal streams.
    
    Allows trend to query seasonal information and vice versa,
    without fully merging them (preserving disentanglement).
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Trend queries Seasonal
        self.trend_to_seasonal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Seasonal queries Trend
        self.seasonal_to_trend = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm_trend = nn.LayerNorm(d_model)
        self.norm_seasonal = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_trend, h_seasonal):
        """
        Args:
            h_trend: [B, C, D, P] trend representation (D=d_model, P=patch_num)
            h_seasonal: [B, C, D, P] seasonal representation
        Returns:
            h_trend_new: [B, C, D, P] trend with seasonal info
            h_seasonal_new: [B, C, D, P] seasonal with trend info
        """
        B, C, D, P = h_trend.shape
        
        # Permute to [B, C, P, D] then reshape for attention: [B*C, P, D]
        h_t = h_trend.permute(0, 1, 3, 2).reshape(B * C, P, D)
        h_s = h_seasonal.permute(0, 1, 3, 2).reshape(B * C, P, D)
        
        # Cross attention: trend queries seasonal
        h_t_cross, _ = self.seasonal_to_trend(query=h_t, key=h_s, value=h_s)
        h_t_new = self.norm_trend(h_t + self.dropout(h_t_cross))
        
        # Cross attention: seasonal queries trend
        h_s_cross, _ = self.trend_to_seasonal(query=h_s, key=h_t, value=h_t)
        h_s_new = self.norm_seasonal(h_s + self.dropout(h_s_cross))
        
        # Reshape back: [B*C, P, D] -> [B, C, P, D] -> [B, C, D, P]
        h_trend_new = h_t_new.reshape(B, C, P, D).permute(0, 1, 3, 2)
        h_seasonal_new = h_s_new.reshape(B, C, P, D).permute(0, 1, 3, 2)
        
        return h_trend_new, h_seasonal_new


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # ============== Hyperparameters ==============
        n_heads: int = 16
        d_ff: int = 256
        head_dropout: float = 0.2
        dropout: float = 0.1
        attn_dropout: float = 0.0
        norm: str = 'RMSNorm'
        act: str = 'silu'
        res_attention: bool = True
        pre_norm: bool = True
        pe: str = 'zeros'
        learn_pe: bool = True
        
        # ============== Config ==============
        self.n_vars = configs.c_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.head_type = configs.head_type
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.horizon_lengths = getattr(configs, 'horizon_lengths', [1, 96, 192, 336, 720])
        self.num_experts = getattr(configs, 'num_experts', 8)
        self.prompt_len = getattr(configs, 'prompt_len', 8)
        
        # ============== Patching ==============
        self.patch_num = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len + self.stride * (self.patch_num - 1)
        self.s_begin = self.seq_len - tgt_len
        
        # ============== Normalization ==============
        self.revin_layer_x = RevIN(self.n_vars, affine=True)
        
        # ============== Projection ==============
        self.projection_x = nn.Linear(self.seq_len, self.seq_len)
        
        # ============== Patch Embedding ==============
        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        
        # ============== Positional Encoding ==============
        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num * self.n_vars, self.d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # ============== Semantic MoE (Core Innovation) ==============
        self.semantic_moe = DualStreamSemanticMoE(
            d_model=self.d_model,
            num_experts=self.num_experts,
            prompt_len=self.prompt_len,
            n_layers=self.e_layers,
            dropout=head_dropout
        )
        
        # ============== Dual-Stream Encoder ==============
        self.trend_encoder = TSTEncoder(
            self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
            activation=act, res_attention=res_attention, n_layers=self.e_layers
        )
        
        self.seasonal_encoder = TSTEncoder(
            self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
            activation=act, res_attention=res_attention, n_layers=self.e_layers
        )
        
        # ============== Split Decoder ==============
        # Trend Decoder
        self.trend_decoder = TowerEncoder(
            self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
            activation=act, res_attention=res_attention, n_layers=self.d_layers
        )
        
        # Seasonal Decoder
        self.seasonal_decoder = TowerEncoder(
            self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
            activation=act, res_attention=res_attention, n_layers=self.d_layers
        )
        
        # ============== Split Reconstruction Heads ==============
        self.trend_recon_head = PretrainHead(self.d_model, self.patch_len, head_dropout)
        self.seasonal_recon_head = PretrainHead(self.d_model, self.patch_len, head_dropout)
        
        # ============== CrossStreamAttention ==============
        self.cross_attention = CrossStreamAttention(self.d_model, n_heads=4, dropout=dropout)
        
        # ============== Dual-Stream Prediction Heads ==============
        self.trend_heads = nn.ModuleList([
            PredictionHead(False, self.n_vars, self.d_model, self.patch_num, h, head_dropout)
            for h in self.horizon_lengths
        ])
        
        self.seasonal_heads = nn.ModuleList([
            PredictionHead(False, self.n_vars, self.d_model, self.patch_num, h, head_dropout)
            for h in self.horizon_lengths
        ])
        
        # ============== Reconstruction Decoders for Physical Loss ==============
        self.trend_recon_decoder = nn.Linear(self.d_model, 1)
        self.seasonal_recon_decoder = nn.Linear(self.d_model, 1)
        
        # ============== Print Stats ==============
        self._print_param_stats()
    
    def _print_param_stats(self):
        """Print parameter statistics with Stage1/Stage2 status."""
        modules = {
            'revin_layer_x': (self.revin_layer_x, '🔥 Train', '❄️ Frozen'),
            'patch_embed': (self.patch_embed, '🔥 Train', '❄️ Frozen'),
            'semantic_moe': (self.semantic_moe, '❄️ Frozen', '🔥 Train'),  # V3: MoE frozen in Stage 1 like SEMPO
            'trend_encoder': (self.trend_encoder, '🔥 Train', '❄️ Frozen'),
            'seasonal_encoder': (self.seasonal_encoder, '🔥 Train', '❄️ Frozen'),
            'trend_decoder': (self.trend_decoder, '🔥 Train', '❄️ Frozen'),
            'seasonal_decoder': (self.seasonal_decoder, '🔥 Train', '❄️ Frozen'),
            'trend_recon_head': (self.trend_recon_head, '🔥 Train', '❄️ Frozen'),
            'seasonal_recon_head': (self.seasonal_recon_head, '🔥 Train', '❄️ Frozen'),
            'cross_attention': (self.cross_attention, '⚫ Unused', '🔥 Train'),
            'trend_heads': (self.trend_heads, '⚫ Unused', '🔥 Train'),
            'seasonal_heads': (self.seasonal_heads, '⚫ Unused', '🔥 Train'),
        }
        
        print("\n" + "=" * 75)
        print("PrismGTS Parameter Statistics")
        print("=" * 75)
        print(f"{'Module':<22} {'Params':>10}   {'Stage1':^12} {'Stage2':^12}")
        print("-" * 75)
        
        total = 0
        stage1_total = 0
        stage2_total = 0
        for name, (module, s1_status, s2_status) in modules.items():
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:<22} {params:>10,}   {s1_status:^12} {s2_status:^12}")
            total += params
            if s1_status == '🔥 Train':
                stage1_total += params
            if s2_status == '🔥 Train':
                stage2_total += params
        
        # print("-" * 75)
        # print(f"{'TOTAL':<22} {total:>10,}   {stage1_total:>10,}   {stage2_total:>10,}")
        print("=" * 75 + "\n")
    def frequency_mask(self, x, mask_ratio=0.3):
        """
        Randomly mask frequency components to force structural learning.
        Args:
            x: [B, L, C] input time series
            mask_ratio: float, ratio of frequencies to mask (0.3 = 30% masked)
        Returns:
            x_masked: [B, L, C] masked time series
        """
        if mask_ratio <= 0:
            return x
            
        B, L, C = x.shape
        x_fft = torch.fft.rfft(x, dim=1)  # [B, F, C]
        freq_len = x_fft.shape[1]
        
        # Create mask (1 = keep, 0 = mask)
        # We want to mask `mask_ratio` of frequencies
        mask = torch.rand(B, freq_len, C, device=x.device) > mask_ratio
        mask[:, 0, :] = 1  # Always keep DC component (Trend bias)
        
        # Apply mask
        x_fft_masked = x_fft * mask
        
        # iFFT back to time domain
        x_masked = torch.fft.irfft(x_fft_masked, n=L, dim=1)
        
        return x_masked

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc: [B, L, C] input time series
        Returns:
            y_list: list of [B, H, C] predictions for each horizon
            x_recon: [B, L', C] total reconstruction (X̂_t + X̂_s)
            x_recon_trend: [B, L', C] trend reconstruction
            x_recon_seasonal: [B, L', C] seasonal reconstruction
            z_trend: [B*C, P, D] trend features (for physical loss)
            z_seasonal: [B*C, P, D] seasonal features (for physical loss)
        """
        B = x_enc.shape[0]
        
        # ============== 1. Normalize ==============
        x = self.revin_layer_x(x_enc, 'norm')  # [B, L, C]

        # ============== Frequency Masking (Stage 1 Only) ==============
        if self.training and self.head_type == 'pretrain':
             x = self.frequency_mask(x, mask_ratio=0.3)
        
        # ============== 2. Projection ==============
        x = self.projection_x(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, C]
        
        # ============== 3. Patching ==============
        x = x[:, self.s_begin:, :]  # Align to patch boundaries
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, patch_num, C, patch_len]
        x = x.permute(0, 1, 3, 2)  # [B, patch_num, patch_len, C]
        patch_num, C = x.shape[1], x.shape[3]
        
        # ============== 4. Patch Embedding ==============
        x = self.patch_embed(x.permute(0, 1, 3, 2))  # [B, patch_num, C, d_model]
        x = x.permute(0, 2, 1, 3)  # [B, C, patch_num, d_model]
        
        # Reshape for attention: [B*C, patch_num, d_model]
        u = x.reshape(B * C, patch_num, self.d_model)
        u = self.dropout1(u + self.W_pos[:patch_num])
        
        # ============== 5. Semantic MoE: Extract Physical Features + Prefixes ==============
        trend_prefix, seasonal_prefix, trend_feat, seasonal_feat = self.semantic_moe(u)
        # trend_prefix: [e_layers, 2, B*C, T, D]
        # trend_feat: [B*C, T, D]
        
        # ============== 6. Dual-Stream Encoding ==============
        h_trend = self.trend_encoder(u, trend_prefix)  # [B*C, patch_num, d_model]
        h_seasonal = self.seasonal_encoder(u, seasonal_prefix)  # [B*C, patch_num, d_model]
        
        # Reshape: [B, C, patch_num, d_model]
        h_trend = h_trend.reshape(B, C, patch_num, self.d_model)
        h_seasonal = h_seasonal.reshape(B, C, patch_num, self.d_model)
        
        # ============== 7. Split Decoder ==============
        # [B, C, d_model, patch_num] for decoder
        h_trend_dec = h_trend.permute(0, 1, 3, 2)
        h_seasonal_dec = h_seasonal.permute(0, 1, 3, 2)
        
        # Independent decoding
        h_trend_decoded = self.trend_decoder(h_trend_dec)  # [B, C, d_model, patch_num]
        h_seasonal_decoded = self.seasonal_decoder(h_seasonal_dec)  # [B, C, d_model, patch_num]
        
        # ============== 8. Split Reconstruction ==============
        # Trend reconstruction
        x_recon_trend = self.trend_recon_head(h_trend_decoded)  # [B, patch_num, C, patch_len]
        x_recon_trend = x_recon_trend.reshape(B, patch_num * self.patch_len, C)
        x_recon_trend = self.revin_layer_x(x_recon_trend, 'denorm')
        
        # Seasonal reconstruction
        x_recon_seasonal = self.seasonal_recon_head(h_seasonal_decoded)  # [B, patch_num, C, patch_len]
        x_recon_seasonal = x_recon_seasonal.reshape(B, patch_num * self.patch_len, C)
        x_recon_seasonal = self.revin_layer_x(x_recon_seasonal, 'denorm')
        
        # Total reconstruction: X̂ = X̂_t + X̂_s
        x_recon = x_recon_trend + x_recon_seasonal
        
        # ============== 9. CrossStreamAttention for Prediction ==============
        h_trend_cross, h_seasonal_cross = self.cross_attention(h_trend_dec, h_seasonal_dec)
        
        # ============== 10. Dual-Stream Prediction (Additive) ==============
        y_list = []
        for i, (t_head, s_head) in enumerate(zip(self.trend_heads, self.seasonal_heads)):
            y_trend = t_head(h_trend_cross)      # [B, H, C]
            y_seasonal = s_head(h_seasonal_cross) # [B, H, C]
            y = y_trend + y_seasonal              # Additive!
            y = self.revin_layer_x(y, 'denorm')
            y_list.append(y)
        
        return y_list, x_recon, x_recon_trend, x_recon_seasonal, trend_feat, seasonal_feat
    
    def freeze_moe_stage1(self):
        """
        Stage 1: Freeze modules not used in pretraining.
        - semantic_moe: trained in Stage 2 only
        - cross_attention: used in Stage 2 prediction
        - trend_heads, seasonal_heads: prediction heads for Stage 2
        """
        print("\n[Stage 1] Freezing unused modules (MoE, CrossAttention, PredictionHeads)...")
        
        # Freeze MoE
        for param in self.semantic_moe.parameters():
            param.requires_grad = False
        
        # Freeze CrossAttention (Stage 2 only)
        for param in self.cross_attention.parameters():
            param.requires_grad = False
        
        # Freeze Prediction Heads (Stage 2 only)
        for param in self.trend_heads.parameters():
            param.requires_grad = False
        for param in self.seasonal_heads.parameters():
            param.requires_grad = False
    
    def freeze_backbone(self):
        """
        Stage 2: Freeze backbone, keep MoE, CrossAttention and prediction heads trainable.
        """
        print("\n" + "=" * 60)
        print("Freezing backbone for Stage 2...")
        print("=" * 60)
        
        # Freeze components
        for param in self.revin_layer_x.parameters():
            param.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.trend_encoder.parameters():
            param.requires_grad = False
        for param in self.seasonal_encoder.parameters():
            param.requires_grad = False
        for param in self.trend_decoder.parameters():
            param.requires_grad = False
        for param in self.seasonal_decoder.parameters():
            param.requires_grad = False
        for param in self.trend_recon_head.parameters():
            param.requires_grad = False
        for param in self.seasonal_recon_head.parameters():
            param.requires_grad = False
        for param in self.projection_x.parameters():
            param.requires_grad = False
        
        # Keep trainable: semantic_moe, cross_attention, trend_heads, seasonal_heads
        
        self._print_trainable_stats()
    
    def _print_trainable_stats(self):
        """Print trainable parameter statistics after freezing."""
        modules = {
            'revin_layer_x': self.revin_layer_x,
            'patch_embed': self.patch_embed,
            'semantic_moe': self.semantic_moe,
            'trend_encoder': self.trend_encoder,
            'seasonal_encoder': self.seasonal_encoder,
            'trend_decoder': self.trend_decoder,
            'seasonal_decoder': self.seasonal_decoder,
            'trend_recon_head': self.trend_recon_head,
            'seasonal_recon_head': self.seasonal_recon_head,
            'cross_attention': self.cross_attention,
            'trend_heads': self.trend_heads,
            'seasonal_heads': self.seasonal_heads,
        }
        
        print(f"\n📊 Parameter Statistics by Module (Stage 2):")
        print("-" * 60)
        print(f"{'Module':<22} {'Total':>10} {'Trainable':>10} {'Status':<12}")
        print("-" * 60)
        
        total = 0
        trainable_total = 0
        for name, module in modules.items():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            status = '🔥 Train' if trainable > 0 else '❄️ Frozen'
            print(f"{name:<22} {params:>10,} {trainable:>10,} {status:<12}")
            total += params
            trainable_total += trainable
        
        # print("-" * 60)
        # print(f"{'TOTAL':<22} {total:>10,} {trainable_total:>10,}")
        print("=" * 60 + "\n")
