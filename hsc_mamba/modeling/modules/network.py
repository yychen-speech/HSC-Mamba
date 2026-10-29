import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from functools import partial
from einops import rearrange, repeat
from torch import Tensor

# Mamba-1 官方包导入
from mamba_ssm import Mamba as OrigMamba

try:
    # 尝试导入 Mamba 的 Triton 优化版 RMSNorm
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None
    # 如果导入失败，提供一个原生的 PyTorch RMSNorm 等效实现
    class RMSNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

try:
    from speechbrain.lobes.models.dual_path import Dual_Path_Model
except ImportError:
    Dual_Path_Model = nn.Module

try:
    from ...utils.decorators import *
except ImportError:
    pass


# ==========================================
# BiMamba Wrapper (Mamba-1 + 自适应动态门控融合)
# ==========================================
class BiMamba(nn.Module):
    """
    使用两个官方 Mamba 模块构建双向 Mamba。
    这避免了触碰底层 C++ 算子，完美兼容 Mamba-3 的复杂状态空间更新。
    """
    def __init__(self, d_model, d_state=16, expand=2, **kwargs):
        super().__init__()
        # 剥离旧版本特有但在 Mamba-3 中被废弃的参数
        kwargs.pop('d_conv', None)
        kwargs.pop('dt_rank', None)
        kwargs.pop('conv_bias', None)
        kwargs.pop('bimamba_type', None)
        kwargs.pop('if_devide_out', None)
        kwargs.pop('init_layer_scale', None)

        self.d_model = d_model
        
        # 实例化前向和后向的官方 Mamba-3 核心
        self.fwd_mamba = OrigMamba(d_model=d_model, d_state=d_state, expand=expand, **kwargs)
        self.bwd_mamba = OrigMamba(d_model=d_model, d_state=d_state, expand=expand, **kwargs)

    def forward(self, hidden_states, inference_params=None):
        # Mamba 官方 API 的输入维度为 (B, L, D)
        out_fwd = self.fwd_mamba(hidden_states)
        # 对时间步维度 (dim=1) 进行翻转
        out_bwd = self.bwd_mamba(hidden_states.flip([1])).flip([1])
        
        return out_fwd + out_bwd

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.fwd_mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# ==========================================
# PreNormBlock & Mamba Blocks Sequential
# ==========================================

class PreNormBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        if not self.fused_add_norm:
            # 💡 极度硬核的“延迟残差”写法，为大模型优化显存读取
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cls=None,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}

    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(ssm_cls, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = PreNormBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )

    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class MambaBlocksSequential(nn.Module):
    def __init__(self, 
        n_mamba: int,
        bidirectional: bool = False,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        bias: bool = False,
        fused_add_norm: bool = True,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.bidirectional = bidirectional

        ssm_cls = BiMamba if bidirectional else OrigMamba

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        ssm_cfg = {
            "d_state": d_state,
            "expand": expand,
            "d_conv": d_conv,
            "bias": bias
        }

        self.layers = nn.Sequential(
            *[
                create_block(
                    d_model=d_model,
                    ssm_cls=ssm_cls,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(n_mamba)
            ]
        )
        
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_mamba,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def forward(self, x, keep_to=None, inference_params=None):
        hidden_states = x
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

# ==========================================
# Dual Path Model
# ==========================================
class SafeGroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        return super().forward(x)

class Dual_Path_Model_Skip(Dual_Path_Model):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        skip_n_block=0,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model_Skip, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=num_layers,
            norm=norm, 
            K=K,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            linear_layer_after_inter_intra=linear_layer_after_inter_intra,
            use_global_pos_enc=use_global_pos_enc,
        )
        self.skip_n_block = skip_n_block
        
        for layer in self.dual_mdl:
            if hasattr(layer, 'intra_norm'):
                layer.intra_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)
            
            if hasattr(layer, 'inter_norm'):
                layer.inter_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)
                
            if hasattr(layer, 'norm'):
                if isinstance(layer.norm, nn.Module):
                    layer.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)

        self.intra_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-8)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1d(x)
        
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        x, gap = self._Segmentation(x, self.K)
        residual = x

        for i in range(self.num_layers):
            if self.skip_n_block > 0 and \
                i % self.skip_n_block == 0 and i != 0: 
                x = 0.5 * x + 0.5 * residual
            
            x = self.dual_mdl[i](x)

        x = self.prelu(x)
        x = self._over_add(x, gap)
        x = self.end_conv1x1(x)
        
        return x


# ==========================================
# 基础组件 (LayerScale, Masking, GCFN, Attention)
# ==========================================

class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            self.layer_scale = torch.nn.Parameter(torch.ones(input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 2:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 3:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,1,input_size)*Layer_scale_init, requires_grad=True)
        self.layer_scale._no_weight_decay = True
    
    def forward(self, x):
        return x*self.layer_scale

class Masking(torch.nn.Module):
    def __init__(self, input_dim, Activation_mask='Sigmoid', **options):
        super(Masking, self).__init__()
        self.options = options
        if self.options['concat_opt']:
            self.pw_conv = torch.nn.Conv1d(input_dim*2, input_dim, 1, stride=1, padding=0)

        if Activation_mask == 'Sigmoid':
            self.gate_act = torch.nn.Sigmoid()
        elif Activation_mask == 'ReLU':
            self.gate_act = torch.nn.ReLU()

    def forward(self, x, skip):
        if self.options['concat_opt']: 
            y = torch.cat([x, skip], dim=-2)
            y = self.pw_conv(y)
        else:
            y = x
        y = self.gate_act(y) * skip
        return y

class GCFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels*6))
        self.depthwise = torch.nn.Conv1d(in_channels*6, in_channels*6, 3, padding=1, groups=in_channels*6)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels*3, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
        
    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + self.Layer_scale(y)


# ==========================================
# 核心网络块 (GlobalBlock & LocalBlock)
# ==========================================
class MultiScaleDepthwiseConv1d(torch.nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 17, 31], bias=False):
        super().__init__()
        self.channels = channels
        self.num_scales = len(kernel_sizes)
        
        base_dim = channels // self.num_scales
        remainder = channels % self.num_scales
        
        self.split_sizes = []
        for i in range(self.num_scales):
            size = base_dim + 1 if i < remainder else base_dim
            self.split_sizes.append(size)
            
        self.convs = torch.nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            in_ch = self.split_sizes[i]
            self.convs.append(torch.nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                padding=(k - 1) // 2,
                groups=in_ch,
                bias=bias
            ))

    def forward(self, x):
        chunks = torch.split(x, self.split_sizes, dim=1)
        outs = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        return torch.cat(outs, dim=1)


class DepthwiseConv1d(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)

class GlobalBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_mamba: int = 4,
                 bidirectional: bool = True,
                 d_state: int = 16,
                 expand: int = 2,
                 d_conv: int = 4,         
                 fused_add_norm: bool = False,
                 rms_norm: bool = False,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        in_ch = int(in_channels)
        self.n_mamba = int(n_mamba)
        self.mamba = MambaBlocksSequential(
            n_mamba=self.n_mamba,
            bidirectional=bool(bidirectional),
            d_model=in_ch,
            d_state=int(d_state),
            expand=int(expand),
            d_conv=int(d_conv),           
            fused_add_norm=bool(fused_add_norm),
            rms_norm=bool(rms_norm)
        )

    def forward(self, x: torch.Tensor):
        x_tn = x.permute(0, 2, 1).contiguous()    
        out_tn = self.mamba(x_tn)                  
        out_nT = out_tn.permute(0, 2, 1).contiguous()  
        return out_nT


class LocalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        # 默认参数
        kernel_sizes = kwargs.get('kernel_sizes', [3, 17, 31])
        expand = kwargs.get('expand', 2)
        dropout_p = kwargs.get('dropout', 0.1)
        
        # 统一使用 GroupNorm(1) 也就是 LayerNorm(channel_first)，绝不碰 BatchNorm
        self.norm1 = torch.nn.GroupNorm(1, in_channels)
        
        hidden_channels = in_channels * expand
        
        # 1. 投影层：将通道放大 2 倍，为后面的门控分支做准备
        self.conv1 = torch.nn.Conv1d(in_channels, hidden_channels * 2, kernel_size=1)
        
        # 2. 保留你优秀的多尺度卷积（注意：现在它只处理 hidden_channels 的一半）
        self.dconv = MultiScaleDepthwiseConv1d(
            channels=hidden_channels, 
            kernel_sizes=kernel_sizes
        )
        
        # 统一使用 GroupNorm，稳住特征分布
        self.norm2 = torch.nn.GroupNorm(1, hidden_channels)
        
        # 3. 激活与降维
        self.act = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor):
        residual = x
        
        # 1. 归一化与通道膨胀
        x = self.norm1(x)
        x = self.conv1(x) # 此时形状为 [B, hidden_channels * 2, T]
        
        # 2. 核心架构重组：通道劈成两半 (一半做特征提取，一半做门控)
        x_feat, x_gate = x.chunk(2, dim=1) # 两者形状均为 [B, hidden_channels, T]
        
        # 3. 先做多尺度空间特征提取 (在裸特征上抓取，不丢失高频)
        x_feat = self.dconv(x_feat)
        x_feat = self.norm2(x_feat)
        
        # 4. Gated Linear Unit 逻辑：用激活后的 gate 去控制特征
        x = x_feat * self.act(x_gate) 
        
        # 5. 投影还原并加上残差
        x = self.conv2(x)
        x = self.dropout(x)
        
        return residual + x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head: int, in_channels: int, dropout_rate: float, Layer_scale_init=1.0e-5):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
    
    def forward(self, x, pos_k, mask):
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        
        A = torch.matmul(q, k.transpose(-2, -1))
        if pos_k is not None:
            reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1)).transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
            
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
            
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v).transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.Layer_scale(self.dropout(self.linear_out(x)))

class SpkAttention(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate)
        
        # 🚀 [新增] 干扰抵消投影层 (Interference Cancellation Projection)
        # 用于将注意力机制提取到的“其他说话人的特征”映射为需要减去的干扰项
        self.cancel_proj = torch.nn.Linear(in_channels, in_channels)
        
        self.feed_forward = GCFN(in_channels=in_channels, dropout_rate=dropout_rate)
    
    def forward(self, x: torch.Tensor, num_spk: int):
        B, F, T = x.shape
        real_B = B // num_spk
        
        # [Batch, num_spk, F, T] -> Reshape 准备跨说话人注意力
        x_reshaped = x.view(real_B, num_spk, F, T).permute(0, 3, 1, 2).reshape(-1, num_spk, F)
        
        # 获取说话人之间的注意力交互特征
        interaction = self.self_attn(x_reshaped, None, None)
        
        # 🚀 [修改核心] 显式干扰抵消 (Cross-speaker Subtraction)
        # 不再使用 x_reshaped = x_reshaped + interaction
        # 而是预估出干扰成分，并从原特征中硬性减去
        interference = self.cancel_proj(interaction)
        x_reshaped = x_reshaped - interference
        
        # 恢复形状
        x = x_reshaped.view(real_B, T, num_spk, F).permute(0, 2, 3, 1).reshape(B, F, T)
        
        x = x.permute(0, 2, 1)
        x = self.feed_forward(x)
        x = x.permute(0, 2, 1)
        return x
