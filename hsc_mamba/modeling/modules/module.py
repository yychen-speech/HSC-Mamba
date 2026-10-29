import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ...utils.decorators import *
from .cpc_mamba import ConservativePairCoupledMamba
from .network import *


# ==========================================
# FFT-SSM Core Math Functions
# ==========================================
#@torch.compiler.disable
def fft_conv(equation, input, kernel, *args):
    input, kernel = input.float(), kernel.float()
    args = tuple(arg.cfloat() for arg in args)
    n = input.shape[-1]
    
    kernel_f = torch.fft.rfft(kernel, 2 * n)
    input_f = torch.fft.rfft(input, 2 * n)
    output_f = torch.einsum(equation, input_f, kernel_f, *args)
    output = torch.fft.irfft(output_f, 2 * n)
    
    return output[..., :n]

def ssm_basis_kernels(A, B, log_dt, length):
    log_A_real, A_imag = A.T  # (2, num_coeffs)
    lrange = torch.arange(length, device=A.device)
    dt = log_dt.exp()
    
    dtA_real, dtA_imag = -dt * F.softplus(log_A_real), dt * A_imag        
    return (dtA_real[:, None] * lrange).exp() * torch.cos(dtA_imag[:, None] * lrange), B * dt[:, None]

def opt_ssm_forward(input, K, B_hat, C):
    batch, c_in, _ = input.shape
    c_out, coeffs = C.shape
    
    if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
        if c_in * c_out <= coeffs:
            kernel = torch.einsum('dn,nc,nl->dcl', C, B_hat, K)
            return fft_conv('bcl,dcl->bdl', input, kernel)
    else:
        if coeffs <= c_in:
            x = torch.einsum('bcl,nc->bnl', input, B_hat)
            x = fft_conv('bnl,nl->bnl', x, K)
            return torch.einsum('bnl,dn->bdl', x, C)
        
    return fft_conv('bcl,nl,nc,dn->bdl', input, K, B_hat, C)

class SSMLayer(torch.nn.Module):
    def __init__(self, num_coeffs: int, in_channels: int, out_channels: int, repeat: int):
        super().__init__()
        # 兼容处理 opt_einsum 以加速爱因斯坦求和约定
        try:
            from torch.backends import opt_einsum
            if opt_einsum.is_available():
                opt_einsum.strategy = 'optimal'
        except Exception:
            pass 
        
        init_parameter = lambda mat: Parameter(torch.tensor(mat, dtype=torch.float))
        normal_parameter = lambda fan_in, shape: Parameter(torch.randn(*shape) * math.sqrt(2 / fan_in))
        
        A_real, A_imag = 0.5 * np.ones(num_coeffs), math.pi * np.arange(num_coeffs)
        log_A_real = np.log(np.exp(A_real) - 1)  # inv softplus
        B = np.ones(num_coeffs)
        A = np.stack([log_A_real, A_imag], -1)
        log_dt = np.linspace(np.log(0.001), np.log(0.1), repeat)
        
        A = np.tile(A, (repeat, 1))
        B = np.tile(B[:, None], (repeat, in_channels)) / math.sqrt(in_channels)
        log_dt = np.repeat(log_dt, num_coeffs)
            
        self.log_dt = init_parameter(log_dt)
        self.A = init_parameter(A)
        self.B = init_parameter(B)
        self.C = normal_parameter(num_coeffs * repeat, (out_channels, num_coeffs * repeat))
    
    def forward(self, input):
        K, B_hat = ssm_basis_kernels(self.A, self.B, self.log_dt, input.shape[-1])
        return opt_ssm_forward(input, K, B_hat, self.C)

# ==========================================
# 原网络结构定义
# ==========================================
class AudioEncoder(torch.nn.Module):
    """
    [重构版] AudioEncoder: CNN (局部) + FFT-SSM (全局降噪)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool, num_coeffs: int = 16, repeat: int = 8):
        super().__init__()
        # 1. 局部声学特征提取与降采样 (CNN)
        # 将输入波形从长度 32000 压缩到 8000，为后续提取高级特征打好基础
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)
        self.gelu = torch.nn.GELU()
        
        # 2. 全局静态背景降噪过滤 (FFT-SSM)
        # 在 8000 长度上极其高效地进行全局频率扫描，过滤噪声，梳理全局静态特征
        self.ssm = SSMLayer(num_coeffs=num_coeffs, in_channels=out_channels, out_channels=out_channels, repeat=repeat)
        self.norm = torch.nn.GroupNorm(1, out_channels)
    
    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [B, 1, T]
        
        # Phase 1: Local Conv & Downsample
        x = self.conv1d(x)
        x = self.gelu(x)
        
        # Phase 2: Global FFT-SSM Filtering with Residual Connection
        residual = x
        x = self.ssm(x)
        x = self.norm(x)
        x = x + residual # 残差连接保证声学纹理不丢失
        
        return x

    
class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x


class Separator(torch.nn.Module):
    def __init__(
        self,
        num_stages: int,
        enc_stage: dict,
        spk_split_stage: dict,
        simple_fusion: dict,
        dec_stage: dict,
        hierarchical_early_separation: dict = None,
        conservative_pair_coupled_mamba: dict = None,
    ):
        super().__init__()

        class BoundedSplitSharedFusion(torch.nn.Module):
            def __init__(self, lower_bound: float, upper_bound: float, stage_name: str, initial_weight=None):
                super().__init__()
                lower_bound = float(lower_bound)
                upper_bound = float(upper_bound)
                if not 0.0 <= lower_bound < upper_bound <= 1.0:
                    raise ValueError(
                        f"Invalid split-weight bounds for {stage_name}: "
                        f"[{lower_bound}, {upper_bound}]."
                    )
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
                self.stage_name = str(stage_name)
                initial_weight = (lower_bound + upper_bound) / 2 if initial_weight is None else float(initial_weight)
                if not lower_bound < initial_weight < upper_bound:
                    raise ValueError(f'Initial weight for {stage_name} must be inside its bounds.')
                probability = (initial_weight - lower_bound) / (upper_bound - lower_bound)
                self.split_logit = torch.nn.Parameter(torch.tensor(math.log(probability / (1 - probability))))

            def split_weight(self):
                span = self.upper_bound - self.lower_bound
                return self.lower_bound + span * torch.sigmoid(self.split_logit)

            def forward(self, split_output: torch.Tensor, shared_output: torch.Tensor):
                if split_output.shape != shared_output.shape:
                    raise RuntimeError(
                        f"{self.stage_name} split/shared outputs do not match: "
                        f"{tuple(split_output.shape)} vs {tuple(shared_output.shape)}."
                    )
                weight = self.split_weight().to(
                    device=split_output.device,
                    dtype=split_output.dtype,
                )
                return weight * split_output + (1.0 - weight) * shared_output

        class AttentionFusionGate(torch.nn.Module):
            def __init__(self, in_channels: int, stage_depth: int = 0):
                super().__init__()
                self.gate = torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels=2 * in_channels, 
                        out_channels=2 * in_channels, 
                        kernel_size=3, 
                        padding=1, 
                        groups=2 * in_channels
                    ),
                    torch.nn.Conv1d(
                        in_channels=2 * in_channels, 
                        out_channels=in_channels, 
                        kernel_size=1
                    ),
                    torch.nn.SiLU(),  
                    torch.nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=in_channels, 
                        kernel_size=1
                    ),
                    torch.nn.Sigmoid()
                )
                self.res_conv = torch.nn.Conv1d(
                    in_channels=2 * in_channels, 
                    out_channels=in_channels, 
                    kernel_size=1
                )
                
                self._init_weights(stage_depth)
        
            def _init_weights(self, stage_depth: int):
                if stage_depth <= 1:
                    init_bias = torch.tensor([0.7] * self.gate[-2].out_channels)  
                else:
                    init_bias = torch.tensor([0.4] * self.gate[-2].out_channels)  
                torch.nn.init.constant_(self.gate[-2].bias, init_bias.mean())
        
            def forward(self, dec_feat: torch.Tensor, enc_skip: torch.Tensor) -> torch.Tensor:
                concat_feat = torch.cat([dec_feat, enc_skip], dim=1)  # [B, 2F, T]
                skip_weight = self.gate(concat_feat)  # [B, F, T]
                fused_feat = skip_weight * enc_skip + (1 - skip_weight) * dec_feat
                fused_feat = fused_feat + self.res_conv(concat_feat)
                return fused_feat
        
        class SepEncStage(torch.nn.Module):
            def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
                super().__init__()
                                
                class DownConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        super().__init__()
                        self.down_conv = torch.nn.Conv1d(
                            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2, padding=(samp_kernel_size-1)//2, groups=in_channels)
                        self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                        self.gelu = torch.nn.GELU()
                    
                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = self.down_conv(x)
                        x = self.BN(x)
                        x = self.gelu(x)
                        x = x.permute([0, 2, 1])
                        return x
                
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                
                self.downconv = DownConvLayer(**down_conv_layer) if down_conv == True else None
                
            def forward(self, x: torch.Tensor):
                x = self.g_block_1(x)
                x = self.l_block_1(x)
                x = self.g_block_2(x)
                x = self.l_block_2(x)
                
                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                return x, skip
        
        class SpkSplitStage(torch.nn.Module):
            def __init__(
                self,
                in_channels: int,
                num_spks: int,
                num_scales: int,
                adapter_rank: int = None,
                adapter_gamma_init: float = 1.0e-3,
            ):
                super().__init__()
                in_channels = int(in_channels)
                num_spks = int(num_spks)
                self.linear = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, 4*in_channels*num_spks, kernel_size=1),
                    torch.nn.GLU(dim=-2),
                    torch.nn.Conv1d(2*in_channels*num_spks, in_channels*num_spks, kernel_size=1))
                self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)
                self.adapters = torch.nn.ModuleList([])
                adapter_rank = max(1, int(adapter_rank or max(8, in_channels // 8)))
                for _ in range(num_scales):
                    adapter = torch.nn.Sequential(
                        torch.nn.Conv1d(in_channels, adapter_rank, kernel_size=1),
                        torch.nn.SiLU(),
                        torch.nn.Conv1d(
                            adapter_rank,
                            in_channels * num_spks,
                            kernel_size=1,
                        ),
                    )
                    torch.nn.init.zeros_(adapter[-1].weight)
                    torch.nn.init.zeros_(adapter[-1].bias)
                    self.adapters.append(adapter)
                self.adapter_gammas = torch.nn.Parameter(
                    torch.full((num_scales,), float(adapter_gamma_init))
                )
                self.num_spks = num_spks
                
            def forward(self, x: torch.Tensor, scale_idx: int):
                adapter = self.adapters[scale_idx](x)
                gamma = self.adapter_gammas[scale_idx].to(dtype=x.dtype)
                x = self.linear(x) + gamma * adapter
                B, _, T = x.shape
                x = x.view(B*self.num_spks,-1, T).contiguous()
                x = self.norm(x)
                return x
        
        class SepDecStage(torch.nn.Module):
            def __init__(self, num_spks: int, global_blocks: dict, local_blocks: dict, spk_attention: dict):
                super().__init__()
                
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.spk_attn_1 = SpkAttention(**spk_attention)
                
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                self.spk_attn_2 = SpkAttention(**spk_attention)
                
                self.g_block_3 = GlobalBlock(**global_blocks)
                self.l_block_3 = LocalBlock(**local_blocks)
                self.spk_attn_3 = SpkAttention(**spk_attention)
                
                self.num_spk = num_spks
            
            def forward(self, x: torch.Tensor):
                x = self.g_block_1(x)
                x = self.l_block_1(x)
                x = self.spk_attn_1(x, self.num_spk)
                x = self.g_block_2(x)
                x = self.l_block_2(x)
                x = self.spk_attn_2(x, self.num_spk)
                x = self.g_block_3(x)
                x = self.l_block_3(x)
                x = self.spk_attn_3(x, self.num_spk)
                skip = x
                return x, skip
        
        self.num_stages = num_stages
        self.num_spks = int(spk_split_stage["num_spks"])
        hses_cfg = hierarchical_early_separation or {}
        self.hses_enabled = bool(hses_cfg.get("enabled", False))
        self.hses_policy = str(hses_cfg.get("policy", "disabled"))
        self.hses_split_scales = (
            self._resolve_hses_split_scales(self.hses_policy)
            if self.hses_enabled
            else {f"skip{i}" for i in range(self.num_stages)} | {"bottleneck"}
        )
        
        # Temporal Contracting Part
        self.enc_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.enc_stages.append(SepEncStage(**enc_stage, down_conv=True))
        
        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)

        self.spk_split = SpkSplitStage(
            **spk_split_stage,
            num_scales=self.num_stages + 1,
        )

        cpc_cfg = dict(conservative_pair_coupled_mamba or {})
        cpc_enabled = bool(cpc_cfg.pop("enabled", False))
        cpc_locations = list(
            cpc_cfg.pop("apply_locations", ["after_bottleneck_split"])
        )
        unknown_locations = set(cpc_locations) - {"after_bottleneck_split"}
        if unknown_locations:
            raise ValueError(
                f"Unsupported CPC-Mamba locations: {sorted(unknown_locations)}"
            )
        with torch.random.fork_rng(devices=[]):
            self.cpc_mamba = (
                ConservativePairCoupledMamba(
                    channels=int(simple_fusion["out_channels"]),
                    num_spks=self.num_spks,
                    **cpc_cfg,
                )
                if cpc_enabled and "after_bottleneck_split" in cpc_locations
                else None
            )
        
        # Temporal Expanding Part
        self.simple_fusion = torch.nn.ModuleList([])
        for stage_idx in range(self.num_stages):
            self.simple_fusion.append(AttentionFusionGate(
                in_channels=simple_fusion['out_channels'],
                stage_depth=stage_idx,
            ))

        split_weight_bounds = hses_cfg.get("split_weight_bounds")
        self.hses_dual_path = self.hses_enabled and split_weight_bounds is not None
        self.split_shared_fusions = torch.nn.ModuleList([])
        if self.hses_dual_path:
            self.hses_split_scales = (
                {f"skip{i}" for i in range(self.num_stages)} | {"bottleneck"}
            )
            for decoder_idx in range(self.num_stages):
                stage_name = self._decoder_stage_name(decoder_idx)
                bounds = split_weight_bounds.get(stage_name)
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(
                        f"split_weight_bounds.{stage_name} must be [lower, upper]."
                    )
                self.split_shared_fusions.append(
                    BoundedSplitSharedFusion(bounds[0], bounds[1], stage_name,
                        hses_cfg.get("split_weight_init", {}).get(stage_name))
                )

        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.dec_stages.append(SepDecStage(**dec_stage))

    def _resolve_hses_split_scales(self, policy: str):
        aliases = {
            "bottleneckonly": {"bottleneck"},
            "bottleneckonly_sharedallskips": {"bottleneck"},
            "hses": {"skip3", "bottleneck"},
            "deepsplit_shallowshared": {"skip2", "skip3", "bottleneck"},
        }
        key = str(policy).strip().lower()
        if key not in aliases:
            raise ValueError(
                "Unsupported hierarchical early separation policy "
                f"'{policy}'."
            )
        return aliases[key]

    def _scale_name(self, scale_idx: int):
        if int(scale_idx) == self.num_stages:
            return "bottleneck"
        return f"skip{int(scale_idx)}"

    def _decoder_stage_name(self, decoder_idx: int):
        return f"D{self.num_stages - int(decoder_idx)}"

    def _hses_should_split(self, scale_idx: int):
        if not self.hses_enabled:
            return True
        return self._scale_name(scale_idx) in self.hses_split_scales

    def get_split_weights(self):
        if not self.hses_dual_path:
            return {}
        return {
            fusion.stage_name: fusion.split_weight()
            for fusion in self.split_shared_fusions
        }

    def set_epoch(self, epoch: int):
        if self.cpc_mamba is not None and hasattr(self.cpc_mamba, "set_epoch"):
            self.cpc_mamba.set_epoch(epoch)

    def _expand_shared_skip(
        self,
        dec_feat: torch.Tensor,
        shared_skip: torch.Tensor,
    ):
        batch = dec_feat.size(0) // self.num_spks
        if dec_feat.size(0) % self.num_spks != 0 or shared_skip.size(0) != batch:
            raise RuntimeError(
                "Shared skip and decoder batches are inconsistent: "
                f"skip={shared_skip.size(0)}, decoder={dec_feat.size(0)}, "
                f"num_spks={self.num_spks}."
            )
        if shared_skip.size(1) != dec_feat.size(1):
            raise RuntimeError(
                f"Shared skip channels {shared_skip.size(1)} do not match "
                f"decoder channels {dec_feat.size(1)}."
            )
        if shared_skip.shape[-1] != dec_feat.shape[-1]:
            shared_skip = F.interpolate(
                shared_skip,
                size=dec_feat.shape[-1],
                mode="linear",
                align_corners=False,
            )

        return (
            shared_skip.unsqueeze(1)
            .expand(batch, self.num_spks, shared_skip.size(1), shared_skip.size(-1))
            .reshape(batch * self.num_spks, shared_skip.size(1), shared_skip.size(-1))
            .contiguous()
        )

    def _align_split_skip(
        self,
        dec_feat: torch.Tensor,
        split_skip: torch.Tensor,
    ):
        if split_skip.size(0) != dec_feat.size(0) or split_skip.size(1) != dec_feat.size(1):
            raise RuntimeError(
                "Split skip and decoder shapes are inconsistent: "
                f"split={tuple(split_skip.shape)}, decoder={tuple(dec_feat.shape)}."
            )
        if split_skip.shape[-1] != dec_feat.shape[-1]:
            split_skip = F.interpolate(
                split_skip,
                size=dec_feat.shape[-1],
                mode="linear",
                align_corners=False,
            )
        return split_skip

    def _fuse_dual_path(
        self,
        dec_feat: torch.Tensor,
        shared_skip: torch.Tensor,
        split_skip: torch.Tensor,
        decoder_idx: int,
    ):
        shared_expanded = self._expand_shared_skip(dec_feat, shared_skip)
        split_skip = self._align_split_skip(dec_feat, split_skip)
        sfg = self.simple_fusion[decoder_idx]
        split_output = sfg(dec_feat, split_skip)
        shared_output = sfg(dec_feat, shared_expanded)
        return self.split_shared_fusions[decoder_idx](
            split_output,
            shared_output,
        )

    def _prepare_decoder_skip(
        self,
        dec_feat: torch.Tensor,
        skip_feat,
        encoder_skip_idx: int,
        decoder_idx: int,
    ):
        if self._hses_should_split(encoder_skip_idx):
            return skip_feat
        return self._expand_shared_skip(dec_feat, skip_feat)
    
    def forward(self, input: torch.Tensor):
        x, _ = self.pad_signal(input)
        skip = []
        for idx in range(self.num_stages):
            x, shared_skip = self.enc_stages[idx](x)
            if self.hses_dual_path:
                split_skip = self.spk_split(shared_skip, idx)
                skip.append((shared_skip, split_skip))
            else:
                skip_value = shared_skip
                if self._hses_should_split(idx):
                    skip_value = self.spk_split(shared_skip, idx)
                skip.append(skip_value)
        x, _ = self.bottleneck_G(x)
        x = self.spk_split(x, self.num_stages) # B, 2F, T
        if self.cpc_mamba is not None:
            x = self.cpc_mamba(x)
        
        each_stage_outputs = []
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            skip_length = (
                skip[idx_en][0].shape[-1]
                if self.hses_dual_path
                else skip[idx_en].shape[-1]
            )
            x = F.interpolate(x, size=skip_length, mode="nearest")
            if self.hses_dual_path:
                shared_skip, split_skip = skip[idx_en]
                x = self._fuse_dual_path(
                    x,
                    shared_skip,
                    split_skip,
                    idx,
                )
            else:
                skip_for_fusion = self._prepare_decoder_skip(
                    x,
                    skip[idx_en],
                    idx_en,
                    idx,
                )
                x = self.simple_fusion[idx](x, skip_for_fusion)
            x, _ = self.dec_stages[idx](x)
        
        last_stage_output = x 
        return last_stage_output, each_stage_outputs
    
    def pad_signal(self, input: torch.Tensor):
        if input.dim() == 1: input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]: raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        batch_size = input.size(0)  
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe//L + 1)*L
        rest = 0 if nframe%L == 0 else padded_len - nframe
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, ndim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_spks: int, masking: bool = False):
        super().__init__()
        self.masking = masking
        self.spe_block = Masking(in_channels, Activation_mask="ReLU", concat_opt=None)
        self.num_spks = num_spks
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4*out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2*out_channels, in_channels))
            
    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[...,:input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks
        
        if self.masking:
            input = input.expand(self.num_spks, B, N, L).transpose(0,1).contiguous()
            input = input.view(B*self.num_spks, N, L)
            x = self.spe_block(x, input)
        
        x = x.view(B, self.num_spks, N, L)
        x = x.transpose(0, 1)
        return x


class AudioDecoder(torch.nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        if x.dim() not in [2, 3]: raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x
        
