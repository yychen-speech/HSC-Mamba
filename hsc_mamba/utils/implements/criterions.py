import torch
import numpy as np

from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field, fields
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from ..decorators import *
from mir_eval.separation import bss_eval_sources


# ==========================================
# Utility functions
# ==========================================
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)


def _valid_lengths(input_sizes, max_length, device):
    return torch.as_tensor(input_sizes, device=device).long().clamp(
        min=1,
        max=int(max_length),
    )


def _length_mask(lengths, max_length, dtype):
    positions = torch.arange(max_length, device=lengths.device)
    return (positions.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=dtype)


def _masked_zero_mean(signal, mask):
    valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (signal * mask).sum(dim=-1, keepdim=True) / valid_count
    return (signal - mean) * mask


def _masked_si_snr(est, tgt, mask, scale_inv, eps):
    est_zm = _masked_zero_mean(est, mask)
    tgt_zm = _masked_zero_mean(tgt, mask)
    if scale_inv:
        dot = torch.sum(est_zm * tgt_zm, dim=-1, keepdim=True)
        norm = torch.sum(tgt_zm ** 2, dim=-1, keepdim=True)
        tgt_scaled = dot / (norm + eps) * tgt_zm
    else:
        tgt_scaled = tgt_zm
    return 20 * torch.log10(
        eps + l2norm(tgt_scaled) / (l2norm(est_zm - tgt_scaled) + eps)
    )

# ==========================================
# Original Base Layers
# ==========================================
@dataclass(slots=True)
class STFTBase(torch.nn.Module):
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    K: torch.nn.Parameter = field(init=False)
    num_bins: int = field(init=False)

    def __post_init__(self):
        super(STFTBase, self).__init__() 
        K = self._init_kernel(self.frame_length, self.frame_shift)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(self.device)
        self.num_bins = self.K.shape[0] // 2
    
    def _init_kernel(self, frame_len, frame_hop):
        N = frame_len
        if self.window == 'hann':
            W = torch.hann_window(frame_len)
        if N//4 == frame_hop:
            const = (2/3)**0.5       
            W = const*W
        elif N//2 == frame_hop:
            W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5
        
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W 
        K = torch.reshape(K, (N + 2, 1, frame_len)) 
        return K

    def extra_repr(self):
        return (f"window={self.window}, stride={self.frame_shift}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}")

@logger_wraps()
@dataclass(slots=True)
class STFT(STFTBase):
    def forward(self, x, cplx=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"{self.__name__} expect 2D/3D tensor, but got {x.dim()}D signal")
            
        # 🌟 [动态挂载卷积核，兼容任意显卡/CPU]
        if self.K.device != x.device:
            self.K = self.K.to(x.device)
            
        N_frame = ceil(x.shape[-1] / self.frame_shift)
        len_padded = N_frame * self.frame_shift
        if x.dim() == 2:
            x = torch.cat((x, torch.zeros(x.shape[0], len_padded-x.shape[-1], device=x.device)), dim=-1)
            x = torch.unsqueeze(x, 1)
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift, padding=0)
            r, i = torch.chunk(c, 2, dim=1)
        else:        
            x = torch.cat((x, torch.zeros(x.shape[0], x.shape[1], len_padded-x.shape[-1], device=x.device)), dim=-1)
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift, padding=0)
            c = c.reshape(N, C, -1, c.shape[-1])
            r, i = torch.chunk(c, 2, dim=2)

        if cplx:
            return r, i
        m = (r**2 + i**2 + 1.0e-10)**0.5
        p = torch.atan2(i, r)
        return m, p

# ==========================================
# Multi-Resolution STFT Loss
# ==========================================
class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, device, fft_sizes=[256, 512, 1024], hop_sizes=[40, 80, 160], win_lengths=[160, 320, 640]):
        super().__init__()
        self.device = device
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self._window_cache = {}

    def stft(self, x, fft_size, hop_size, win_length):
        n_fft = max(fft_size, win_length)
        window_key = (int(win_length), x.device, torch.get_default_dtype())
        window = self._window_cache.get(window_key)
        if window is None:
            window = torch.hann_window(
                win_length,
                device=x.device,
                dtype=torch.get_default_dtype(),
            )
            self._window_cache[window_key] = window
        x_stft = torch.stft(x, n_fft, hop_size, win_length, window=window, return_complex=True)
        x_mag = torch.abs(x_stft).clamp(min=1e-7)
        return x_mag

    def spectral_convergence_loss(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-9)

    def log_stft_magnitude_loss(self, x_mag, y_mag):
        return torch.nn.functional.l1_loss(torch.log(x_mag), torch.log(y_mag))

    def forward(self, x, y):
        loss = 0.0
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_mag = self.stft(x, fs, hs, wl)
            y_mag = self.stft(y, fs, hs, wl)
            
            sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
            
            loss += sc_loss + mag_loss
        
        return loss / len(self.fft_sizes)


# ==========================================
# Criterions (Loss Functions)
# ==========================================

@logger_wraps()
@dataclass(slots=True)
class PIT_HybridLoss:
    device: torch.device
    num_spks: int
    scale_inv: bool
    stft_weight: float = 0.1
    mr_stft_loss: MultiResolutionSTFTLoss = field(init=False)

    def __post_init__(self):
        self.mr_stft_loss = MultiResolutionSTFTLoss(
            self.device, 
            fft_sizes=[256, 1024, 2048], 
            hop_sizes=[50, 120, 240], 
            win_lengths=[240, 600, 1200]
        )

    def sisnr_loss_batch(self, est, tgt, mask, eps=1.0e-8):
        pair_loss = -_masked_si_snr(est, tgt, mask, self.scale_inv, eps)
        pair_loss = torch.clamp(pair_loss, min=-30)
        return torch.sum(pair_loss, dim=1)

    def __call__(self, **kwargs):
        estims = kwargs['estims']
        
        # 🌟 [动态对齐当前设备]
        if isinstance(estims, list):
            est_stack = torch.stack(estims, dim=1)
            current_device = estims[0].device
        else:
            est_stack = estims
            current_device = estims.device
            
        targets = [target.to(current_device) for target in kwargs["target_attr"]]
        tgt_stack = torch.stack(targets, dim=1)
        max_length = min(est_stack.shape[-1], tgt_stack.shape[-1])
        est_stack = est_stack[..., :max_length]
        tgt_stack = tgt_stack[..., :max_length]
        valid_lengths = _valid_lengths(
            kwargs["input_sizes"],
            max_length,
            current_device,
        )
        mask = _length_mask(valid_lengths, max_length, est_stack.dtype).unsqueeze(1)
        
        B = est_stack.shape[0]
        perms = list(permutations(range(self.num_spks)))
        losses = []
        for p in perms:
            curr_tgt = tgt_stack[:, list(p), :]
            loss = self.sisnr_loss_batch(est_stack, curr_tgt, mask)
            losses.append(loss)
        
        losses_stack = torch.stack(losses, dim=0) 
        min_loss_values, min_indices = torch.min(losses_stack, dim=0) 
        loss_sisnr = torch.mean(min_loss_values)

        if self.stft_weight > 0.0:
            loss_stft_acc = 0.0
            for b in range(B):
                best_perm = perms[min_indices[b]]
                valid_length = int(valid_lengths[b].item())
                for s in range(self.num_spks):
                    est_wav = est_stack[b, s, :valid_length]
                    tgt_wav = tgt_stack[b, best_perm[s], :valid_length]
                    if self.scale_inv:
                        alpha = torch.sum(est_wav * tgt_wav) / (torch.sum(est_wav ** 2) + 1e-8)
                        est_wav = alpha * est_wav
                    loss_stft_acc += self.mr_stft_loss(est_wav, tgt_wav)
            # Match SI-SNR's sum-over-speakers convention; Engine averages later.
            loss_stft = loss_stft_acc / B
        else:
            loss_stft = 0.0

        return loss_sisnr + self.stft_weight * loss_stft

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNR_mag:
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    num_stages: int
    num_spks: int
    scale_inv: bool
    mel_opt: bool
    stft: List[Any] = field(init=False)
    mel_fb: Callable[[torch.Tensor], torch.Tensor] = field(init=False)
    
    def __post_init__(self):
        self.stft = [STFT(self.device, self.frame_length, self.frame_shift, self.window) for _ in range(self.num_stages)]
        self.mel_fb = MelScale(n_mels=80, sample_rate=16000, n_stft=int(self.frame_length / 2) + 1).to(self.device) if self.mel_opt else lambda x: x

    def __call__(self, **kwargs):
        estims = kwargs['estims']
        idx = kwargs['idx']
        
        # 🌟 [动态对齐当前设备]
        current_device = estims[0].device 
        targets = [t.to(current_device) for t in kwargs["target_attr"]]
        max_length = min(
            *(estimate.shape[-1] for estimate in estims),
            *(target.shape[-1] for target in targets),
        )
        valid_lengths = _valid_lengths(
            kwargs["input_sizes"],
            max_length,
            current_device,
        )
        
        if self.mel_opt and hasattr(self.mel_fb, 'to'):
            self.mel_fb = self.mel_fb.to(current_device)
        
        def _STFT_Mag_SDR_loss(permute, eps=1.0e-12):
            loss_for_permute = []
            for batch_idx, valid_length in enumerate(valid_lengths.tolist()):
                utterance_loss = 0.0
                for s, t in enumerate(permute):
                    mix = estims[s][batch_idx:batch_idx + 1, :valid_length]
                    src = targets[t][batch_idx:batch_idx + 1, :valid_length]
                    mix_zm = mix - torch.mean(mix, dim=-1, keepdim=True)
                    src_zm = src - torch.mean(src, dim=-1, keepdim=True)
                    if self.scale_inv:
                        scale = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
                        src_zm = torch.clamp(scale, min=1e-2) * src_zm

                    mix_mag = self.stft[idx](mix_zm)[0]
                    src_mag = self.stft[idx](src_zm)[0]

                    if self.mel_opt:
                        mix_mag = self.mel_fb(mix_mag)
                        src_mag = self.mel_fb(src_mag)
                    utterance_loss = utterance_loss - 20 * torch.log10(
                        eps
                        + l2norm(l2norm(src_mag))
                        / (l2norm(l2norm(mix_mag - src_mag)) + eps)
                    ).squeeze()
                loss_for_permute.append(utterance_loss)
            return torch.stack(loss_for_permute)
        
        pscore = torch.stack([_STFT_Mag_SDR_loss(p) for p in permutations(range(self.num_spks))])
        min_perutt, _ = torch.min(pscore, dim=0)
        return torch.mean(min_perutt)

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNR_time:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __call__(self, **kwargs):
        estims = kwargs['estims']
        
        # 🌟 [动态对齐当前设备]
        current_device = estims[0].device
        input_sizes = kwargs["input_sizes"].to(current_device)
        targets = [target.to(current_device) for target in kwargs["target_attr"]]
        
        def _SDR_loss(permute, eps=1.0e-8):
            loss_for_permute = []
            for s, t in enumerate(permute):
                mix = estims[s]
                src = targets[t]
                
                mix_zm = mix - torch.mean(input=mix, dim=-1, keepdim=True)
                src_zm = src - torch.mean(input=src, dim=-1, keepdim=True)
                if self.scale_inv:
                    scale_factor = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
                    src_zm_scale = scale_factor * src_zm
                else:
                    src_zm_scale = src_zm
                
                utt_loss = - 20 * torch.log10(eps + l2norm(src_zm_scale) / (l2norm(mix_zm - src_zm_scale) + eps))
                utt_loss = torch.clamp(utt_loss, min=-30)
                
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)
        
        pscore = torch.stack([_SDR_loss(p) for p in permutations(range(self.num_spks))])
        min_perutt, _ = torch.min(pscore, dim=0)
        num_utts = input_sizes.shape[0]
        return torch.sum(min_perutt) / num_utts

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNRi:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __call__(self, **kwargs):
        estims = kwargs['estims']
        
        # 🌟 [动态对齐当前设备]
        current_device = estims[0].device
        targets = [t.to(current_device) for t in kwargs["target_attr"]]
        mixture = kwargs['mixture'].to(current_device)
        eps = kwargs['eps']
        est_stack = torch.stack(estims, dim=1)
        tgt_stack = torch.stack(targets, dim=1)
        max_length = min(
            est_stack.shape[-1],
            tgt_stack.shape[-1],
            mixture.shape[-1],
        )
        est_stack = est_stack[..., :max_length]
        tgt_stack = tgt_stack[..., :max_length]
        mixture = mixture[..., :max_length]
        valid_lengths = _valid_lengths(
            kwargs["input_sizes"],
            max_length,
            current_device,
        )
        mask = _length_mask(valid_lengths, max_length, est_stack.dtype).unsqueeze(1)
        mixture_stack = mixture.unsqueeze(1).expand(-1, self.num_spks, -1)

        permutation_scores = []
        for permute in permutations(range(self.num_spks)):
            permuted_targets = tgt_stack[:, list(permute), :]
            estimate_score = _masked_si_snr(
                est_stack,
                permuted_targets,
                mask,
                self.scale_inv,
                eps,
            )
            mixture_score = _masked_si_snr(
                mixture_stack,
                permuted_targets,
                mask,
                self.scale_inv,
                eps,
            )
            permutation_scores.append(estimate_score - mixture_score)

        score_stack = torch.stack(permutation_scores, dim=0)
        best_total, best_indices = torch.max(score_stack.sum(dim=-1), dim=0)
        batch_indices = torch.arange(est_stack.shape[0], device=current_device)
        best_source_scores = score_stack[best_indices, batch_indices]
        return best_total.mean(), best_source_scores.mean(dim=0)

@logger_wraps()
@dataclass(slots=True)
class PIT_SDRi:
    device: torch.device
    dump: int
    
    def __call__(self, **kwargs):
        estims = torch.stack(kwargs['estims'], dim=0).squeeze(1)
        
        # 🌟 [动态对齐当前设备，保护 mir_eval]
        current_device = estims.device
        input_sizes = kwargs["input_sizes"].to(current_device)
        targets = [t.to(current_device) for t in kwargs["target_attr"]]
        
        targets = torch.stack(targets, dim=0).squeeze(1)
        input = torch.cat([kwargs['mixture'], kwargs['mixture']], dim=0)
        
        targets = targets.float().cpu().numpy()
        estims = estims.float().cpu().numpy()
        input = input.float().cpu().numpy()

        min_perutt_out, _, _, _ = bss_eval_sources(targets, estims)
        min_perutt_in, _, _, _ = bss_eval_sources(targets, input)
        
        num_utts = input_sizes.shape[0]
        return np.sum(min_perutt_out - min_perutt_in) / num_utts, min_perutt_out - min_perutt_in
