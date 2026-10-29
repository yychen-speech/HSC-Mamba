import math

import torch

from .network import GlobalBlock


class ConservativePairCoupledMamba(torch.nn.Module):
    """Bounded, permutation-equivariant evidence exchange for two speaker streams."""

    def __init__(
        self,
        channels: int,
        num_spks: int = 2,
        hidden_channels: int = 256,
        n_mamba: int = 1,
        bidirectional: bool = True,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        fused_add_norm: bool = False,
        rms_norm: bool = True,
        dropout: float = 0.0,
        residual_scale_init: float = 0.01,
        residual_scale_max: float = 0.05,
        residual_scale_learnable: bool = True,
        confidence_gate_hidden_channels: int = 64,
        confidence_gate_zero_init: bool = True,
        confidence_gate_bias_init: float = 0.0,
        presence_temperature: float = 1.0,
        confidence_min: float = 0.2,
        use_relation_norm: bool = True,
        rho_ramp_epochs: int = 10,
        zero_init_output: bool = True,
        log_stats: bool = True,
    ):
        super().__init__()
        if int(num_spks) != 2:
            raise ValueError("ConservativePairCoupledMamba supports exactly two speakers.")
        if int(channels) < 1 or int(hidden_channels) < 1:
            raise ValueError("channels and hidden_channels must be positive.")
        if float(residual_scale_max) <= 0.0:
            raise ValueError("residual_scale_max must be positive.")
        if not 0.0 <= float(residual_scale_init) <= float(residual_scale_max):
            raise ValueError("residual_scale_init must be in [0, residual_scale_max].")
        if int(confidence_gate_hidden_channels) < 1:
            raise ValueError("confidence_gate_hidden_channels must be positive.")
        if float(presence_temperature) <= 0.0:
            raise ValueError("presence_temperature must be positive.")
        if not 0.0 <= float(confidence_min) <= 1.0:
            raise ValueError("confidence_min must be in [0, 1].")
        if int(rho_ramp_epochs) < 0:
            raise ValueError("rho_ramp_epochs must be non-negative.")

        self.channels = int(channels)
        self.num_spks = 2
        self.residual_scale_max = float(residual_scale_max)
        self.residual_scale_learnable = bool(residual_scale_learnable)
        self.presence_temperature = float(presence_temperature)
        self.confidence_min = float(confidence_min)
        self.use_relation_norm = bool(use_relation_norm)
        self.rho_ramp_epochs = int(rho_ramp_epochs)
        self.current_epoch = self.rho_ramp_epochs
        self.log_stats = bool(log_stats)

        self.stream_norm = torch.nn.GroupNorm(1, self.channels, eps=1.0e-8)
        self.input_proj = torch.nn.Conv1d(
            4 * self.channels,
            int(hidden_channels),
            kernel_size=1,
        )
        self.input_act = torch.nn.SiLU()
        self.temporal_processor = GlobalBlock(
            in_channels=int(hidden_channels),
            n_mamba=int(n_mamba),
            bidirectional=bool(bidirectional),
            d_state=int(d_state),
            expand=int(expand),
            d_conv=int(d_conv),
            fused_add_norm=bool(fused_add_norm),
            rms_norm=bool(rms_norm),
        )
        self.dropout = torch.nn.Dropout(float(dropout))
        self.output_proj = torch.nn.Conv1d(
            int(hidden_channels),
            self.channels,
            kernel_size=1,
        )
        if bool(zero_init_output):
            torch.nn.init.zeros_(self.output_proj.weight)
            torch.nn.init.zeros_(self.output_proj.bias)

        confidence_hidden = int(confidence_gate_hidden_channels)
        self.confidence_gate = torch.nn.Sequential(
            torch.nn.Conv1d(3 * self.channels, confidence_hidden, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(confidence_hidden, 1, kernel_size=1),
        )
        if bool(confidence_gate_zero_init):
            torch.nn.init.zeros_(self.confidence_gate[-1].weight)
        torch.nn.init.constant_(
            self.confidence_gate[-1].bias,
            float(confidence_gate_bias_init),
        )

        if self.residual_scale_learnable:
            fraction = float(residual_scale_init) / self.residual_scale_max
            fraction = min(max(fraction, 1.0e-6), 1.0 - 1.0e-6)
            self.raw_scale = torch.nn.Parameter(
                torch.tensor(
                    math.log(fraction / (1.0 - fraction)),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_buffer(
                "fixed_scale",
                torch.tensor(float(residual_scale_init), dtype=torch.float32),
            )

        self.diagnostic_module_type = "cpc_mamba"
        self.diagnostic_location = "after_bottleneck_split"
        self.diagnostic_stage_idx = -1
        self.diagnostic_module_idx = 0
        self.diagnostic_global_idx = 0
        self.last_diag_stats = {}

    def effective_scale(self):
        if self.residual_scale_learnable:
            return self.residual_scale_max * torch.sigmoid(self.raw_scale)
        return self.fixed_scale.clamp(0.0, self.residual_scale_max)

    def set_epoch(self, epoch: int):
        self.current_epoch = max(0, int(epoch))

    def rho_ramp_factor(self):
        if self.rho_ramp_epochs <= 0:
            return 1.0
        return min(
            1.0,
            max(0.0, float(self.current_epoch) / self.rho_ramp_epochs),
        )

    @staticmethod
    def _rms_normalize(features: torch.Tensor):
        rms = features.square().mean(dim=1, keepdim=True).add(1.0e-8).sqrt()
        return features / rms

    @staticmethod
    def _stream_correlation(first: torch.Tensor, second: torch.Tensor):
        first = first.flatten(1)
        second = second.flatten(1)
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)
        denominator = first.norm(dim=1) * second.norm(dim=1)
        return (first * second).sum(dim=1) / denominator.clamp_min(1.0e-8)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise RuntimeError("CPC-Mamba input must have shape [B * 2, C, T].")
        if x.size(0) % self.num_spks != 0:
            raise RuntimeError(
                "CPC-Mamba batch dimension must be divisible by two speakers."
            )
        if x.size(1) != self.channels:
            raise RuntimeError(
                f"CPC-Mamba expected {self.channels} channels, received {x.size(1)}."
            )

        batch_size = x.size(0) // self.num_spks
        pair = x.reshape(batch_size, self.num_spks, self.channels, x.size(-1))
        normalized = self.stream_norm(x).reshape_as(pair)
        other = normalized.flip(1)
        ordered_terms = torch.cat(
            [normalized, other, normalized - other, normalized * other],
            dim=2,
        ).reshape(
            batch_size * self.num_spks,
            4 * self.channels,
            x.size(-1),
        )

        response = self.input_act(self.input_proj(ordered_terms))
        response = self.temporal_processor(response)
        response = self.output_proj(self.dropout(response))
        response = response.reshape(
            batch_size,
            self.num_spks,
            self.channels,
            x.size(-1),
        )

        raw_delta = 0.5 * (response[:, 0] - response[:, 1])
        reference_rms = (pair.square().mean(dim=(1, 2, 3)) + 1.0e-12).sqrt()
        raw_delta_rms = (raw_delta.square().mean(dim=(1, 2)) + 1.0e-12).sqrt()

        delta_direction = raw_delta / raw_delta_rms.clamp_min(1.0e-8)[:, None, None]
        presence = torch.tanh(
            raw_delta_rms
            / (self.presence_temperature * reference_rms).clamp_min(1.0e-8)
        )
        symmetric_terms = torch.cat(
            [
                0.5 * (normalized[:, 0] + normalized[:, 1]),
                (normalized[:, 0] - normalized[:, 1]).abs(),
                normalized[:, 0] * normalized[:, 1],
            ],
            dim=1,
        )
        if self.use_relation_norm:
            symmetric_terms = self._rms_normalize(symmetric_terms)
        confidence_raw = torch.sigmoid(self.confidence_gate(symmetric_terms))
        confidence = self.confidence_min + (
            1.0 - self.confidence_min
        ) * confidence_raw
        calibrated_delta = (
            confidence
            * presence[:, None, None]
            * reference_rms[:, None, None]
            * delta_direction
        )
        calibrated_delta_rms = (
            calibrated_delta.square().mean(dim=(1, 2)) + 1.0e-12
        ).sqrt()

        scale = self.effective_scale().to(device=x.device, dtype=x.dtype)
        rho_ramp = x.new_tensor(self.rho_ramp_factor())
        exchange = scale * rho_ramp * calibrated_delta
        output_pair = torch.stack(
            [pair[:, 0] + exchange, pair[:, 1] - exchange],
            dim=1,
        )

        if self.log_stats:
            with torch.no_grad():
                output_detached = output_pair.detach()
                pair_detached = pair.detach()
                raw_ratio = raw_delta_rms.detach() / reference_rms.detach()
                calibrated_ratio = (
                    calibrated_delta_rms.detach() / reference_rms.detach()
                )
                exchange_rms = exchange.detach().square().mean(dim=(1, 2)).sqrt()
                conservation_error = (
                    output_detached.sum(dim=1) - pair_detached.sum(dim=1)
                ).abs().mean()
                pre_corr = self._stream_correlation(
                    pair_detached[:, 0], pair_detached[:, 1]
                ).mean()
                post_corr = self._stream_correlation(
                    output_detached[:, 0], output_detached[:, 1]
                ).mean()
                self.last_diag_stats = {
                    "residual_scale": scale.detach(),
                    "effective_residual_scale": (scale * rho_ramp).detach(),
                    "rho_ramp": rho_ramp.detach(),
                    "raw_delta_ratio_mean": raw_ratio.mean(),
                    "raw_delta_ratio_max": raw_ratio.max(),
                    "presence_mean": presence.detach().mean(),
                    "presence_max": presence.detach().max(),
                    "confidence_mean": confidence.detach().mean(),
                    "confidence_std": confidence.detach().std(unbiased=False),
                    "confidence_min": confidence.detach().min(),
                    "confidence_max": confidence.detach().max(),
                    "confidence_saturation_ratio": (
                        (confidence.detach() < 0.05)
                        | (confidence.detach() > 0.95)
                    ).float().mean(),
                    "calibrated_delta_ratio_mean": calibrated_ratio.mean(),
                    "calibrated_delta_ratio_max": calibrated_ratio.max(),
                    "effective_change_ratio": (
                        exchange_rms / reference_rms.detach()
                    ).mean(),
                    "conservation_error": conservation_error,
                    "pre_stream_corr": pre_corr,
                    "post_stream_corr": post_corr,
                    "corr_reduction": pre_corr - post_corr,
                }

        return output_pair.reshape_as(x)
