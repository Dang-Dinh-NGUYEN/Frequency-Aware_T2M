import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
import ptwt

class CSI_DiscreteContentGate(nn.Module):
    """
    Routes through a small, shared codebook of K frequency-modulation
    profiles. Two clips with similar energy get pulled toward the same profile,
    collapsing exactly the clip-to-clip variance that doesn't trace back
    to the action itself -- the mechanism behind the denoiser regression.
    """

    def __init__(self, num_joints, num_bands, num_regimes=2, hidden=None):
        super().__init__()
        hidden = hidden or max(num_joints * 4, 16)

        # K shared profiles, each a full [V, B] pattern. SAME codebook for
        # every sample -- only the ASSIGNMENT below is per-sample.
        self.regimes = nn.Parameter(torch.zeros(num_regimes, num_joints, num_bands))

        # small classifier: joint energy -> soft assignment over K regimes
        self.assign = nn.Sequential(
            nn.Linear(num_joints, hidden), nn.ReLU(), nn.Linear(hidden, num_regimes)
        )
        nn.init.normal_(self.assign[-1].weight, std=0.01)
        nn.init.zeros_(self.assign[-1].bias) 

    def forward(self, joint_energy):
        # joint_energy: [N, V]
        weights = torch.softmax(self.assign(joint_energy), dim=-1)      # [N, K]
        delta = torch.einsum('nk,kvb->nvb', weights, self.regimes)      # [N, V, B]
        return delta 
 
class CSI_DCTEnhancedModule(nn.Module):
    def __init__(self, opt, num_joints=7, use_log_compression=False, zero_init_freq_weights=True):
        super().__init__()
        self.opt = opt
        self.num_bands = int(opt.b)
        self.num_joints = num_joints
        self.use_log_compression = use_log_compression
 
        self.partition_type = getattr(opt, "partition_type", "linear")
        self.decay_type = getattr(opt, "decay_type", "linear")
        self.soft_width = getattr(opt, "threshold_softness", 0.05)
 
        # [V, B] - per-joint, per-band base weights
        init_fn = torch.zeros if zero_init_freq_weights else torch.ones
        self.freq_weights = nn.Parameter(init_fn(self.num_joints, self.num_bands))
 
        self.k = nn.Parameter(torch.tensor(float(opt.k)))
        self.freq_threshold = nn.Parameter(torch.tensor(float(opt.freq_threshold)))
 
        self.content_gate = CSI_DiscreteContentGate(self.num_joints, self.num_bands)
 
        self.register_buffer("cached_bin_indices", None, persistent=False)
        self.cached_T = None
 
    def build_bins(self, T, device):
        freqs = torch.arange(T, device=device, dtype=torch.float32) / T
        if self.partition_type == "linear":
            edges = torch.linspace(0, 1, self.num_bands + 1, device=device)
        elif self.partition_type == "log":
            alpha = 100.0
            edges = torch.linspace(0, math.log1p(alpha), self.num_bands + 1, device=device)
            edges = (torch.exp(edges) - 1) / alpha
            edges[0] = 0.0
            edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}")
        return torch.bucketize(freqs, edges[1:-1])
 
    def compute_decay(self, band_pos):
        if self.decay_type == "linear":
            return 1.0 - band_pos
        elif self.decay_type == "log":
            return 1.0 - torch.log1p(band_pos) / math.log(2.0)
        elif self.decay_type == "exp":
            return torch.exp(-3.0 * band_pos)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
 
    def forward(self, x):
        N, T, V, C = x.shape
        assert V == self.num_joints, f"expected V={self.num_joints} joints, got {V}"
 
        x = x.permute(0, 2, 3, 1)  # [N, V, C, T]
 
        if (
            self.cached_bin_indices is None
            or self.cached_T != T
            or self.cached_bin_indices.device != x.device
        ):
            self.cached_bin_indices = self.build_bins(T, x.device)
            self.cached_T = T
        bin_indices = self.cached_bin_indices  # [T]
 
        x_dct = dct.dct(x, norm='ortho')  # [N, V, C, T]

        # Coarse per-joint motion energy from the raw (pre-DCT) input, pooled
        # over channels and time -- "legs are static, hand is fast, for THIS clip".
        joint_energy = x[..., 1:] - x[..., :-1]
        joint_energy = joint_energy.pow(2).mean(dim=(2, 3))  # [N, V]
 
        delta = self.content_gate(joint_energy)  # [N, V, B]
 
        stable_weights = torch.tanh(self.freq_weights.unsqueeze(0) + delta)   # [N, V, B]
        stable_k = torch.sigmoid(self.k)
        threshold = torch.sigmoid(self.freq_threshold)
 
        band_range = torch.arange(self.num_bands, device=x.device, dtype=x.dtype)
        band_pos = band_range / max(self.num_bands - 1, 1)  # [B]
 
        decay_low = self.compute_decay(band_pos)
        decay_high = self.compute_decay(1.0 - band_pos)
        blend = torch.sigmoid((band_pos - threshold) / self.soft_width)
        response = (1.0 - blend) * (-decay_low) + blend * decay_high  # [B]
 
        band_multipliers = 1 + stable_k * stable_weights * response  # [N, V, B]
        # band_multipliers = torch.clamp(band_multipliers, min=0.01)
 
        full_mask = band_multipliers[..., bin_indices]  # [N, V, T]
        full_mask = full_mask.unsqueeze(2).to(x_dct.dtype)  # [N, V, 1, T] - broadcasts over C
 
        if self.use_log_compression:
            x_dct_sign = torch.sign(x_dct)
            x_dct_log = torch.log1p(torch.abs(x_dct)) * full_mask
            x_dct = x_dct_sign * torch.expm1(x_dct_log)
        else:
            x_dct = x_dct * full_mask
 
        x_idct = dct.idct(x_dct, norm='ortho')
        return x_idct.permute(0, 3, 1, 2) # [N, T, V, C]

    def extract_condition_signals(self, x):
        # x: [N, T, V, C] -- same shape forward() expects, no DCT/IDCT needed
        x = x.permute(0, 2, 3, 1)  # [N, V, C, T]
        joint_energy = x[..., 1:] - x[..., :-1]
        joint_energy = joint_energy.pow(2).mean(dim=(2, 3))          # [N, V]
        regime_weights = torch.softmax(self.content_gate.assign(joint_energy), dim=-1)  # [N, K]
        return regime_weights

class SWTEnhancedModule(nn.Module):
    """
    Stationary-Wavelet-Transform version of DCTEnhancedModule_v2.
 
    Unlike DCT, which gives one global frequency spectrum per (joint, axis)
    covering the whole clip, the SWT is undecimated: every decomposition
    level keeps the same length T as the input. So instead of a single
    [N, V, C, T] spectrum, you get L+1 arrays (L detail levels + one final
    approximation), each still [N, V, C, T] — a genuine time x scale grid,
    not just a time-blind frequency axis. That's the capability this class
    exists to use: band_gains() still gives one scalar gain per (joint,
    level) by default (a direct, "drop-in" port of the DCT version's
    behavior), but because every level is still indexed by time, you can
    later make the gain time-varying too (see the comment in forward()) —
    that's the actual payoff of moving to SWT, not something DCT could do.
    """
 
    def __init__(
        self,
        opt,
        num_joints=7,
        wavelet="haar",
        num_levels=None,          # defaults to getattr(opt, "num_levels", 1)
        boundary_mode="symmetric",  # NOT "periodic" -- motion clips aren't periodic
        use_log_compression=False,
        clamp_min=None,
        clip_high_enhance=False,
        learn_threshold=False,
    ):
        super().__init__()
        self.opt = opt
        self.num_joints = num_joints
        self.wavelet = wavelet
        self.boundary_mode = boundary_mode
        self.use_log_compression = use_log_compression
        self.clamp_min = clamp_min
        self.clip_high_enhance = clip_high_enhance
 
        # Bands = decomposition levels now, not an arbitrary bucket count.
        # L detail levels (cD_1 .. cD_L) + 1 final approximation (cA_L).
        self.num_levels = int(num_levels if num_levels is not None else getattr(opt, "num_levels", 1))
        self.num_bands = self.num_levels + 1
 
        self.soft_width = max(getattr(opt, "threshold_softness", 0.05), 0.02)
 
        tau = float(opt.freq_threshold)
        beta = float(getattr(opt, "b_ratio", tau * 30.0 / 35.0))
        if not (0.0 < tau < 1.0):
            raise ValueError("freq_threshold must be a fraction in (0, 1).")
        if beta <= 0.0:
            raise ValueError("b_ratio must be > 0.")
        self.beta = beta
 
        if learn_threshold:
            self.thr_logit = nn.Parameter(torch.tensor(_logit(tau)))
        else:
            self.register_buffer("thr_logit", torch.tensor(_logit(tau)))
 
        self.k = nn.Parameter(torch.tensor(float(opt.k)))
        self.freq_weights = nn.Parameter(torch.ones(num_joints, self.num_bands))
 
    @property
    def threshold(self):
        return torch.sigmoid(self.thr_logit)
 
    # ------------------------------------------------------------------
    def band_gains(self):
        """g(level) for every body part and band: [V, num_bands].
 
        Same hard/soft threshold blend as the DCT version; `p` is now just
        the level index normalized to [0, 1] (level 0 = finest/highest
        frequency detail, level num_bands-1 = coarsest approximation),
        instead of a bucketized DCT frequency bin.
        """
        B = self.num_bands
        device = self.freq_weights.device
        p = torch.arange(B, device=device, dtype=torch.float32) / B
        tau = self.threshold.float()
        beta = self.beta
 
        kw = self.k * self.freq_weights  # [V, B]
 
        g_low = 1.0 + kw * (1.0 - p / beta)
        atten = 1.0 - (p - beta) / beta
        if self.clip_high_enhance:
            atten = torch.clamp(atten, min=0.0)
        g_high = 1.0 - kw * atten
 
        hard = (p >= tau - 1e-6).float()
        soft = torch.sigmoid((p - tau + 0.5 / B) / self.soft_width)
        blend = hard + soft - soft.detach()
 
        g = (1.0 - blend) * g_low + blend * g_high
        if self.clamp_min is not None:
            g = torch.clamp(g, min=self.clamp_min)
        return g  # [V, B]
 
    # ------------------------------------------------------------------
    def forward(self, x):
        """x: [N, T, V, C] -> [N, T, V, C]"""
        N, T, V, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N * V * C, T)  # ptwt wants [..., T] flattened to 2D
 
        x = x.float()

        # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1]  (ptwt's ordering --
        # VERIFY this against your installed version; some releases
        # order coarsest-first, others finest-first).
        coeffs = ptwt.swt(x, self.wavelet, level=self.num_levels)

        gains = self.band_gains()  # [V, num_bands], band 0 = finest detail .. band B-1 = cA_L
        # Reorder gains to match whatever order `coeffs` comes back in.
        # Below assumes coeffs[0] = cA_L (coarsest) ... coeffs[-1] = cD_1 (finest),
        # i.e. band index B-1 -> coeffs[0], band index 0 -> coeffs[-1].
        gains_per_level = gains.flip(0)  # [B] ordered coarsest -> finest, per joint: [V, B]

        refined_coeffs = []
        for level_idx, c in enumerate(coeffs):
            # c: [N*V*C, T]
            c = c.view(N, V, C, T)
            g = gains_per_level[:, level_idx].view(1, V, 1, 1)  # [1, V, 1, 1]

            # --- Optional: exploit time-localization here instead of a
            # static per-level scalar. E.g. replace the broadcasted `g`
            # above with a gain that also depends on a local energy
            # envelope of `c` along the T axis (sliding-window RMS,
            # or an SNR estimate like the adaptive Kalman module),
            # giving you a true per-(joint, level, time) gain rather
            # than one constant per (joint, level). That's the actual
            # capability switching to SWT buys you over DCT -- worth
            # doing as a deliberate follow-up once the basic swap below
            # is verified correct, not bolted on here at the same time.

            if self.use_log_compression:
                c = torch.sign(c) * torch.expm1(torch.log1p(c.abs()) * g)
            else:
                c = c * g

            refined_coeffs.append(c.reshape(N * V * C, T))

        x_out = ptwt.iswt(refined_coeffs, self.wavelet)  # [N*V*C, T]
 
        x_out = x_out.view(N, V, C, T).permute(0, 3, 1, 2)  # [N, T, V, C]
        return x_out.to(x.dtype)


class DiscreteContentGate(nn.Module):
    """
    Routes through a small, shared codebook of K frequency-modulation
    profiles instead of computing a bespoke continuous delta per clip.
    Two clips with similar energy get pulled toward the same profile,
    collapsing exactly the clip-to-clip variance that doesn't trace back
    to the action itself -- the mechanism behind the denoiser regression.

    Bonus: the K profiles ARE "joints/clips grouped by frequency behavior" --
    this gives you the stage-2 grouping structure as a byproduct of VAE
    training instead of a separate clustering pass afterward.
    """

    def __init__(self, num_joints, num_bands, num_regimes=2, hidden=None):
        super().__init__()
        hidden = hidden or max(num_joints * 4, 16)

        # K shared profiles, each a full [V, B] pattern. SAME codebook for
        # every sample -- only the ASSIGNMENT below is per-sample.
        self.regimes = nn.Parameter(torch.zeros(num_regimes, num_joints, num_bands))

        # small classifier: joint energy -> soft assignment over K regimes
        self.assign = nn.Sequential(
            nn.Linear(num_joints, hidden), nn.ReLU(), nn.Linear(hidden, num_regimes)
        )
        nn.init.normal_(self.assign[-1].weight, std=0.01)
        nn.init.zeros_(self.assign[-1].bias)  # uniform-ish at init; harmless since regimes=0 too

    def forward(self, joint_energy):
        # joint_energy: [N, V]
        weights = torch.softmax(self.assign(joint_energy), dim=-1)      # [N, K]
        delta = torch.einsum('nk,kvb->nvb', weights, self.regimes)      # [N, V, B]
        return delta 
 
class DCTEnhancedModule(nn.Module):
    def __init__(self, opt, num_joints=7, use_log_compression=False, zero_init_freq_weights=True):
        super().__init__()
        self.opt = opt
        self.num_bands = int(opt.b)
        self.num_joints = num_joints
        self.use_log_compression = use_log_compression
 
        self.partition_type = getattr(opt, "partition_type", "linear")
        self.decay_type = getattr(opt, "decay_type", "linear")
        self.soft_width = getattr(opt, "threshold_softness", 0.05)
 
        # [V, B] - per-joint, per-band base weights. No group axis anymore.
        init_fn = torch.zeros if zero_init_freq_weights else torch.ones
        self.freq_weights = nn.Parameter(init_fn(self.num_joints, self.num_bands))
 
        self.k = nn.Parameter(torch.tensor(float(opt.k)))
        self.freq_threshold = nn.Parameter(torch.tensor(float(opt.freq_threshold)))
 
        self.content_gate = DiscreteContentGate(self.num_joints, self.num_bands)
 
        self.register_buffer("cached_bin_indices", None, persistent=False)
        self.cached_T = None
 
    def build_bins(self, T, device):
        freqs = torch.arange(T, device=device, dtype=torch.float32) / T
        if self.partition_type == "linear":
            edges = torch.linspace(0, 1, self.num_bands + 1, device=device)
        elif self.partition_type == "log":
            alpha = 100.0
            edges = torch.linspace(0, math.log1p(alpha), self.num_bands + 1, device=device)
            edges = (torch.exp(edges) - 1) / alpha
            edges[0] = 0.0
            edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}")
        return torch.bucketize(freqs, edges[1:-1])
 
    def compute_decay(self, band_pos):
        if self.decay_type == "linear":
            return 1.0 - band_pos
        elif self.decay_type == "log":
            return 1.0 - torch.log1p(band_pos) / math.log(2.0)
        elif self.decay_type == "exp":
            return torch.exp(-3.0 * band_pos)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
 
    def forward(self, x):
        N, T, V, C = x.shape
        assert V == self.num_joints, f"expected V={self.num_joints} joints, got {V}"
 
        x = x.permute(0, 2, 3, 1)  # [N, V, C, T]
 
        if (
            self.cached_bin_indices is None
            or self.cached_T != T
            or self.cached_bin_indices.device != x.device
        ):
            self.cached_bin_indices = self.build_bins(T, x.device)
            self.cached_T = T
        bin_indices = self.cached_bin_indices  # [T]
 
        x_dct = dct.dct(x, norm='ortho')  # [N, V, C, T]
        spectrum = x_dct

        # Coarse per-joint motion energy from the raw (pre-DCT) input, pooled
        # over channels and time -- "legs are static, hand is fast, for THIS clip".
        joint_energy = x[..., 1:] - x[..., :-1]
        joint_energy = joint_energy.pow(2).mean(dim=(2, 3))  # [N, V]
 
        delta = self.content_gate(joint_energy)  # [N, V, B]
 
        stable_weights = torch.tanh(self.freq_weights.unsqueeze(0) + delta)   # [N, V, B]
        stable_k = torch.sigmoid(self.k)
        threshold = torch.sigmoid(self.freq_threshold)
 
        band_range = torch.arange(self.num_bands, device=x.device, dtype=x.dtype)
        band_pos = band_range / max(self.num_bands - 1, 1)  # [B]
 
        decay_low = self.compute_decay(band_pos)
        decay_high = self.compute_decay(1.0 - band_pos)
        blend = torch.sigmoid((band_pos - threshold) / self.soft_width)
        response = (1.0 - blend) * (-decay_low) + blend * decay_high  # [B]
 
        band_multipliers = 1 + stable_k * stable_weights * response  # [N, V, B]
        # band_multipliers = torch.clamp(band_multipliers, min=0.01)
 
        # RMS-normalize across bands so the mask reshapes the spectrum without
        # drifting overall energy up or down.
        # rms = band_multipliers.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # band_multipliers = band_multipliers / (rms + 1e-6)  # [N, V, B]
 
        full_mask = band_multipliers[..., bin_indices]  # [N, V, T]
        full_mask = full_mask.unsqueeze(2).to(x_dct.dtype)  # [N, V, 1, T] - broadcasts over C
 
        if self.use_log_compression:
            x_dct_sign = torch.sign(x_dct)
            x_dct_log = torch.log1p(torch.abs(x_dct)) * full_mask
            x_dct = x_dct_sign * torch.expm1(x_dct_log)
        else:
            x_dct = x_dct * full_mask
 
        x_idct = dct.idct(x_dct, norm='ortho')
        # spectrum = dct.idct(x_dct, norm='ortho').permute(0, 3, 1, 2)
        return x_idct.permute(0, 3, 1, 2)
    
    
def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))
 
class DCTEnhancedModule_v5(nn.Module):
    def __init__(self, opt, num_joints=7,
        hidden_dim=64,
        num_bands=None,
        max_delta=0.20,
        use_joint_energy=True,
        use_spectral_energy=True,
        normalize_spectrum=True,
        zero_init=True,
        clamp_min=0.01,
        renorm_bands=False,
    ):
        super().__init__()

        self.opt = opt
        self.num_joints = num_joints

        # Number of frequency bands.
        self.num_bands = int(num_bands if num_bands is not None else opt.b)

        self.hidden_dim = hidden_dim
        self.max_delta = max_delta

        self.use_joint_energy = use_joint_energy
        self.use_spectral_energy = use_spectral_energy
        self.normalize_spectrum = normalize_spectrum

        self.clamp_min = clamp_min
        self.renorm_bands = renorm_bands

       
        # Global frequency prior
        self.freq_weights = nn.Parameter(torch.zeros(num_joints, self.num_bands)) # [V, B]

        # Global strength of frequency modulation.
        self.k = nn.Parameter(torch.tensor(float(opt.k)))

        # ---------------------------------------------------------
        # Content encoder
        # ---------------------------------------------------------
        #
        # Input per joint:
        #
        #   joint motion energy : 1
        #   spectral energy    : B
        #
        # Output:
        #
        #   sample-specific correction : B
        #
        input_dim = 0

        if self.use_joint_energy:
            input_dim += 1

        if self.use_spectral_energy:
            input_dim += self.num_bands

        if input_dim == 0:
            raise ValueError(
                "At least one of use_joint_energy or "
                "use_spectral_energy must be True."
            )

        self.content_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.content_to_delta = nn.Linear(
            hidden_dim,
            self.num_bands
        )

        # ---------------------------------------------------------
        # Initialize close to identity
        # ---------------------------------------------------------
        #
        # Initially:
        #
        #       delta ≈ 0
        #
        # so the module behaves approximately as:
        #
        #       X' = X
        #
        if zero_init:
            nn.init.zeros_(self.content_to_delta.weight)
            nn.init.zeros_(self.content_to_delta.bias)

        # ---------------------------------------------------------
        # Optional cached frequency bins
        # ---------------------------------------------------------

        self.register_buffer("cached_bin_indices", None, persistent=False)
        self.cached_T = None

    # =============================================================
    # Frequency bins
    # =============================================================

    def build_bins(self, T, device):

        # Normalized frequency:
        #
        # 0 ---------------------- 1
        #
        # DC                         Nyquist

        freqs = torch.arange(T, device=device, dtype=torch.float32) / max(T - 1, 1)

        edges = torch.linspace(0, 1, self.num_bands + 1, device=device)

        bin_indices = torch.bucketize(freqs, edges[1:-1])

        return bin_indices

    # Joint motion energy
    @staticmethod
    def compute_joint_energy(x):
        """
        x: [N, V, C, T]
        Returns: [N, V]
        Measures how much each body part moves.
        """
        velocity = x[..., 1:] - x[..., :-1]
        energy = velocity.pow(2).mean(dim=(2, 3))

        # Log compression makes the signal less dominated by highly active joints.
        # energy = torch.log1p(energy)

        return energy

    # =============================================================
    # Spectral energy
    # =============================================================

    def compute_spectral_energy(self, x_dct):
        """
        x_dct:
            [N, V, C, T]

        Returns:
            band_energy:
                [N, V, B]
        """

        # Power spectrum
        #
        # [N, V, C, T]
        power = x_dct.pow(2)

        # Average channels
        #
        # [N, V, T]
        spectral_energy = power.mean(dim=2)

        # ---------------------------------------------------------
        # Map individual DCT coefficients to frequency bands
        # ---------------------------------------------------------

        N, V, T = spectral_energy.shape

        if (
            self.cached_bin_indices is None
            or self.cached_T != T
            or self.cached_bin_indices.device != x_dct.device
        ):
            self.cached_bin_indices = self.build_bins(
                T,
                x_dct.device
            )
            self.cached_T = T

        bin_indices = self.cached_bin_indices

        # ---------------------------------------------------------
        # Aggregate frequency coefficients into bands
        # ---------------------------------------------------------

        band_energy = torch.zeros(
            N,
            V,
            self.num_bands,
            device=x_dct.device,
            dtype=x_dct.dtype
        )

        for b in range(self.num_bands):

            mask = bin_indices == b

            if mask.any():
                band_energy[:, :, b] = (
                    spectral_energy[:, :, mask]
                    .mean(dim=-1)
                )

        # ---------------------------------------------------------
        # Normalize across frequency
        # ---------------------------------------------------------

        if self.normalize_spectrum:

            total_energy = band_energy.sum(dim=-1, keepdim=True)

            band_energy = (band_energy/ (total_energy + 1e-6))

        # Log compression is useful when the DC/low-frequency
        # coefficients dominate strongly.
        band_energy = torch.log1p(
            band_energy
        )

        return band_energy

    # =============================================================
    # Content-dependent frequency modulation
    # =============================================================

    def compute_content_delta(
        self,
        joint_energy,
        spectral_energy
    ):
        """
        joint_energy:
            [N, V]

        spectral_energy:
            [N, V, B]

        Returns:
            delta:
                [N, V, B]
        """

        features = []

        if self.use_joint_energy:

            joint_energy_norm = (
                joint_energy
                / (
                    joint_energy.mean(
                        dim=1,
                        keepdim=True
                    )
                    + 1e-6
                )
            )

            features.append(
                joint_energy_norm.unsqueeze(-1)
            )

        if self.use_spectral_energy:

            features.append(
                spectral_energy
            )

        # [N, V, input_dim]
        condition = torch.cat(features, dim=-1)

        # Flatten joints into batch dimension
        N, V, D = condition.shape

        condition = condition.reshape(
            N * V,
            D
        )

        hidden = self.content_encoder(
            condition
        )

        delta = self.content_to_delta(
            hidden
        )

        delta = delta.reshape(
            N,
            V,
            self.num_bands
        )

        # Bounded modulation.
        #
        # delta ∈ [-max_delta, +max_delta]
        #
        delta = self.max_delta * torch.tanh(delta)

        return delta

    def forward(self, x, return_info=False):
        """
        x: [N, T, V, C]
        out: [N, T, V, C]
        optionally:
            info dictionary for analysis.
        """
        N, T, V, C = x.shape

        # Rearrange for temporal DCT
        x_perm = x.permute(0, 2, 3, 1) # [N, V, C, T]

        # DCT
        with torch.amp.autocast('cuda', enabled=False):
            x_dct = dct.dct(x_perm.float(), norm='ortho')

        # 1. Joint-level motion condition
        joint_energy = self.compute_joint_energy(x_perm) #[N, V]

        # 2. Spectral condition
        spectral_energy = self.compute_spectral_energy(x_dct) # [N, V, B]

        # 3. Content-adaptive correction
        delta = self.compute_content_delta(joint_energy, spectral_energy) # [N, V, B]

        # 4. Global frequency prior
        global_weight = self.freq_weights.unsqueeze(0)

        combined_weight = (global_weight + delta) # [1, V, B] + [N, V, B]

        # Stable bounded weight
        stable_weight = torch.tanh(combined_weight)

        # Global modulation strength
        stable_k = torch.sigmoid(self.k)

        # ---------------------------------------------------------
        # 5. Frequency response
        # ---------------------------------------------------------
        #
        # We use a simple smooth low/high frequency response.
        #
        # low frequencies:
        #       response < 0
        #
        # high frequencies:
        #       response > 0
        #
        # The content module determines how strongly each
        # joint/sample follows this prior.
        # ---------------------------------------------------------

        band_pos = torch.linspace(0, 1, self.num_bands, device=x.device, dtype=x.dtype)

        # Smooth transition around the middle.
        threshold = 0.3

        response = torch.tanh(
            (band_pos - threshold) * 4.0
        )

        # [1, 1, B]
        response = response.view(
            1, 1, -1
        )

        # ---------------------------------------------------------
        # Final band multiplier
        # ---------------------------------------------------------

        band_multipliers = (1.0 + stable_k * stable_weight * response)

        # Optional RMS normalization
        if self.renorm_bands:
            rms = (band_multipliers.pow(2).mean(dim=-1, keepdim=True).sqrt())
            band_multipliers = (band_multipliers/ (rms + 1e-6))

        # Prevent negative spectral scaling
        if self.clamp_min is not None:
            band_multipliers = torch.clamp(band_multipliers, min=self.clamp_min)

        # ---------------------------------------------------------
        # Expand [N,V,B] → [N,V,T]
        # ---------------------------------------------------------

        bin_indices = self.cached_bin_indices

        full_mask = (band_multipliers[..., bin_indices])
        
        full_mask = full_mask.unsqueeze(2) # [N,V,1,T]

        # Apply modulation
        x_dct_enhanced = (x_dct * full_mask)

        # IDCT
        with torch.amp.autocast('cuda', enabled=False):
            x_out = dct.idct(x_dct_enhanced,norm="ortho")
            x_out = x_out.permute(0, 3, 1, 2)

        if return_info:
            info = {
                "joint_energy": joint_energy.detach(),
                "spectral_energy": spectral_energy.detach(),
                "content_delta": delta.detach(),
                "global_weight": self.freq_weights.detach(),
                "band_multipliers": band_multipliers.detach(),
                "spectrum":x_dct.detach(),
            }

            return x_out, info

        return x_out


class SpectralKalmanFilter(nn.Module):
    """
    1D Differentiable Spectral Kalman Filter with SNR-adaptive process/
    measurement noise.
 
    Implements the Adaptive High-Frequency Refinement algorithm: the first
    `k0` DCT coefficients (low frequencies) are passed through unchanged;
    coefficients from k0 onward are smoothed with a per-(joint, axis, batch
    sample) Kalman filter whose process variance Q and measurement variance
    R are derived from that curve's own high-frequency energy ratio, rather
    than being fixed constants.
    """
    def __init__(self, opt=None, k0=5, init_process_variance=1e-4, init_measure_variance=1e-2,
                 lambda_q=0.2, lambda_r=0.5, eps=1e-8):
        super().__init__()
        self.opt = opt
 
        if opt is not None:
            k0 = getattr(opt, "k0", k0)
            init_process_variance = getattr(opt, "kalman_q0", init_process_variance)
            init_measure_variance = getattr(opt, "kalman_r0", init_measure_variance)
            lambda_q = getattr(opt, "kalman_lambda_q", lambda_q)
            lambda_r = getattr(opt, "kalman_lambda_r", lambda_r)
 
        # Learnable log-variance *base* parameters (Q0, R0 in the pseudocode)
        self.log_q = nn.Parameter(torch.tensor(math.log(init_process_variance)))
        self.log_r = nn.Parameter(torch.tensor(math.log(init_measure_variance)))
 
        self.k_base = k0           # frequency threshold k0: low bands kept untouched
        self.lambda_q = lambda_q   # process-noise sensitivity to SNR
        self.lambda_r = lambda_r   # measurement-noise sensitivity to SNR
        self.eps = eps
 
        assert isinstance(self.k_base, int), (
            f"k0 must be an int, got {type(self.k_base)} — check the opt attribute "
            f"name this is reading from, or pass k0=... explicitly."
        )
 
    def forward(self, x):
        """
        x: [N, T, V, C] predicted motion (batch, frames, joints, xyz)
        Returns: refined motion, same shape [N, T, V, C]
 
        Vectorized across (N, V, C): the only genuinely sequential part is
        the Kalman recursion over time, so that's the only Python-level
        loop. Everything else (DCT energy / SNR / Q / R, and each Kalman
        step itself) is one batched tensor op covering every batch sample,
        joint, and axis at once instead of looping over them one at a time.
        """
        N, T, V, C = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # [N, V, C, T]
 
        # --- 1. Frequency Domain Transformation ---
        x_dct = dct.dct(x_perm, norm="ortho")  # [N, V, C, T]
 
        Q0 = torch.exp(self.log_q)
        R0 = torch.exp(self.log_r)
        eps = self.eps
 
        # Guard against sequences shorter than the configured low-frequency
        # band (e.g. padded/short clips): nothing left to adaptively filter.
        k0 = min(self.k_base, T)
        if k0 >= T:
            x_idct = dct.idct(x_dct, norm="ortho")
            return x_idct.permute(0, 3, 1, 2)
 
        # --- 2. Adaptive High-Frequency Refinement (vectorized) ---
        low = x_dct[..., :k0]   # [N, V, C, k0]    preserved low frequencies
        hf = x_dct[..., k0:]    # [N, V, C, T-k0]  band to be Kalman-filtered
 
        # Per-(N, V, C) SNR estimate from the high-frequency energy ratio —
        # one reduction over the whole tensor instead of V*C separate ones.
        hf_energy = (hf ** 2).sum(dim=-1)          # [N, V, C]
        total_energy = (x_dct ** 2).sum(dim=-1)     # [N, V, C]
        rho = hf_energy / (total_energy + eps)      # [N, V, C]
        snr_est = (1.0 - rho) / (rho + eps)          # [N, V, C]
 
        # Adaptive process/measurement variance, per (N, V, C)
        Q = Q0 * (1.0 + self.lambda_q / (snr_est + eps))  # [N, V, C]
        R = R0 / (1.0 + self.lambda_r * snr_est)            # [N, V, C]
 
        # Sequential Kalman filter over k = k0 .. T-1, batched over (N, V, C)
        # per step instead of once per (j, d) pair. Built out-of-place
        # (stack/cat) rather than in-place slice assignment: writing into a
        # tensor that's itself the output of a differentiable op (dct.dct)
        # breaks autograd ("modified by an in-place operation") on backward.
        x_hat = hf[..., 0]   # [N, V, C]
        P = R                # P_{k0|k0} <- R
        hf_filtered = [x_hat]
        T_hf = hf.shape[-1]
        for k in range(1, T_hf):
            x_pred = x_hat
            P_pred = P + Q
            z_k = hf[..., k]
            K = P_pred / (P_pred + R)
            x_hat = x_pred + K * (z_k - x_pred)
            P = (1.0 - K) * P_pred
            hf_filtered.append(x_hat)
        hf_filtered = torch.stack(hf_filtered, dim=-1)  # [N, V, C, T-k0]
 
        x_dct_refined = torch.cat([low, hf_filtered], dim=-1)  # [N, V, C, T]
 
        # --- 3. Inverse Transformation ---
        x_idct = dct.idct(x_dct_refined, norm="ortho")  # [N, V, C, T]
        return x_idct.permute(0, 3, 1, 2)  # [N, T, V, C]
 
 
 
class SmoothFrequencyGate(nn.Module):
    def __init__(self, num_joints, hidden=32, hf_split_ratio=0.3):
        super().__init__()
        self.num_joints = num_joints
        self.hf_split_ratio = hf_split_ratio
 
        # Differentiates joints for the shared net (e.g., root/torso vs extremities)
        self.joint_bias = nn.Parameter(torch.zeros(num_joints))
 
        # MLP predicting spectral gain adjustments
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
 
        # Differentiable Kalman Filter for High-Frequency Denoising
        self.kalman_filter = SpectralKalmanFilter()
 
    def forward(self, freq_pos, energy_dct, k):
        """
        freq_pos: [T]
        energy_dct: [N, V, T]
        k: scalar in (0, 1)
        """
        N, V, T = energy_dct.shape
        assert V == self.num_joints, (
            f"SmoothFrequencyGate was built with num_joints={self.num_joints} "
            f"but received data with V={V} joints — self.joint_bias can't be "
            f"broadcast against a mismatched joint count."
        )
        fp = freq_pos.view(1, 1, T).expand(N, V, T)
        jb = self.joint_bias.view(1, V, 1).expand(N, V, T)
 
        # 1. Predict spectral gain features
        feat = torch.stack([fp, torch.log1p(energy_dct), jb], dim=-1)  # [N, V, T, 3]
        raw = self.net(feat).squeeze(-1)                              # [N, V, T]
        log_gain = k * torch.tanh(raw)                                # [N, V, T]
 
        # 2. Convert to linear gain multiplier domain
        gain = torch.exp(log_gain)                                    # [N, V, T]
 
        # 3. Apply Kalman Filtering selectively on the High-Frequency Spectrum
        hf_start_idx = int(T * (1.0 - self.hf_split_ratio))
 
        if hf_start_idx < T:
            # Low-frequency part (kept raw/unfiltered to preserve macro dynamics)
            gain_lf = gain[:, :, :hf_start_idx]
 
            # High-frequency part (filtered with state estimation)
            gain_hf_raw = gain[:, :, hf_start_idx:]
            gain_hf_filtered = self.kalman_filter(gain_hf_raw)
 
            # Reassemble full frequency spectrum
            gain = torch.cat([gain_lf, gain_hf_filtered], dim=-1)
 
        # 4. Energy conservation per joint per sample
        # Convert back to log space for zero-mean energy normalization
        log_gain_balanced = torch.log(gain + 1e-8)
        log_gain_balanced = log_gain_balanced - log_gain_balanced.mean(dim=-1, keepdim=True)
 
        return torch.exp(log_gain_balanced)  # [N, V, T]
 
 
class DCTEnhancedModule_v9(nn.Module):
    def __init__(self, opt, num_joints=7, hidden=32, hf_split_ratio=0.3):
        super().__init__()
        self.opt = opt
        self.num_joints = num_joints
 
        self.gate = SmoothFrequencyGate(
            num_joints=num_joints,
            hidden=hidden,
            hf_split_ratio=hf_split_ratio
        )
 
        # `self.k` is the pre-sigmoid (logit-space) parameter, so that
        # sigmoid(self.k) == opt.k at initialization, matching the intended
        # starting cap on the log-gain adjustment. A plain
        # nn.Parameter(torch.tensor(opt.k)) would NOT do this: sigmoid(0.05)
        # is ~0.51, not 0.05.
        k0 = float(getattr(opt, "k", 0.05))
        assert 0.0 < k0 < 1.0, f"opt.k must be in (0, 1), got {k0}"
        self.k = nn.Parameter(torch.logit(torch.tensor(k0)))
 
    def forward(self, x):
        # x shape: [N, T, V, C]
        N, T, V, C = x.shape
 
        x = x.permute(0, 2, 3, 1)  # [N, V, C, T]
 
        # 1. Compute Orthogonal DCT Spectrum
        x_dct = dct.dct(x, norm='ortho')  # [N, V, C, T]
 
        # 2. Extract per-position spectral energy
        energy_dct = x_dct.pow(2).mean(dim=2)  # [N, V, T]
        freq_pos = torch.arange(T, device=x.device, dtype=x.dtype) / max(T - 1, 1)  # [T]
 
        # 3. Predict & Kalman-filter frequency multipliers
        stable_k = torch.sigmoid(self.k)
        band_multipliers = self.gate(freq_pos, energy_dct, stable_k)  # [N, V, T]
 
        # 4. Apply filtered mask over frequency spectrum
        full_mask = band_multipliers.unsqueeze(2).to(x_dct.dtype)  # [N, V, 1, T]
        x_dct = x_dct * full_mask
 
        # 5. Inverse DCT back to time domain
        x_idct = dct.idct(x_dct, norm='ortho')
        return x_idct.permute(0, 3, 1, 2)  # Restored to [N, T, V, C]
 

class SpatialFrequencyGate(nn.Module):
    """Gates the DCT-over-joints spectrum. J is fixed and known at
    construction (always num_joints groups) -- unlike T, which varies with
    clip length -- so this uses a direct per-spatial-frequency-index
    parameterization rather than the length-agnostic coordinate-MLP trick
    the temporal gate needs. This is the same shape as the earlier buggy
    content_router, but correctly applied here: fixed input/output size is
    the right fit for a fixed axis, it just wasn't the right fit for T.
    """

    def __init__(self, num_joints, hidden=None):
        super().__init__()
        self.num_joints = num_joints
        hidden = hidden or max(num_joints * 4, 16)

        self.freq_bias = nn.Parameter(torch.zeros(num_joints))  # static per-spatial-freq baseline
        self.net = nn.Sequential(
            nn.Linear(num_joints, hidden), nn.GELU(), nn.Linear(hidden, num_joints)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # content term starts at 0 -> identity pass-through

    def forward(self, spatial_energy, k):
        # spatial_energy: [N, T, J'] -- real per-spatial-frequency energy,
        # pooled over channels, one full J'-length vector per (sample, frame).
        content = self.net(torch.log1p(spatial_energy))     # [N, T, J']
        raw = self.freq_bias.view(1, 1, -1) + content        # [N, T, J']
        log_gain = k * torch.tanh(raw)
        # Conserve each FRAME's total spatial energy across its J'
        # spatial-frequency components -- reallocate, don't freely amplify.
        log_gain = log_gain - log_gain.mean(dim=-1, keepdim=True)
        return torch.exp(log_gain)  # [N, T, J']


class DCTEnhancedModule_v3(nn.Module):
    """Sequential round-trip: DCT over joints -> gate -> IDCT back to
    joint-identity space -> existing temporal gate (unchanged) -> IDCT.
    Joint identity is fully restored before the temporal gate runs, so the
    validated chest-vs-limb finding stays expressible exactly as before --
    the temporal gate is conditioning on the same per-joint signal it
    always did, just after a spatially-refined version of it.
    """

    def __init__(self, opt, num_joints=7, hidden=32):
        super().__init__()
        self.opt = opt
        self.num_joints = num_joints

        self.spatial_gate = SpatialFrequencyGate(num_joints, hidden=hidden)
        self.temporal_gate = SmoothFrequencyGate(num_joints, hidden=hidden)

        self.k_spatial = nn.Parameter(torch.tensor(float(getattr(opt, "k_spatial", 0.5))))
        self.k_temporal = nn.Parameter(torch.tensor(float(getattr(opt, "k_temporal", 0.5))))

    def forward(self, x):
        # x: [N, T, J, C]
        N, T, J, C = x.shape
        assert J == self.num_joints, f"expected J={self.num_joints}, got {J}"

        # --- Spatial frequency gating (DCT over J, per frame independently) ---
        x_sj = x.permute(0, 1, 3, 2)              # [N, T, C, J]
        x_sdct = dct.dct(x_sj, norm='ortho')       # [N, T, C, J] -- DCT over J (last dim)

        spatial_energy = x_sdct.pow(2).mean(dim=2)  # [N, T, J] -- real per-spatial-freq energy
        stable_k_s = torch.sigmoid(self.k_spatial)
        spatial_mult = self.spatial_gate(spatial_energy, stable_k_s)  # [N, T, J]

        x_sdct = x_sdct * spatial_mult.unsqueeze(2)  # broadcast over C: [N,T,1,J]
        x_sidct = dct.idct(x_sdct, norm='ortho')     # [N, T, C, J]
        x = x_sidct.permute(0, 1, 3, 2)              # [N, T, J, C] -- joint identity restored

        # --- Temporal frequency gating (unchanged) ---
        x_t = x.permute(0, 2, 3, 1)                # [N, J, C, T]
        x_tdct = dct.dct(x_t, norm='ortho')         # [N, J, C, T]
        spectrum = x_tdct  # pre-mask spectrum

        energy_dct = x_tdct.pow(2).mean(dim=2)      # [N, J, T]
        freq_pos = torch.arange(T, device=x.device, dtype=x.dtype) / max(T - 1, 1)

        stable_k_t = torch.sigmoid(self.k_temporal)
        temporal_mult = self.temporal_gate(freq_pos, energy_dct, stable_k_t)  # [N, J, T]

        full_mask = temporal_mult.unsqueeze(2).to(x_tdct.dtype)  # [N, J, 1, T]
        x_tdct = x_tdct * full_mask

        x_out = dct.idct(x_tdct, norm='ortho')      # [N, J, C, T]
        return x_out.permute(0, 3, 1, 2)  # [N, T, J, C]

    def get_learned_spatial_response(self, device=None):
        """Static per-spatial-frequency-index curve (content=0). Length
        num_joints -- confirm the 7-group ordering respects skeletal
        adjacency before reading meaning into the shape."""
        device = device or self.spatial_gate.freq_bias.device
        energy_zero = torch.zeros(1, 1, self.num_joints, device=device)
        stable_k_s = torch.sigmoid(self.k_spatial)
        return self.spatial_gate(energy_zero, stable_k_s).squeeze(0).squeeze(0)  # [J]

    def get_learned_temporal_response(self, T=64, device=None):
        """Same diagnostic as the temporal-only version, for comparison
        against runs without spatial gating."""
        device = device or self.temporal_gate.joint_bias.device
        freq_pos = torch.arange(T, device=device, dtype=torch.float32) / max(T - 1, 1)
        energy_zero = torch.zeros(1, self.num_joints, T, device=device)
        stable_k_t = torch.sigmoid(self.k_temporal)
        return self.temporal_gate(freq_pos, energy_zero, stable_k_t).squeeze(0)  # [J, T]

class DCTEnhancedModule_v4(nn.Module):
    def __init__(self, opt, k=0.3, num_freq_bands=7, split_freq=4, m=1):
        super().__init__()
        self.opt = opt
        self.k = nn.Parameter(torch.tensor(k))
        self.num_freq_bands = num_freq_bands
        self.split_freq = split_freq  # Note: if split_freq was an absolute frame index, it needs to be normalized or handled carefully!
        self.m = m
        self.freq_weights = nn.Parameter(torch.ones(num_freq_bands)) 

    def forward(self, x):
        N, T, V, C = x.shape
        device = x.device

        # Permute to [N, V, C, T] and apply DCT along T
        x_dct = dct.dct(x.permute(0, 2, 3, 1), norm='ortho')  # [N, V, C, T]

        num_bands = int(self.num_freq_bands)
        
        # 1. Create normalized frequency bins from 0.0 to 1.0 (independent of T)
        # This ensures band boundaries mean the same thing for any sequence length T.
        band_indices = torch.arange(T, device=device, dtype=torch.float32) / max(T - 1, 1) # [T]
        
        # Map each time/frequency step to a band index float from 0 to num_bands
        scaled_freqs = band_indices * num_bands
        band_idx_map = torch.clamp(scaled_freqs.long(), 0, num_bands - 1) # [T]

        # 2. Build scale factors per frequency band based on your original logic
        band_scales = torch.zeros(num_bands, device=device, dtype=x.dtype)
        
        # Assuming split_freq is treated as a normalized threshold or bin index ratio if T changes,
        # or adapt it safely. Let's map split_freq relative to num_bands if it was index-based:
        split_freq_val = int(self.split_freq)
        m_val = int(self.m)

        for i in range(num_bands):
            freq_weight = self.freq_weights[i]

            # Using band index relative condition
            if i <= split_freq_val:
                decay_factor = 1.0 - (i / num_bands)
                scale_val = 1.0 + freq_weight * self.k * decay_factor
            else:
                decay_factor = 1.0 - (i - num_bands // max(m_val, 1)) / num_bands
                scale_val = 1.0 - freq_weight * self.k * decay_factor
            
            band_scales[i] = scale_val

        # 3. Map band scales back to each temporal frequency step T via indexing
        scales = band_scales[band_idx_map]  # [T]

        # Reshape to broadcast across [N, V, C, T] -> [1, 1, 1, T]
        scales = scales.view(1, 1, 1, T)

        # Out-of-place multiplication
        x_dct_enhanced = x_dct * scales

        # Inverse DCT and permute back to [N, T, V, C]
        x_idct = dct.idct(x_dct_enhanced, norm='ortho')
        return x_idct.permute(0, 3, 1, 2)
     
class WaveletEnhancedModule(nn.Module):
    def __init__(self, opt):
        super(WaveletEnhancedModule, self).__init__()
        # Wavelet decomposition
        # self.dwt =  DWT1D(wave=opt.wavelet_mother, mode='zero')
        # self.idwt = IDWT1D(wave=opt.wavelet_mother, mode='zero')

        # Low-frequency branch - Structural motion
        self.attn_salient = DecouplingAttention(opt)
        self.low_joint_transformer = JointTransformer(channels=opt.latent_dim, V=7)

        # High-frequency branch - Dynamic motion
        self.attn_subtle = DecouplingAttention(opt)
        self.trajectory_wise_attn = TrajectoryWiseAttention(channels=opt.latent_dim, V=7)

        # Frequency adaptive gate
        self.frequency_gate = FrequencyGate(channels=opt.latent_dim)

        # Output projection
        self.output_norm = nn.LayerNorm(opt.latent_dim)

    def forward(self, x):
        N, T, V, C = x.shape
        # print(f'----- DWT Enhanced Module -----')
        # print(f'input shape: {x.shape}')

        # Save residual
        residual = x

        # Padding for odd length
        is_odd = (T % 2 != 0)
        if is_odd:
            pass
            # x = F.pad(x, (0, 0, 0, 0, 0, 1), mode='replicate') 
        _, T_padded, V, C = x.shape
        
        # 1. Reshape using T_padded -> [N, V * C, T_padded]
        x_embed = x.permute(0, 2, 3, 1).reshape(N, V * C, T)
        # print(f'x_embed shape: {x_embed.shape}')

        # 2. 1D-DWT Layer (using pytorch_wavelets)
        x_low, x_high = ptwt.swt(x_embed, wavelet="haar", level=1, axis=-1) # self.dwt(x_embed) 
        # print(f'x_low shape: {x_low.shape}')
        # x_high = x_high_list[0]
        # print(f'x_high shape: {x_high.shape}')
        # T_half = x_low.shape[-1] # This is T // 2

        # Restore:
        x_low = x_low.reshape(N, V, C, T).permute(0, 3, 1, 2)
        x_high = x_high.reshape(N, V, C, T).permute(0, 3, 1, 2)
        
        # 3. Low frequency processing
        x_low = self.low_joint_transformer(x_low)
        # print(f'x_low after joint_transformer: {x_low.shape}')
        x_low = self.attn_salient(x_low)
        # print(f'x_low after decoupling attn: {x_low.shape}')

        # 4. High frequency processing
        # x_high = x_high.permute(0, 2, 3, 1)
        x_high = self.attn_subtle(x_high)
        # print("x_high after decoupling attn:", x_high.shape)
        x_high = x_high.permute(0, 3, 1, 2)
        x_high = self.trajectory_wise_attn(x_high)
        # print(f'x_high after trajectory_wise_attn: {x_high.shape}')
        x_high = x_high.permute(0, 2, 3, 1)

        # 5. Frequency adaptive weighting
        low, high = self.frequency_gate(x_low, x_high)

        # 6. Back to wavelet layout
        low = low.permute(0, 2, 3, 1).reshape(N, V*C, T)
        high = high.permute(0, 2, 3, 1).reshape(N, V*C, T)
        
        # IDWT reconstruction
        out = ptwt.iswt([low, high], wavelet="haar", axis=-1) # self.idwt(low, [high])
        out = out.reshape(N, V, C, T).permute(0, 3, 1, 2)

        if is_odd:
            out = out[:,:T]

        out = residual + out

        # out = self.output_norm(out)

        return out

        # 3. DecouplingAttention 
        # mask_salient = self.attn_salient(x_padded).permute(0, 2, 1, 3) 
        # mask_subtle = self.attn_subtle(x_padded).permute(0, 2, 1, 3)

        # 4. Calibration over matching shapes: 
        # x_low/x_high are [N, V*C, T_half]. Reshape to [N, V, C, T_half] -> [N, C, T_half, V]
        # x_salient_base = x_low.view(N, V, C, T_half).permute(0, 2, 3, 1) 
        # x_subtle_base = x_high.view(N, V, C, T_half).permute(0, 2, 3, 1)  

        # Element-wise multiplication
        # x_salient = x_salient_base * mask_salient
        # x_subtle = x_subtle_base * mask_subtle
       
        # 5. Pass subtle features through Trajectory-Wise Attention blocks (if active)
        # x_subtle = self.trajectory_wise_attn(x_subtle)

        # 6. Flatten back to Wavelet coefficient layout for inverse transform: [N, V * C, T_half]
        # x_salient_flat = x_salient.permute(0, 3, 1, 2).reshape(N, V * C, T_half)
        # x_subtle_flat = x_subtle.permute(0, 3, 1, 2).reshape(N, V * C, T_half)

        # 7. Inverse Wavelet Transform (using pytorch_wavelets IDWT matching self.dwt)
        # pytorch_wavelets IDWT expects: idwt((approx_coeffs, [detail_coeffs_list]))
        # x_recon = self.idwt((x_salient_flat, [x_subtle_flat])) # Shape: [N, V * C, T_padded]
        
        # 8. Reconstruct back to native input shape: [N, T_padded, V, C]
        # x_split = x_recon.reshape(N, V, C, T_padded).permute(0, 3, 1, 2)

        # 9. Slice off boundary padding artifacts if frame count was odd
        # if is_odd:
            # x_split = x_split[:, :T, :, :]
            
        # return x_split


class DecouplingAttention(nn.Module):

    def __init__(self, opt):
        super().__init__()
        C = opt.latent_dim

        # Temporal attention
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C),
            nn.GELU(),
            nn.Conv1d(C, 1, kernel_size=1)
        )

        # Joint attention
        self.joint_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, 7)
        )

        # Channel attention
        self.channel_mlp = nn.Sequential(
            nn.Linear(C, C//4),
            nn.GELU(),
            nn.Linear(C//4, C)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # Temporal attention
        # [N,T,V,C] -> [N,C,T]
        xt = x.mean(dim=2)
        xt = xt.permute(0, 2, 1)

        t_mask = self.temporal_conv(xt) # [N,1,T]
        t_mask = t_mask.permute(0, 2, 1) # [N,T,1,1]
        t_mask = t_mask.unsqueeze(2)

        # Joint attention
        xj = x.mean(dim=-1) # average channel [N,T,V]
        xj = xj.mean(dim=1) # [N,V]
      
        j_mask = self.joint_mlp(xj)
        j_mask = self.sigmoid(j_mask)
        j_mask = j_mask[:,None,:,None] # [N,1,V,1]

        # Channel attention
        xc = x.mean(dim=(1,2)) # [N,C]
        c_mask = self.channel_mlp(xc)
        c_mask = self.sigmoid(c_mask)
        c_mask = c_mask[:,None,None,:] # [N,1,1,C]

        # Combine
        mask = (t_mask * j_mask * c_mask)

        # Residual recalibration
        out = x + x * mask

        return out

"""
class DecouplingAttention(nn.Module):
    def __init__(self, opt):
        super(DecouplingAttention, self).__init__()
        self.C_dim = opt.latent_dim  
        
        # Learns temporal patterns per-channel without mixing channels
        self.temporal_compressor = nn.Conv1d(
            in_channels=self.C_dim, 
            out_channels=self.C_dim, 
            kernel_size=2, 
            stride=2,
            groups=self.C_dim 
        )

        # self.temporal_compressor = nn.Linear(in_features=49, out_features=49)
        
        self.conv1d = nn.Conv1d(in_channels=self.C_dim, out_channels=1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # N, T, V, C = x.shape
        # target_T = T // 2

        # Step 1: Spatial Pooling (V) -> [N, C, T]
        x_perm = x.permute(0, 3, 1, 2)  # [N, C, T, V]
        x_spatial = x_perm.mean(dim=-1) # [N, C, T]
        # print(x_spatial.shape)
        
        # Step 2: Learnable temporal compression down to T // 2 
        x_time = self.temporal_compressor(x_spatial) # [N, C, target_T]
        
        # Step 3: Channel compression -> [N, 1, target_T]
        attn_weights = self.conv1d(x_time) 
        attn_mask = self.sigmoid(attn_weights) # [N, 1, target_T]

        # Step 4: Broadcast natively to match [N, target_T, V, C]
        # Reshape to [N, target_T, 1, 1] so it easily broadcasts over V and C
        attn_mask = attn_mask.permute(0, 2, 1).unsqueeze(-1) 
        
        return attn_mask
""" 

class TrajectoryWiseAttention(nn.Module):
    def __init__(self, channels, V=7):
        super().__init__()
        # 1. Temporal attention branch: preserves temporal sharpness (time dimension T)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )
        
        # 2. Spatial/Joint attention branch: dynamically weights important joints (e.g., lower limbs)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(V, V),
            nn.GELU(),
            nn.Linear(V, V)
        )
        
        # Softmax to create normalized attention weights
        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_v = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Input shape: [N, C, T, V] 
        """
        # --- Branch A: Temporal Trajectory Attention ---
        # Collapse spatial dimension to focus purely on the time trajectory path
        x_t_pooled = x.mean(dim=3) # Shape: [N, C, T]
        t_weights = self.temporal_conv(x_t_pooled) # Shape: [N, C, T]
        t_weights = self.softmax_t(t_weights).unsqueeze(-1) # Shape: [N, C, T, 1]
        
        # --- Branch B: Joint/Spatial Attention ---
        # Collapse temporal dimension to focus on which joints matter most per channel
        x_v_pooled = x.mean(dim=2) # Shape: [N, C, V]
        v_weights = self.spatial_mlp(x_v_pooled) # Shape: [N, C, V]
        v_weights = self.softmax_v(v_weights).unsqueeze(2) # Shape: [N, C, 1, V]
        
        # --- Feature Recalibration (Residual style to prevent information loss) ---
        # Instead of multiplying into a zero-sum bottleneck, use a residual scaling gate:
        attention_mask = t_weights * v_weights # Shape: [N, C, T, V]
        
        # Residual connection ensures high frequencies aren't choked out
        out = x + (x * attention_mask)
        
        return out

class JointTransformer(nn.Module):
    def __init__(self, channels, V):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels*4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=1)


    def forward(self,x):
        N,T,V,C=x.shape

        x = x.reshape(N*T, V, C)
        x = self.encoder(x)
        x = x.reshape(N, T, V, C)

        return x

class FrequencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, low, high):
        fusion = torch.cat([low, high], dim=-1)

        weights = self.gate(fusion)

        w_low = weights[...,0:1]
        w_high = weights[...,1:2]

        low = low*w_low
        high = high*w_high

        return low, high

class StationaryWaveletModule(nn.Module):
    def __init__(self, opt):
        super(StationaryWaveletModule, self).__init__()

        self.latent_dim = opt.latent_dim          # C = 32
        self.num_joints = 7 # opt.num_joints          # e.g. 7
        self.level = 3

        ####################################################
        # Frequency-specific temporal convolutions
        ####################################################

        self.low_conv = nn.Conv1d(
            in_channels=self.num_joints * self.latent_dim,
            out_channels=self.latent_dim,
            kernel_size=9,
            padding=4,
        )

        self.high_conv = nn.Conv1d(
            in_channels=self.num_joints * self.latent_dim,
            out_channels=self.latent_dim,
            kernel_size=3,
            padding=1,
        )

        ####################################################
        # Intra-frequency MLP
        ####################################################

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )

        ####################################################
        # Low-frequency Transformer
        ####################################################

        low_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=self.latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.low_transformer = nn.TransformerEncoder(
            low_layer,
            num_layers=2,
        )

        ####################################################
        # High-frequency Transformer
        ####################################################

        high_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=self.latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.high_transformer = nn.TransformerEncoder(
            high_layer,
            num_layers=2,
        )

        ####################################################
        # Inter-frequency fusion MLP
        ####################################################

        self.inter_mlp = nn.Sequential(
            nn.Linear((self.level + 1) * self.latent_dim,
                      self.latent_dim * 2),
            nn.GELU(),
            nn.Linear(self.latent_dim * 2,
                      self.latent_dim),
        )

        ####################################################
        # Inter-frequency Transformer
        ####################################################

        inter_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=self.latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.inter_transformer = nn.TransformerEncoder(
            inter_layer,
            num_layers=2,
        )

        ####################################################
        # Decoder for low-frequency latent
        ####################################################

        self.low_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Linear(
                self.latent_dim * 2,
                self.num_joints * self.latent_dim,
            ),
        )

        ####################################################
        # Decoder for high-frequency latent
        ####################################################

        self.high_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Linear(
                self.latent_dim * 2,
                self.num_joints * self.latent_dim,
            ),
        )

        ####################################################
        # Decoder for inter-frequency latent
        ####################################################

        self.inter_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Linear(
                self.latent_dim * 2,
                self.num_joints * self.latent_dim,
            ),
        )

    def forward(self, x):
        ####################################################
        # Input
        ####################################################
        # x : [N, T, V, C]
        ####################################################

        N, T, V, C = x.shape

        # print("=" * 60)
        # print(f"Input              : {x.shape}")

        ####################################################
        # Prepare for SWT
        ####################################################
        # Each (joint, latent channel) is treated as an
        # independent temporal signal.
        #
        # [N,T,V,C]
        #      ↓
        # [N,V,C,T]
        #      ↓
        # [N*V,C,T]
        ####################################################

        x_wave = (
            x.permute(0, 2, 3, 1)
            .reshape(N * V, C, T)
        )

        # print(f"x_wave             : {x_wave.shape}")

        ####################################################
        # Stationary Wavelet Transform
        ####################################################

        coeffs = ptwt.swt(
            x_wave,
            wavelet="haar",
            level=self.level,
            axis=-1
        )

        approx = coeffs[0]
        details = coeffs[1:]

        # print(f"Approximation      : {approx.shape}")

        # for i, d in enumerate(details):
            # print(f"Detail Level {i+1}   : {d.shape}")

        ####################################################
        # Recover low-frequency tensor
        #
        # [N*V,C,T]
        #      ↓
        # [N,V,C,T]
        #      ↓
        # [N,T,V,C]
        ####################################################

        x_low = (
            approx.reshape(N, V, C, T)
                .permute(0, 3, 1, 2)
        )

        # print(f"x_low              : {x_low.shape}")

        ####################################################
        # Recover high-frequency tensor
        #
        # stack ->
        # [S,N*V,C,T]
        #
        # reshape ->
        # [S,N,V,C,T]
        #
        # permute ->
        # [N,S,T,V,C]
        ####################################################

        x_high = torch.stack(details, dim=0)

        # print(f"x_high stacked     : {x_high.shape}")

        x_high = (
            x_high.reshape(
                self.level,
                N,
                V,
                C,
                T
            )
            .permute(1, 0, 4, 2, 3)
        )

        # print(f"x_high             : {x_high.shape}")

        ####################################################
        # LOW-FREQUENCY BRANCH
        ####################################################

        low_freq_wave = x_low.reshape(
            N,
            T,
            V * C
        )

        # print(f"Low reshape        : {low_freq_wave.shape}")

        low_freq_wave = low_freq_wave.transpose(1, 2)

        # print(f"Low transpose      : {low_freq_wave.shape}")

        low_freq_wave = self.low_conv(low_freq_wave)

        # print(f"Low Conv1D         : {low_freq_wave.shape}")

        low_freq_wave = low_freq_wave.transpose(1, 2)

        # print(f"Low transpose back : {low_freq_wave.shape}")

        ####################################################
        # HIGH-FREQUENCY BRANCH
        ####################################################

        S = self.level

        high_freq_wave = x_high.reshape(
            N * S,
            T,
            V * C
        )

        # print(f"High reshape       : {high_freq_wave.shape}")

        high_freq_wave = high_freq_wave.transpose(1, 2)

        # print(f"High transpose     : {high_freq_wave.shape}")

        high_freq_wave = self.high_conv(high_freq_wave)

        # print(f"High Conv1D        : {high_freq_wave.shape}")

        high_freq_wave = high_freq_wave.transpose(1, 2)

        # print(f"High transpose back: {high_freq_wave.shape}")

        high_freq_wave = high_freq_wave.reshape(
            N,
            S,
            T,
            self.latent_dim
        )

        # print(f"High final         : {high_freq_wave.shape}")

        ####################################################
        # Intra-Frequency MLP
        ####################################################

        low_freq_wave = self.mlp(low_freq_wave)

        # print(f"Low MLP            : {low_freq_wave.shape}")

        # [N,S,T,D]
        high_freq_wave = self.mlp(high_freq_wave)

        # print(f"High MLP           : {high_freq_wave.shape}")

        ####################################################
        # Intra-Frequency Transformer
        ####################################################

        low_freq_wave = self.low_transformer(low_freq_wave)

        # print(f"Low Transformer    : {low_freq_wave.shape}")

        ####################################################
        # High-frequency Transformer
        ####################################################

        N, S, T, D = high_freq_wave.shape

        high_freq_wave = high_freq_wave.reshape(
            N * S,
            T,
            D
        )

        # print(f"High reshape       : {high_freq_wave.shape}")

        high_freq_wave = self.high_transformer(
            high_freq_wave
        )

        # print(f"High Transformer   : {high_freq_wave.shape}")

        high_freq_wave = high_freq_wave.reshape(
            N,
            S,
            T,
            D
        )

        # print(f"High restore       : {high_freq_wave.shape}")

        ####################################################
        # Inter-Frequency Fusion
        ####################################################
        #
        # M_low : [N,T,D]
        # M_high: [N,S,T,D]
        #
        # concatenate along feature dimension
        #
        ####################################################

        high_multi = high_freq_wave.permute(
            0,
            2,
            1,
            3
        )

        # print(f"High permute       : {high_multi.shape}")

        high_multi = high_multi.reshape(
            N,
            T,
            S * D
        )

        # print(f"High flatten       : {high_multi.shape}")

        multi = torch.cat(
            [
                low_freq_wave,
                high_multi
            ],
            dim=-1
        )

        # print(f"Multi concat       : {multi.shape}")

        ####################################################
        # Inter-Frequency MLP
        ####################################################

        multi = self.inter_mlp(multi)

        # print(f"Inter MLP          : {multi.shape}")

        ####################################################
        # Inter-Frequency Transformer
        ####################################################

        multi = self.inter_transformer(multi)

        # print(f"Inter Transformer  : {multi.shape}")

        ###################################################
        # Decode Low-frequency representation
        ####################################################

        low_rec = self.low_decoder(low_freq_wave)

        # print(f"Low decoder        : {low_rec.shape}")

        low_rec = low_rec.reshape(
            N,
            T,
            V,
            C
        )

        # print(f"Low reconstruction : {low_rec.shape}")

        ####################################################
        # Decode High-frequency representation
        ####################################################

        high_rec = self.high_decoder(high_freq_wave)

        # print(f"High decoder       : {high_rec.shape}")

        high_rec = high_rec.reshape(
            N,
            S,
            T,
            V,
            C
        )

        # print(f"High reconstruction: {high_rec.shape}")

        ####################################################
        # Decode Inter-frequency representation
        ####################################################

        inter_rec = self.inter_decoder(multi)

        # print(f"Inter decoder      : {inter_rec.shape}")

        inter_rec = inter_rec.reshape(
            N,
            T,
            V,
            C
        )

        # print(f"Inter reconstruction: {inter_rec.shape}")

        ####################################################
        # Prepare Approximation
        ####################################################

        approx = (
            low_rec
            .permute(0, 2, 3, 1)
            .reshape(
                N * V,
                C,
                T
            )
        )

        # print(f"ISWT approx        : {approx.shape}")
        
        ####################################################
        # Prepare Details
        ####################################################

        details = []

        for s in range(S):

            d = (
                high_rec[:, s]
                .permute(0, 2, 3, 1)
                .reshape(
                    N * V,
                    C,
                    T
                )
            )

            # print(f"ISWT detail {s}     : {d.shape}")

            details.append(d)

        ####################################################
        # Inverse SWT
        ####################################################

        coeffs = [approx] + details

        recon = ptwt.iswt(
            coeffs,
            wavelet="haar",
            axis=-1
        )

        # print(f"ISWT output        : {recon.shape}")

        ####################################################
        # Recover latent tensor
        ####################################################

        recon = (
            recon.reshape(
                N,
                V,
                C,
                T
            )
            .permute(0, 3, 1, 2)
        )

        # print(f"Final reconstruction: {recon.shape}")

        return recon
    
        """
        return {
        "low_feature": low_freq_wave,
        "high_feature": high_freq_wave,
        "multi_feature": multi,
        "low_reconstruction": low_rec,
        "high_reconstruction": high_rec,
        "inter_reconstruction": inter_rec,
        "wavelet_reconstruction": recon,
        }
        """