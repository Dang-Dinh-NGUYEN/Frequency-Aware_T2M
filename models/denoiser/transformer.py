import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from models.denoiser.embedding import RotaryEmbedding, apply_rope
from models.skeleton.conv import STConv

# def _get_core_adj_matrix():
#     out = torch.zeros(7, 7, dtype=torch.float32)
#     out[0, [1, 2, 3]] = 1
#     out[1, 0] = 1
#     out[2, 0] = 1
#     out[3, [0, 4, 5, 6]] = 1
#     out[4, 3] = 1
#     out[5, 3] = 1
#     out[6, 3] = 1
#     return out

def _get_core_edges():
    return [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)]

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return x * (scale + 1) + shift


class DenseFiLM(nn.Module):
    def __init__(self, opt):
        super(DenseFiLM, self).__init__()
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim * 2),
        )

    def forward(self, cond):
        """
        cond: [B, D]
        """
        cond = self.linear(cond)
        cond = cond[:, None, None, :] # unsqueeze for skeleto-temporal dimensions
        scale, shift = cond.chunk(2, dim=-1)
        return scale, shift


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, batch_first=True, use_rope=False):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model); self.Wo = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, average_attn_weights=False):
        B, T1, D = query.size(); _, T2, _ = key.size()
        query = self.Wq(query).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)
        key   = self.Wk(key).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            assert T1 == T2, "RoPE here assumes self-attention (T1 == T2)"
            cos, sin = self.rope(T1, query.device, query.dtype)
            query, key = apply_rope(query, key, cos, sin)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            # dtype-safe fill value: -1e9 overflows fp16 under autocast
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :],
                                                    torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)
        attn_output = self.Wo(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        return attn_output, None
    
    def forward_with_fixed_attn_weights(self, attn_weights, value):
        """
        Assume that the attention weights are already computed.
        """
        B, H, _, T2 = attn_weights.size()
        D = value.size(-1)

        # linear transformation
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        return attn_output, attn_weights


class FreqAttention(nn.Module):
    def __init__(self, opt):
        super().__init__()
        D = opt.latent_dim
        self.dim = D
        self.norm = nn.LayerNorm(D)
        self.band_emb_type = getattr(opt, "freq_band_emb", "sinusoidal")
        if self.band_emb_type not in ("none", "sinusoidal", "sinusoidal_norm"):
            raise ValueError(f"freq_band_emb must be 'none', 'sinusoidal' or 'sinusoidal_norm', "
                            f"got {self.band_emb_type!r}")
        # position scale for the normalized embedding: k/L * scale.
        # Set to the maximum sequence length the denoiser sees.
        self.band_pos_scale = float(getattr(opt, "freq_band_pos_scale", 49.0))
        self.attn = MultiheadAttention(D, opt.n_heads, opt.dropout, batch_first=True)
        self.dropout = nn.Dropout(opt.dropout)

    @staticmethod
    def _dct_matrices(lengths, T, device):
        """
        Orthonormal DCT-II matrices, one per row, each sized to that row's valid length.
        lengths: [N] valid frames per row.  Returns C: [N, T, T] with
            C[i, k, n] = sqrt(2/L) * cos(pi * (n + 0.5) * k / L)   (row k = 0 scaled by 1/sqrt(2))
        for k, n < L_i, and 0 elsewhere. X = C @ x is the DCT; x = C^T @ X is its inverse.
        Same convention as torch_dct.dct(..., norm="ortho") when L == T.
        """
        k = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)
        n = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T)
        L = lengths.to(device=device, dtype=torch.float32).clamp(min=1).view(-1, 1, 1)
        C = torch.cos(math.pi * (n + 0.5) * k / L) * torch.sqrt(2.0 / L)
        C = torch.where(k == 0, C / math.sqrt(2.0), C)
        inside = (k < L) & (n < L)
        return C * inside

    @staticmethod
    def _sinusoidal(K, D, device):
        """Standard sin/cos positional encoding for band indices 0..K-1 -> [K, D]."""
        k = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(1)          # [K,1]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / D))                                   # [D/2]
        pe = torch.zeros(K, D, device=device)
        pe[:, 0::2] = torch.sin(k * div)
        pe[:, 1::2] = torch.cos(k * div[: D // 2])
        return pe

    @staticmethod
    def _sinusoidal_normalized(lengths, T, D, scale, device):
        """
        Sin/cos encoding of the normalized band frequency k / L_i, per sample.
        lengths: [N] valid frames per row. Returns [N, T, D].
        Band k of a length-L DCT has frequency k / (2L) cycles per frame, so equal
        k / L means equal physical frequency, whatever the sequence length.
        """
        k = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)          # [1,T,1]
        L = lengths.to(device=device, dtype=torch.float32).clamp(min=1).view(-1, 1, 1)  # [N,1,1]
        pos = k / L * scale                                                             # [N,T,1]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / D))                                     # [D/2]
        pe = torch.zeros(lengths.shape[0], T, D, device=device)
        pe[..., 0::2] = torch.sin(pos * div)
        pe[..., 1::2] = torch.cos(pos * div[: D // 2])
        return pe

    def forward(self, x, cond, pad_mask=None, need_weights=False):
        """
        x:        [N, T, D], N = B*J, rows ordered b*J + j. No maximum on T.
        cond:     [B, D] timestep embedding -- unused here, kept so the call
                  site needs no change if a conditioned variant is added later.
        pad_mask: [N, T], True == padded frame (the temporal attention mask).
                  A sample with L valid frames has exactly L real bands;
                  bands >= L are zeroed before attention and again before the
                  inverse DCT, and are also excluded as attention keys via
                  key_padding_mask (same convention _ta_block uses).
        Returns (y, attn): attn is the band-attention map when need_weights,
        else None.
        """
        N, T, D = x.shape
        out_dtype = x.dtype

        if pad_mask is None:
            pad_mask = torch.zeros(N, T, dtype=torch.bool, device=x.device)
        valid = (~pad_mask).unsqueeze(-1).float()                  # [N,T,1]
        lengths = (~pad_mask).sum(dim=1)                           # [N]

        # fp32 for the transform (stable under autocast)
        C = self._dct_matrices(lengths, T, x.device)                # [N,T,T]
        X = torch.bmm(C, x.float() * valid)                         # [N,T,D], bands >= L are 0
        X = self.norm(X)
        if self.band_emb_type == "sinusoidal":
            X = X + self._sinusoidal(T, D, X.device)                                   # [T,D]
        elif self.band_emb_type == "sinusoidal_norm":
            X = X + self._sinusoidal_normalized(lengths, T, D,
                                                self.band_pos_scale, X.device)                   # [T,D], broadcasts over N

        Y, attn = self.attn(X, X, X, key_padding_mask=pad_mask,
                            need_weights=need_weights, average_attn_weights=False)
        Y = Y.float()

        Y = Y * valid                                               # zero bands >= L before inverting
        y = torch.bmm(C.transpose(1, 2), Y)                         # inverse DCT over this sample's length
        y = self.dropout(y)
        return y.to(out_dtype), attn


class BandMixer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.K = getattr(opt, "freq_max_bands", 64)
        hidden = getattr(opt, "freq_mixer_hidden", self.K)
        self.mix1 = nn.Linear(self.K, hidden)
        self.act = nn.GELU()
        self.mix2 = nn.Linear(hidden, self.K)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x, mask=None):
        """
        x:    [N, T, D]
        mask: [N, T], True == padded band. T <= self.K is required (raise
              opt.freq_max_bands if a longer motion appears).
        Returns (y, None): no attention weights to report.
        """
        N, T, D = x.shape
        if T > self.K:
            raise ValueError(f"sequence has {T} bands but freq_max_bands={self.K}; "
                             f"raise opt.freq_max_bands")
        valid = None
        if mask is not None:
            valid = (~mask).unsqueeze(-1).to(x.dtype)       # [N,T,1]
            x = x * valid

        Xt = x
        if T < self.K:
            Xt = F.pad(Xt, (0, 0, 0, self.K - T))            # [N,K,D]
        Xt = Xt.transpose(1, 2)                               # [N,D,K]
        Y = self.mix2(self.act(self.mix1(Xt)))                # token-mixing MLP along bands
        Y = Y.transpose(1, 2)[:, :T]                          # [N,T,D], discard bands >= T

        if valid is not None:
            Y = Y * valid
        Y = self.dropout(Y)
        return Y, None


class STTransformerLayer(nn.Module):
    """
    Setting
        - Normalization first
        - Sub-block order: temporal -> skeletal -> [frequency] -> cross -> FFN.
          [frequency] is sequential by default (reads x after temporal+skeletal
          have updated it) or, with opt.freq_attn_parallel=True, parallel to
          the temporal+skeletal combo: it reads the layer's own input instead,
          independent of what temporal/skeletal did this layer, and its output
          is simply summed in alongside theirs. Code-execution order is the
          same either way (frequency still runs third); only which tensor it
          reads changes. See STTransformerLayer.forward.
    """
    def __init__(self, opt, is_middle=False):
        """
        is_middle: True for the one STTransformerLayer built as SkipTransformer's
            middle_block, False for every input/output block. Only used to decide
            whether this layer gets freq_attn when opt.freq_attn_middle_only=True
            (see below) -- it has no other effect.
        """
        super(STTransformerLayer, self).__init__()
        self.opt = opt

        # skeletal attention
        self.skel_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # temporal mixing: full self-attention over time (default), or, when the
        # backbone is already running on DCT coefficients (freq_aware_denoising)
        # and freq_temporal_mixer=="mlp", an MLP-Mixer over the band axis instead
        # -- see BandMixer's docstring for why that fits the frequency-domain
        # case better than attention does.
        self.freq_aware_denoising = getattr(opt, "freq_aware_denoising", False)
        self.temporal_mixer = getattr(opt, "freq_temporal_mixer", "attention")
        self.use_band_mixer = self.freq_aware_denoising and self.temporal_mixer == "mlp"
        if self.use_band_mixer:
            self.temp_attn = BandMixer(opt)
        else:
            self.temp_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True, use_rope=getattr(opt, "temporal_rope", False))
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # frequency attention (sequential, right after skeletal attention --
        # see the forward()/class docstring for the full sub-block order)
        #
        # opt.freq_attn_middle_only restricts this sub-block to the one middle
        # layer instead of every input/output block (default: every layer, as
        # before). This is a pure capacity dial, not a new mechanism: freq_attn
        # is an additive, zero-init branch (see below), not a replacement for
        # anything, so it's fine for most layers to simply not have it. Cuts
        # both parameter count and per-sample perturbation surface by roughly
        # n_layers-fold once training moves the branch off zero-init, directly
        # targeting the overfitting read from the earlier train/generation
        # mismatch -- cheaper than re-architecting the sub-block again.
        self.use_freq_attn = getattr(opt, "freq_attn", False) and (
            is_middle or not getattr(opt, "freq_attn_middle_only", False))
        print(f"using freq_attn {self.use_freq_attn}")
        # opt.freq_attn_parallel (default False, preserves old checkpoints'
        # behavior exactly): False keeps freq_attn sequential -- it reads x
        # AFTER temporal+skeletal have already updated it this layer, as
        # before. True makes it read the layer's own input instead, the same
        # input temporal+skeletal attention see, so it is computed
        # independently of what they do this layer and its contribution is
        # simply summed in alongside theirs -- parallel to the spatio-temporal
        # combo rather than built on top of its output. See forward().
        self.freq_attn_parallel = getattr(opt, "freq_attn_parallel", False)
        if self.use_freq_attn:
            self.freq_attn = FreqAttention(opt)
            self.freq_film = DenseFiLM(opt)

            # Conservative residual injection: start close to the original
            # time-domain denoiser, but keep gradients flowing through the branch.
            self.freq_res_scale = nn.Parameter(torch.tensor(1e-3))

            # Start the frequency FiLM at identity (scale=shift=0).
            # This does not zero the frequency output; freq_res_scale controls
            # the initial contribution independently.
            final = self.freq_film.linear[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        # cross attention
        self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_dropout = nn.Dropout(opt.dropout)

        # ffn
        self.ffn_linear1 = nn.Linear(opt.latent_dim, opt.ff_dim)
        self.ffn_linear2 = nn.Linear(opt.ff_dim, opt.latent_dim)
        self.ffn_norm = nn.LayerNorm(opt.latent_dim)
        self.ffn_dropout = nn.Dropout(opt.dropout)

        # activation
        self.act = F.relu if opt.activation == "relu" else F.gelu

        # FiLM
        self.skel_film = DenseFiLM(opt)
        self.temp_film = DenseFiLM(opt)
        self.cross_film = DenseFiLM(opt)
        self.ffn_film = DenseFiLM(opt)

    def _sa_block(self, x, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.skel_dropout(x)
        return x, attn

    def _ta_block(self, x, mask=None, fixed_attn=None):
        x = self.temp_norm(x)
        if self.use_band_mixer:
            if fixed_attn is not None:
                raise ValueError("fixed_ta replay is not supported when freq_temporal_mixer=='mlp': "
                                 "the band mixer has no attention map to replay.")
            x, attn = self.temp_attn(x, mask=mask)
        elif fixed_attn is None:
            x, attn = self.temp_attn.forward(x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.temp_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.temp_dropout(x)
        return x, attn

    def _ca_block(self, x, mem, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(fixed_attn, mem)
        x = self.cross_dropout(x)
        return x, attn
    
    def _ff_block(self, x):
        x = self.ffn_norm(x)
        x = self.ffn_linear1(x)
        x = self.act(x)
        x = self.ffn_linear2(x)
        x = self.ffn_dropout(x)
        return x



    def forward(self, x, memory, cond, x_mask=None, memory_mask=None,
                skel_attn=None, temp_attn=None, cross_attn=None):
        """
        x_mask: [B*J, T], True == padded frame (temporal padding mask)
        """
        B, T, J, D = x.size()

        # diffusion timestep embedding
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # saved for the frequency branch when freq_attn_parallel=True: the
        # layer's own input, before temporal/skeletal touch it at all.
        x_in = x

        # temporal attention
        ta_out, ta_weight = self._ta_block(x.transpose(1, 2).reshape(B * J, T, D),
                                            mask=x_mask,
                                            fixed_attn=temp_attn)
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # skeletal attention
        sa_out, sa_weight = self._sa_block(x.reshape(B * T, J, D),
                                            fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out

        # frequency attention: sequential (default) reads x as it stands after
        # temporal+skeletal this layer, same as before. parallel reads x_in --
        # the layer's own input, independent of this layer's temporal/skeletal
        # update -- and its contribution is summed in alongside theirs rather
        # than built on top of their output.
        if self.use_freq_attn:
            freq_in = x_in if self.freq_attn_parallel else x
            fa_out, _ = self.freq_attn(freq_in.transpose(1, 2).reshape(B * J, T, D),
                                       cond, pad_mask=x_mask)
            fa_out = fa_out.reshape(B, J, T, D).transpose(1, 2)
            fa_out = featurewise_affine(fa_out, self.freq_film(cond))
            x = x + self.freq_res_scale * fa_out

        # cross attention
        ca_out, ca_weight = self._ca_block(x.reshape(B, T * J, D),
                                        memory,
                                        mask=memory_mask,
                                        fixed_attn=cross_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)

        return x, attn_weights

class SkipTransformer(nn.Module):
    def __init__(self, opt):
        super(SkipTransformer, self).__init__()
        self.opt = opt
        if self.opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for SkipTransformer, but got {self.opt.n_layers}")
        if getattr(opt, "freq_attn", False) and getattr(opt, "freq_aware_denoising", False):
            raise ValueError("freq_attn and freq_aware_denoising are alternatives: with both on, "
                             "the frequency attention would take a DCT of DCT coefficients.")
        if getattr(opt, "freq_temporal_mixer", "attention") not in ("attention", "mlp"):
            raise ValueError("opt.freq_temporal_mixer must be 'attention' or 'mlp', got "
                             f"{opt.freq_temporal_mixer!r}")
        if getattr(opt, "freq_temporal_mixer", "attention") == "mlp" and not getattr(opt, "freq_aware_denoising", False):
            raise ValueError("freq_temporal_mixer='mlp' only makes sense when freq_aware_denoising=True "
                             "-- it mixes the band axis the backbone receives in that mode, and there is "
                             "no fixed band axis to mix over in time domain.")

        # transformer encoder
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(opt, is_middle=False)
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for i in range((self.opt.n_layers - 1) // 2):
            self.input_blocks.append(STTransformerLayer(opt, is_middle=False))
            self.output_blocks.append(STTransformerLayer(opt, is_middle=False))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))
        

    def forward(self, x, timestep_emb, word_emb, sa_mask=None, ca_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None):
        """
        x: [B, T, J, D]
        timestep_emb: [B, D]
        word_emb: [B, N, D]
        sa_mask: [B*J, T] temporal padding mask, True == pad (name kept for compatibility)
        ca_mask: [B, N]

        fixed_sa: [bsz*nframes, nlayers, nheads, njoints, njoints]
        fixed_ta: [bsz*njoints, nlayers, nheads, nframes, nframes]
        fixed_ca: [bsz, nlayers, nheads, nframes*njoints, dclip]

        fixed_ta is unsupported (raises) when opt.freq_temporal_mixer=='mlp':
        the band mixer has no attention map to replay. need_attn=True then
        also returns None for the temporal slot of attn_weights, for the same
        reason -- there is nothing to stack across layers.
        """
        # B, T, J, D = x.size()
        
        xs = []

        attn_weights = [[], [], []]
        layer_idx = 0
        for i, block in enumerate(self.input_blocks):
            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca)
            xs.append(x)
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1
        
        sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
        ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
        ca = None if fixed_ca is None else fixed_ca[:, layer_idx]
        x, attns = self.middle_block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                     skel_attn=sa, temp_attn=ta, cross_attn=ca)
        
        for j in range(len(attn_weights)):
            attn_weights[j].append(attns[j])
        layer_idx += 1

        for (block, skip) in zip(self.output_blocks, self.skip_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = skip(x)

            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca)
            
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1

        if need_attn:
            for j in range(len(attn_weights)):
                # the temporal slot (index 1) holds None per layer when
                # freq_temporal_mixer=='mlp' -- the band mixer has no
                # attention map, so there is nothing to stack for it.
                if any(a is None for a in attn_weights[j]):
                    attn_weights[j] = None
                else:
                    attn_weights[j] = torch.stack(attn_weights[j], dim=1)
        else:
            for j in range(len(attn_weights)):
                attn_weights[j] = None

        return x, attn_weights, None