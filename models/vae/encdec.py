import math

import torch
import torch.nn as nn

import torch_dct as dct
from models.skeleton.conv import ResSTConv, get_activation
from models.skeleton.pool import STPool, STUnpool
from utils.paramUtil import kit_adj_list, t2m_adj_list
from utils.skeleton import adj_list_to_edges


class MotionEncoder(nn.Module):
    def __init__(self, opt):
        super(MotionEncoder, self).__init__()

        self.pose_dim = opt.pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0:
                input_dim = 7
            elif i in self.contact_joints:
                input_dim = 13
            else:
                input_dim = 12
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, self.latent_dim),
            ))

    def forward(self, x):
        """
        x: [bs, nframes, pose_dim]
        
        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, D = x.size()

        # split
        root, ric, rot, vel, contact = torch.split(x, [4, 3 * (self.joints_num - 1), 6 * (self.joints_num - 1), 3 * self.joints_num, 4], dim=-1)
        ric = ric.reshape(B, T, self.joints_num - 1, 3)
        rot = rot.reshape(B, T, self.joints_num - 1, 6)
        vel = vel.reshape(B, T, self.joints_num, 3)

        # joint-wise input
        joints = [torch.cat([root, vel[:, :, 0]], dim=-1)] # [B, T, 7]]
        for i in range(1, self.joints_num):
            joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1))
        for cidx, jidx in enumerate(self.contact_joints):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)
        
        # encode
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](joints[i]))
        out = torch.stack(out, dim=2)

        return out


class MotionDecoder(nn.Module):
    def __init__(self, opt):
        super(MotionDecoder, self).__init__()
        
        self.pose_dim = opt.pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        # network components
        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0:
                output_dim = 7
            elif i in self.contact_joints:
                output_dim = 13
            else:
                output_dim = 12
            self.layers.append(nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, output_dim),
            ))

    def forward(self, x):
        """
        x: [bs, nframes, joints_num, latent_dim]
        """
        B, T, J, D = x.size()
        
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](x[:, :, i]))
        
        root = out[0]
        ric_list, rot_list, vel_list = [], [], []
        for i in range(1, self.joints_num):
            ric = out[i][:, :, :3]
            rot = out[i][:, :, 3:9]
            vel = out[i][:, :, 9:12]

            ric_list.append(ric)
            rot_list.append(rot)
            vel_list.append(vel)

        contact = [out[i][:, :, -1] for i in self.contact_joints]

        ric = torch.stack(ric_list, dim=2).reshape(B, T, (J - 1) * 3)
        rot = torch.stack(rot_list, dim=2).reshape(B, T, (J - 1) * 6)
        vel = torch.stack(vel_list, dim=2).reshape(B, T, (J - 1) * 3)
        contact = torch.stack(contact, dim=2).reshape(B, T, len(self.contact_joints))

        motion = torch.cat([
            root[..., :4], # root
            ric, # ric
            rot, # rot
            torch.cat([root[..., 4:], vel], dim=-1), # vel
            contact, # contact
        ], dim=-1)

        return motion


class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()

        # adjacency list
        self.adj_list = {
            "t2m": t2m_adj_list,
            "kit": kit_adj_list,
        }[opt.dataset_name]

        # topology
        self.edge_list = [adj_list_to_edges(self.adj_list)]
        self.mapping_list = []

        # network
        self.layers = nn.ModuleList()
        for i in range(opt.n_layers):
            layers = []
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                self.edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))

            pool = STPool(opt.dataset_name, i)
            layers.append(pool)
            self.layers.append(nn.Sequential(*layers))

            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class STConvDecoder(nn.Module):
    def __init__(self, opt, encoder: STConvEncoder):
        super(STConvDecoder, self).__init__()

        # network modules
        self.layers = nn.ModuleList()

        # build network
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for i in range(opt.n_layers):
            layers = []

            # unpooling
            layers.append(STUnpool(skeleton_mapping=mapping_list.pop()))

            # conv
            edges = edge_list.pop()
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))

            self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        x: [B, T, J_in, D]
        out: [B, T, J_out, D]
        """
        for layer in self.layers:
            x = layer(x)
        return x


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))

class DCTEnhancedModule(nn.Module):
    def __init__(self, opt, num_joints=7, clamp_min=None, clip_high_enhance=False):
        super().__init__()
        self.opt = opt
        self.num_joints = num_joints
        self.clamp_min = clamp_min
        self.clip_high_enhance = clip_high_enhance
        self.learn_threshold = getattr(opt, "learn_threshold", False)
 
        self.num_bands = int(getattr(opt, "num_bands", opt.b))
        self.partition_type = getattr(opt, "partition_type", "linear")
        self.soft_width = max(getattr(opt, "threshold_softness", 0.05), 0.02)
 
        tau = float(opt.freq_threshold)
        beta = float(getattr(opt, "b_ratio", tau * opt.window_size))

        self.beta = beta
 
        # Threshold: fixed by default (FS-VAE does not learn it). If learned, it is
        # stored as a logit initialised so that sigmoid(param) == tau exactly.
        if self.learn_threshold:
            self.thr_logit = nn.Parameter(torch.tensor(_logit(tau)))
        else:
            self.register_buffer("thr_logit", torch.tensor(_logit(tau)))
 
        self.k = nn.Parameter(torch.tensor(float(opt.k)))
        self.freq_weights = nn.Parameter(torch.ones(num_joints, self.num_bands))
 
        self._bin_cache = {}
 
    # ------------------------------------------------------------------
    def build_bins(self, T, device):
        freqs = torch.arange(T, device=device, dtype=torch.float32) / T
        if self.partition_type == "linear":
            edges = torch.linspace(0, 1, self.num_bands + 1, device=device)
        elif self.partition_type == "log":
            alpha = 100.0
            edges = torch.linspace(0, math.log1p(alpha), self.num_bands + 1, device=device)
            edges = (torch.exp(edges) - 1) / alpha
            edges[0], edges[-1] = 0.0, 1.0
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}")
        return torch.bucketize(freqs, edges[1:-1])
 
    def get_bins(self, T, device):
        key = (T, str(device))
        if key not in self._bin_cache:
            self._bin_cache[key] = self.build_bins(T, device)
        return self._bin_cache[key]
 
    @property
    def threshold(self):
        return torch.sigmoid(self.thr_logit)
 
    # ------------------------------------------------------------------
    def band_gains(self):
        """g(i) for every body part and band: [V, B]."""
        B = self.num_bands
        device = self.freq_weights.device
        p = torch.arange(B, device=device, dtype=torch.float32) / B
        tau = self.threshold.float()
        beta = self.beta
 
        kw = self.k * self.freq_weights                       # [V, B]
 
        g_low = 1.0 + kw * (1.0 - p / beta)
        atten = 1.0 - (p - beta) / beta
        g_high = 1.0 - kw * atten
 
        # Band i is high iff p_i >= tau. Exact in the forward pass; if the
        # threshold is learnable, a soft straight-through gradient reaches it.
        hard = (p >= tau - 1e-6).float()
        soft = torch.sigmoid((p - tau + 0.5 / B) / self.soft_width)
        blend = hard + soft - soft.detach()
 
        g = (1.0 - blend) * g_low + blend * g_high
        if self.clamp_min is not None:
            g = torch.clamp(g, min=self.clamp_min)
        return g
 
    # ------------------------------------------------------------------
    def forward(self, x):
        """x: [N, T, V, C] -> [N, T, V, C]"""
        N, T, V, C = x.shape
        x = x.permute(0, 2, 3, 1)                             # [N, V, C, T]
        bin_indices = self.get_bins(T, x.device)
 
        x_dct = dct.dct(x.float(), norm="ortho")

        gains = self.band_gains()                         # [V, B]
        mask = gains[:, bin_indices][None, :, None, :]    # [1, V, 1, T]

        x_dct = x_dct * mask

        x_out = dct.idct(x_dct, norm="ortho")
 
        return x_out.permute(0, 3, 1, 2).to(x.dtype) 
