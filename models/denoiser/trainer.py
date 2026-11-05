import os
from os.path import join as pjoin
import time
import numpy as np
from collections import OrderedDict, defaultdict
from typing import List, Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn as nn
from utils.ddp import reduce_loss

from utils.eval_t2m import evaluation_denoiser, test_denoiser,test_denoiser_composed
from utils.utils import print_current_loss, attn2img
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor 

import torch_dct as dct

def def_value():
    return 0.0

def lengths_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    max_frames = torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask

class PartEncoder(nn.Module):
    """q(z_part | x0): joint over slots (coordination), bottlenecked (anti-leakage).
    x0: [B, T, 7, C] clean SALAD latent -> z_part: [B, 7, d] + (mu, logvar)."""
    def __init__(self, C, n_slots=7, d=8, hidden=128):
        super().__init__()
        self.n_slots, self.d = n_slots, d
        # per-slot pooled stats: mean & std over time -> [B,7,2C]. Deliberately
        # discards the full trajectory so q CANNOT encode the whole motion.
        self.in_proj = nn.Linear(2 * C, hidden)
        # cross-slot attention: lets arm-code see torso-code (coordination)
        self.slot_attn = nn.TransformerEncoderLayer(hidden, nhead=4,
                          dim_feedforward=hidden*2, batch_first=True)
        self.to_mu     = nn.Linear(hidden, d)
        self.to_logvar = nn.Linear(hidden, d)

    def forward(self, x0, len_mask=None):          # x0: [B,T,7,C]
        if len_mask is not None:                   # masked temporal pooling
            m = len_mask[:, :, None, None].float()
            mean = (x0*m).sum(1) / m.sum(1).clamp(min=1)          # [B,7,C]
            var  = ((x0-mean[:,None])**2 * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            mean, var = x0.mean(1), x0.var(1)
        stats = torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)  # [B,7,2C]
        h = self.in_proj(stats)                    # [B,7,hidden]
        h = self.slot_attn(h)                      # coordinated across slots
        mu, logvar = self.to_mu(h), self.to_logvar(h)             # [B,7,d]
        z = mu + torch.randn_like(mu) * (0.5*logvar).exp()        # reparam
        return z, mu, logvar

def kl_per_slot(mu, logvar):                       # standard VAE KL, summed over d, mean over B,slots
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()

"""
def predict_x0(pred, noisy_latent, timesteps, scheduler, prediction_type):
    if prediction_type == "sample":
        return pred
    alphas_cumprod = scheduler.alphas_cumprod.to(noisy_latent.device)[timesteps]
    sqrt_ac = alphas_cumprod ** 0.5
    sqrt_1mac = (1 - alphas_cumprod) ** 0.5
    while sqrt_ac.dim() < noisy_latent.dim():
        sqrt_ac, sqrt_1mac = sqrt_ac.unsqueeze(-1), sqrt_1mac.unsqueeze(-1)
    if prediction_type == "epsilon":
        return (noisy_latent - sqrt_1mac * pred) / sqrt_ac
    elif prediction_type == "v_prediction":
        return sqrt_ac * noisy_latent - sqrt_1mac * pred
"""

class MaskedDCTLoss(nn.Module):
    """Equation (10), fixed: mask in the TIME domain (before the DCT), transform
    over the temporal axis, then weight each frequency explicitly. Uniform
    weights == masked time-domain MSE (Parseval); a non-uniform profile is what
    actually makes this 'frequency supervision'."""

    def __init__(self, freq_weight: str = "highboost", high_boost: float = 3.0):
        super().__init__()
        self.freq_weight = freq_weight
        self.high_boost = high_boost

    def _weights(self, T, device, dtype):
        if self.freq_weight == "uniform":
            return torch.ones(T, device=device, dtype=dtype)
        if self.freq_weight == "highboost":
            k = torch.arange(T, device=device, dtype=dtype) / max(T - 1, 1)
            return 1.0 + self.high_boost * k          # low freq -> 1, high freq -> 1+boost
        raise NotImplementedError(self.freq_weight)

    @staticmethod
    def _dct_T(x):                                     # DCT along dim=1 of [B,T,J,D]
        x = x.permute(0, 2, 3, 1)                      # [B,J,D,T]
        x = dct.dct(x, norm="ortho")
        return x.permute(0, 3, 1, 2)                   # [B,T,J,D], T = frequency

    def forward(self, pred_time, target_time, mask, reduction="none"):
        assert pred_time.ndim == 4, "expects [B, T, J, D]"
        m = mask[:, :, None, None].to(pred_time.dtype)          # mask in TIME
        P = self._dct_T(pred_time * m)
        Q = self._dct_T(target_time * m)
        T = P.shape[1]
        w = self._weights(T, P.device, P.dtype).view(1, T, 1, 1)
        err = ((P - Q) ** 2) * w
        feat = err.shape[2] * err.shape[3]                       # J * D_feat
        per_sample = err.flatten(1).sum(dim=1)                   # [B]
        denom = mask.sum(dim=1).clamp(min=1.0) * feat            # per-element count
        per_sample = per_sample / denom                          # [B], per-element scale
        if reduction == "none":
            return per_sample
        return per_sample.mean()

class DenoiserTrainer:
    def __init__(self, opt, denoiser, vae, scheduler):
        self.opt = opt
        self.rank = getattr(opt, 'rank', 0)
        self.denoiser = denoiser
        self.vae = vae
        self.noise_scheduler = scheduler
 
        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss()
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()
            elif opt.recon_loss == "l2":
                self.recon_criterion = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"Reconstruction loss {opt.recon_loss} not implemented")        
 
            if self.rank == 0:
                wandb.init(
                    project='t2m-project',
                    name=opt.name,
                    config=vars(opt) # Pass all training options as hyperparameters
                )
 
                wandb.define_metric("iter")
                wandb.define_metric("Train/*", step_metric="iter")
                wandb.define_metric("Val/*", step_metric="iter")
                wandb.define_metric("Eval/*", step_metric="iter")
 
                # self.logger = None # SummaryWriter(opt.log_dir)
            # else:
 
            self.logger = None
 
        if self.amp:
            self.scaler = GradScaler()
 
    def train_forward(self, batch_data):
        # setup input
        text, motion, m_lens = batch_data
 
        # random drop during training
        text = ["" if np.random.rand(1) < self.opt.cond_drop_prob else t for t in text]
 
        # to device
        motion = motion.to(self.opt.device, dtype=torch.float32)
        m_lens = m_lens.to(self.opt.device, dtype=torch.long)
        len_mask = lengths_to_mask(m_lens // 4) # [B, T]
 
        # latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion) # [B, T, J, D]
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()
 
        # sample diffusion timesteps
        timesteps = torch.randint(0, self.opt.num_train_timesteps,(latent.shape[0],),device=latent.device,).long()
 
        # add noise
        noise = torch.randn_like(latent) # [B, T, J, D]
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
 
        # predict the noise
        pred, attn_list, pred_dct_feat = self.denoiser.forward(noisy_latent, timesteps, text, len_mask=len_mask)
        pred = pred * len_mask[..., None, None].float()
 
        # loss
        loss_dict = {}
        if self.opt.prediction_type == "sample":
            main_target = latent
        elif self.opt.prediction_type == "epsilon":
            main_target = noise
        elif self.opt.prediction_type == "v_prediction":
            main_target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
        else:
            raise NotImplementedError(f"Prediction type {self.opt.prediction_type} not implemented")
 
        loss_recon = self.recon_criterion(pred, main_target)
        loss_dict[f"{self.opt.prediction_type}"] = loss_recon
        loss = loss_recon
        loss_dict["loss"] = loss
 
        return loss, attn_list, loss_dict
 
 
    @torch.no_grad()
    def generate(self, batch_data, need_attn=False, z_part=None, zpart_scale=1.0):
        part_composition = getattr(self.opt, "part_composition", False)
        
        self.denoiser.eval()
        model = self.denoiser.module if hasattr(self.denoiser, 'module') else self.denoiser
 
        text, motion, m_lens = batch_data
        motion = motion.to(self.opt.device, dtype=torch.float32)
        m_lens = m_lens.to(self.opt.device, dtype=torch.long) // 4
        len_mask = lengths_to_mask(m_lens)
 
        input_text = [""] * len(text)
        if self.opt.classifier_free_guidance:
            input_text.extend(text)
 
        z, _ = self.vae.encode(motion)
        latents = torch.randn_like(z) * self.noise_scheduler.init_noise_sigma
        
        len_mask = F.pad(len_mask, (0, latents.shape[1] - len_mask.shape[1]), value=False)
        mask4 = len_mask[..., None, None].float()
        latents = latents * mask4
 
        B, T = latents.shape[0], latents.shape[1]
 
        self.noise_scheduler.set_timesteps(self.opt.num_inference_timesteps)
        timesteps = self.noise_scheduler.timesteps.to(self.opt.device)

        skel_attn_weights, temp_attn_weights, cross_attn_weights = [], [], []
        for timestep in timesteps:
            if self.opt.classifier_free_guidance:
                inp = torch.cat([latents]*2, 0); im = torch.cat([len_mask]*2, 0)
            else:
                inp, im = latents, len_mask
 
            pred, attn, _ = model.forward(inp, timestep, input_text, len_mask=im,
                                    need_attn=False, use_cached_clip=True)
 
            if self.opt.classifier_free_guidance:
                pu, pc = torch.chunk(pred, 2, 0)
                pred = pu + self.opt.cond_scale * (pc - pu)
 
            latents = self.noise_scheduler.step(pred, timestep, latents).prev_sample * mask4

            skel_attn_weights.append(attn[0])
            temp_attn_weights.append(attn[1])
            cross_attn_weights.append(attn[2])

        pred_motion = self.vae.decode(latents)
        if isinstance(pred_motion, (tuple, list)): pred_motion = pred_motion[0]

        if need_attn:
            skel_attn_weights = torch.stack(skel_attn_weights, dim=1) # [bsz * nframes, ntimesteps, nlayers, nheads, njoints, njoints]
            temp_attn_weights = torch.stack(temp_attn_weights, dim=1) # [bsz * njoints, ntimesteps, nlayers, nheads, nframes, nframes]
            cross_attn_weights = torch.stack(cross_attn_weights, dim=1) # [bsz, ntimesteps, nlayers, nheads, nframes * njoints, dclip]
            attn_weights = (skel_attn_weights, temp_attn_weights, cross_attn_weights)
        else:
            attn_weights = (None, None, None)

        model.remove_clip_cache()
        return pred_motion, attn_weights


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def save(self, file_name, epoch, total_iter):
        if self.rank == 0:
            state = {
                "denoiser": self.denoiser.module.state_dict_without_clip() if hasattr(self.denoiser, 'module') else self.denoiser.state_dict_without_clip(),
                "optim": self.optim.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "total_iter": total_iter,
            }
            torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        
        if hasattr(self.denoiser, 'module'):
            missing_keys, unexpected_keys = self.denoiser.module.load_state_dict(checkpoint["denoiser"], strict=False)
        else:
            missing_keys, unexpected_keys = self.denoiser.load_state_dict(checkpoint["denoiser"], strict=False)

        assert len(unexpected_keys) == 0
        assert all([k.startswith("clip_model.") for k in missing_keys])

        try:
            self.optim.load_state_dict(checkpoint["optim"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        except:
            print("Fail to load optimizer and lr_scheduler")
        return checkpoint["epoch"], checkpoint["total_iter"]


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.denoiser.to(self.opt.device)
        self.vae.to(self.opt.device)

        # optimizer
        self.optim = torch.optim.AdamW(list(self.denoiser.parameters()), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)

        if self.rank == 0:
            print(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}")
            print(f"Iters Per Epoch, Training: {len(train_loader)}, Validation: {len(eval_val_loader)}")

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 100, 0, 0, 0, 100
            model_to_eval = self.denoiser.module if hasattr(self.denoiser, 'module') else self.denoiser
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, gt_motion, gen_motion, m_length, cond_list = evaluation_denoiser(
                self.opt.model_dir, eval_val_loader, model_to_eval, self.generate, self.logger, epoch,
                best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, save=True, draw=True, device=self.opt.device
            )

        dist.barrier(device_ids=[self.opt.rank])
        logs = defaultdict(def_value, OrderedDict())
       
        # training loop
        while epoch < self.opt.max_epoch:
            train_loader.sampler.set_epoch(epoch)
            torch.cuda.empty_cache()
            self.denoiser.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                # forward
                loss, attn_list, loss_dict = self.train_forward(batch_data)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if it >= self.opt.warm_up_iter:
                    self.lr_scheduler.step()
                
                # log
                logs["lr"] += self.optim.param_groups[0]["lr"]
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    # 1. Create a tensor of the sums to sync
                    # We include 'loss' and all components from 'loss_dict'
                    tags = ["loss"] + list(loss_dict.keys())
                    data_to_sync = torch.tensor([logs[tag] for tag in tags], device=self.opt.device)
                    
                    # 2. Sum everything across all GPUs
                    reduce_loss(data_to_sync)
                    
                    if self.rank == 0:
                        # 3. Divide by (log_every * world_size) to get the true global mean
                        divisor = self.opt.log_every * dist.get_world_size()
                        mean_loss = OrderedDict()
                        
                        wandb_metrics = {"epoch": epoch, "iter": it}

                        for idx, tag in enumerate(tags):
                            global_val = data_to_sync[idx].item() / divisor
                            # self.logger.add_scalar('Train/%s' % tag, global_val, it)
                            mean_loss[tag] = global_val
                            wandb_metrics[f"Train/{tag}"] = global_val

                        wandb_metrics["Train/lr"] = logs["lr"] / self.opt.log_every
                        wandb.log(wandb_metrics)

                        print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                    
                    # Reset logs on all ranks
                    logs = defaultdict(def_value, OrderedDict())

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            epoch += 1
            
            val_loader.sampler.set_epoch(epoch)
            self.denoiser.eval()
            val_log = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, attn_list, loss_dict = self.train_forward(batch_data)

                    val_log["loss"] += loss.item()
                    for tag, value in loss_dict.items():
                        val_log[tag] += value.item()

            val_tags = ["loss"] + list(loss_dict.keys())
            val_data = torch.tensor([val_log[tag] for tag in val_tags], device=self.opt.device)

            # Sum across all GPUs
            reduce_loss(val_data)

            if self.rank == 0:
                # total_batches is the sum of all batches processed by all GPUs
                total_batches = len(val_loader) * dist.get_world_size()
                wandb_val_metrics = {"epoch": epoch, "iter": it}

                msg = "Validation "
                for i, tag in enumerate(val_tags):
                    avg_val = val_data[i].item() / total_batches
                    # self.logger.add_scalar('Val/%s' % tag, avg_val, epoch)
                    msg += "%s: %.4f, " % (tag, avg_val) # Using .4f for better precision
                    wandb_val_metrics[f"Val/{tag}"] = avg_val

                print(msg)
            
            # evaluation
            if epoch % self.opt.eval_every_e == 0:
                if self.rank == 0:
                    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, gt_motion, gen_motion, m_length, cond_list = evaluation_denoiser(
                        self.opt.model_dir, eval_val_loader, self.denoiser, self.generate, it, epoch,
                        best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                        eval_wrapper=eval_wrapper, save=True, draw=True, device=self.opt.device
                    )

                    data = np.concatenate([gt_motion[:4], gen_motion[:4]], axis=0)
                    length = np.concatenate([m_length[:4], m_length[:4]], axis=0)
                    cond_list = cond_list[:4] + cond_list[:4]
                    save_dir = pjoin(self.opt.eval_dir, "E%04d" % (epoch))
                    os.makedirs(save_dir, exist_ok=True)
                    # plot_eval(data, save_dir, cond_list, length)

                dist.barrier(device_ids=[self.opt.rank])

        if self.rank == 0:
            wandb.finish()
    
    
    @torch.no_grad()
    def test(self, eval_wrapper, eval_val_loader, repeat_time, save_dir, cal_mm=True, save_motion=False):
        os.makedirs(save_dir, exist_ok=True)
        f = open(pjoin(save_dir, f"eval_steps{self.opt.num_inference_timesteps}_scale{self.opt.cond_scale}.log"), "w")

        self.denoiser.eval()
        self.vae.eval()
        self.noise_scheduler.set_timesteps(self.opt.num_inference_timesteps)
        metrics = {
            "fid": [],
            "div": [],
            "top1": [],
            "top2": [],
            "top3": [],
            "matching": [],
            "mm": []
        }
        for i in range(repeat_time):
            msg, fid, div, R_precision, matching, l1_dist, mm, pred_motion, caption_list = test_denoiser(
                eval_val_loader, self.generate, i, eval_wrapper, self.opt.joints_num, cal_mm
            )
            print(msg, file=f, flush=True)
            metrics["fid"].append(fid)
            metrics["div"].append(div)
            metrics["top1"].append(R_precision[0])
            metrics["top2"].append(R_precision[1])
            metrics["top3"].append(R_precision[2])
            metrics["matching"].append(matching)
            metrics["mm"].append(mm)

            if save_motion:
                converter = Joint2BVHConvertor()
                motion_save_dir = pjoin(save_dir, f"motion-steps{self.opt.num_inference_timesteps}-{i:02d}")
                os.makedirs(motion_save_dir, exist_ok=True)
                for i, (motion, caption) in enumerate(zip(pred_motion, caption_list)):
                    _, ik_joint = converter.convert(motion, pjoin(motion_save_dir, f"{i:06d}_ik.bvh"), foot_ik=True)
                    plot_3d_motion(pjoin(motion_save_dir, f"{i:06d}.mp4"), self.opt.kinematic_chain, motion, title=caption, fps=self.opt.fps)
                    np.savez(pjoin(motion_save_dir, f"{i:06d}.npz"), motion=motion, caption=caption)

        fid = np.array(metrics["fid"])
        div = np.array(metrics["div"])
        top1 = np.array(metrics["top1"])
        top2 = np.array(metrics["top2"])
        top3 = np.array(metrics["top3"])
        matching = np.array(metrics["matching"])
        mm = np.array(metrics["mm"])
        
        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality: {np.mean(mm):.3f}, conf. {np.std(mm)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        print(msg_final)
        print(msg_final, file=f, flush=True)

        f.close()
