import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn as nn
import os
import time
import datetime
import numpy as np
from collections import OrderedDict
import wandb
from utils.eval_t2m import evaluation_vae, test_vae
from utils.motion_process import recover_from_ric
from utils.utils import print_current_loss
import torch.distributed as dist

# Let ops that stay in fp32 (the DCT block, anything autocast keeps in fp32) use TF32 tensor cores.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _unwrap(model):
    """Return the underlying model if wrapped in DDP (or torch.compile)."""
    while hasattr(model, 'module') or hasattr(model, '_orig_mod'):
        model = getattr(model, 'module', None) or getattr(model, '_orig_mod')
    return model


def _accumulate(store, loss, loss_dict):
    """Add losses to running sums WITHOUT leaving the GPU (no .item(), no CPU/GPU sync)."""
    store["loss"] = store.get("loss", 0) + loss.detach().float()
    for tag, value in loss_dict.items():
        store[tag] = store.get(tag, 0) + value.detach().float()


class TemporalSmoothnessLoss(nn.Module):
    def forward(self, joints, len_mask=None):
        # joints: [B, T, J, 3] decoded poses
        diff = joints[:, 1:] - joints[:, :-1]          # [B, T-1, J, 3]
        sq   = diff.pow(2).sum(dim=(-2, -1))           # ||Y_t - Y_{t-1}||^2 -> [B, T-1]
        if len_mask is not None:
            valid = (len_mask[:, 1:] & len_mask[:, :-1]).float()
            return (sq * valid).sum() / valid.sum().clamp(min=1.0)
        return sq.mean()


class VAETrainer:
    def __init__(self, opt, vae):
        self.opt = opt
        self.vae = vae
        self.rank = getattr(opt, 'rank', 0)
        self.amp = getattr(opt, 'amp', False)
        print(f"Using mixed precision (bf16) {self.amp}")
        # No GradScaler: bf16 has the same exponent range as fp32, so loss scaling is unnecessary.

        if opt.is_train:
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss()
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()

            # self.temp_loss = TemporalSmoothnessLoss()
            # ONLY Rank 0 gets the logger
            if self.rank == 0:
                wandb.init(
                    project='t2m-project',
                    name=opt.name,
                    config=vars(opt)  # Pass all training options as hyperparameters
                )

                wandb.define_metric("iter")
                wandb.define_metric("Train/*", step_metric="iter")
                wandb.define_metric("Val/*", step_metric="iter")
                wandb.define_metric("Eval/*", step_metric="iter")

            self.logger = None

    def _autocast(self):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp)

    def train_forward(self, batch_data):
        # non_blocking lets the copy overlap with GPU work (needs pin_memory=True in the DataLoader)
        motion = batch_data.to(self.opt.device, dtype=torch.float32, non_blocking=True)
        root, ric, rot, vel, contact = torch.split(motion, [4, 3 * (self.opt.joints_num - 1), 6 * (self.opt.joints_num - 1), 3 * self.opt.joints_num, 4], dim=-1)

        pred_motion, loss_dict = self.vae.forward(motion)
        pred_root, pred_ric, pred_rot, pred_vel, pred_contact = torch.split(pred_motion, [4, 3 * (self.opt.joints_num - 1), 6 * (self.opt.joints_num - 1), 3 * self.opt.joints_num, 4], dim=-1)

        self.motion = motion
        self.pred_motion = pred_motion

        # loss
        loss_rec = self.recon_criterion(pred_motion, motion)
        loss_vel = self.recon_criterion(pred_vel, vel)
        loss_pos = self.recon_criterion(pred_ric, ric)
        # loss_temp   = self.temp_loss(pred_motion)
        loss_kl = loss_dict["loss_kl"]

        loss = loss_rec + loss_vel * self.opt.lambda_vel + loss_pos * self.opt.lambda_pos + loss_kl * self.opt.lambda_kl  # + 0.02 * loss_temp

        loss_dict["loss_recon"] = loss_rec
        loss_dict["loss_vel"] = loss_vel
        loss_dict["loss_pos"] = loss_pos
        # loss_dict["loss_temp"] = loss_temp

        return loss, loss_dict

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

    def save(self, file_name, epoch, total_iter):
        if self.rank == 0:  # ONLY Rank 0 should save files
            state = {
                "vae": _unwrap(self.vae).state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "total_iter": total_iter,
            }
            torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        _unwrap(self.vae).load_state_dict(checkpoint["vae"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint["epoch"], checkpoint["total_iter"]

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vae.to(self.opt.device)
        world_size = dist.get_world_size()

        # optimizer (fused=True: one kernel for the whole update instead of many small ones)
        self.optim = torch.optim.AdamW(self.vae.parameters(), lr=self.opt.lr, betas=(0.9, 0.99),
                                       weight_decay=self.opt.weight_decay, fused=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)

        if self.rank == 0:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))

            # Evaluate the unwrapped model: calling the DDP wrapper on rank 0 alone can hang.
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_vae(
                self.opt.model_dir, eval_val_loader, _unwrap(self.vae), epoch, it, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=100,
                eval_wrapper=eval_wrapper, save=False)

        # Other ranks wait here instead of blocking inside the first all_reduce.
        dist.barrier(device_ids=[self.opt.rank])

        # Running sums live on the GPU; we only pull them to the CPU every log_every steps.
        logs = OrderedDict()
        n_accum = 0
        lr_sum = 0.0

        # training loop
        while epoch < self.opt.max_epoch:
            train_loader.sampler.set_epoch(epoch)
            self.vae.train()

            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                # forward / backward
                self.optim.zero_grad(set_to_none=True)
                with self._autocast():
                    loss, loss_dict = self.train_forward(batch_data)
                loss.backward()
                self.optim.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()

                # log: accumulate on GPU, no .item() here
                _accumulate(logs, loss, loss_dict)
                lr_sum += self.optim.param_groups[0]['lr']  # plain Python float, no sync
                n_accum += 1

                if it % self.opt.log_every == 0:
                    tags = list(logs.keys())
                    data_to_sync = torch.stack([logs[tag] for tag in tags])

                    # Sum across all GPUs (this is the only CPU/GPU sync point in the loop)
                    dist.all_reduce(data_to_sync, op=dist.ReduceOp.SUM)

                    if self.rank == 0:
                        values = (data_to_sync / (n_accum * world_size)).tolist()
                        mean_loss = OrderedDict(zip(tags, values))

                        wandb_metrics = {"iter": it, "epoch": epoch}
                        for tag, val in mean_loss.items():
                            wandb_metrics[f"Train/{tag}"] = val
                        wandb_metrics["Train/lr"] = lr_sum / n_accum
                        wandb.log(wandb_metrics)

                        print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    # Reset on all ranks
                    logs = OrderedDict()
                    n_accum = 0
                    lr_sum = 0.0

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)

            epoch += 1

            # validation
            val_loader.sampler.set_epoch(epoch)
            self.vae.eval()
            val_log = OrderedDict()
            n_val = 0

            with torch.no_grad():
                for batch_data in val_loader:
                    with self._autocast():
                        loss, loss_dict = self.train_forward(batch_data)
                    _accumulate(val_log, loss, loss_dict)
                    n_val += 1

            val_tags = list(val_log.keys())
            val_data = torch.stack([val_log[tag] for tag in val_tags])

            # Sum across all GPUs
            dist.all_reduce(val_data, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                values = (val_data / (n_val * world_size)).tolist()
                wandb_val_metrics = {"epoch": epoch, "iter": it}

                msg = "Validation "
                for tag, avg_val in zip(val_tags, values):
                    msg += "%s: %.4f, " % (tag, avg_val)
                    wandb_val_metrics[f"Val/{tag}"] = avg_val
                print(msg)
                wandb.log(wandb_val_metrics)

            # evaluation
            if epoch % self.opt.eval_every_e == 0:
                if self.rank == 0:
                    best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_vae(
                        self.opt.model_dir, eval_val_loader, _unwrap(self.vae), epoch, it, best_fid=best_fid,
                        best_div=best_div, best_top1=best_top1,
                        best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)

                    if plot_eval:
                        data = torch.cat([self.motion[:4], self.pred_motion[:4]], dim=0).detach().float().cpu().numpy()
                        save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                        os.makedirs(save_dir, exist_ok=True)
                        plot_eval(data, save_dir)

                dist.barrier(device_ids=[self.opt.rank])

        if self.rank == 0:
            wandb.finish()

    @torch.no_grad()
    def test(self, eval_wrapper, eval_val_loader, repeat_time, save_dir, cal_mm=True):
        os.makedirs(save_dir, exist_ok=True)
        f = open(pjoin(save_dir, 'eval.log'), 'w')

        self.vae.eval()
        metrics = {
            "fid": [],
            "div": [],
            "top1": [],
            "top2": [],
            "top3": [],
            "matching": [],
            "mpjpe": [],
            "mm": []
        }
        for i in range(repeat_time):
            fid, diversity, R_precision, matching_score, mpjpe, multimodality = test_vae(
                eval_val_loader, self.vae, i, eval_wrapper, self.opt.joints_num, cal_mm=cal_mm
            )
            metrics["fid"].append(fid)
            metrics["div"].append(diversity)
            metrics["top1"].append(R_precision[0])
            metrics["top2"].append(R_precision[1])
            metrics["top3"].append(R_precision[2])
            metrics["matching"].append(matching_score)
            metrics["mpjpe"].append(mpjpe)
            metrics["mm"].append(multimodality)

        fid = np.array(metrics["fid"])
        div = np.array(metrics["div"])
        top1 = np.array(metrics["top1"])
        top2 = np.array(metrics["top2"])
        top3 = np.array(metrics["top3"])
        matching = np.array(metrics["matching"])
        mpjpe = np.array(metrics["mpjpe"])
        mm = np.array(metrics["mm"])

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMPJPE: {np.mean(mpjpe):.3f}, conf. {np.std(mpjpe)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality: {np.mean(mm):.3f}, conf. {np.std(mm)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        print(msg_final)
        print(msg_final, file=f, flush=True)

        f.close()