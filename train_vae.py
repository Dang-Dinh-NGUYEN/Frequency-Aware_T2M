import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import numpy as np
from torchinfo import summary

from options.vae_option import arg_parse
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from data.t2m_dataset import MotionDataset

from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.fixseed import fixseed
from utils.ddp import *


def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, opt.kinematic_chain, joint, title="None", fps=opt.fps, radius=opt.radius)

def load_vae(vae_opt, filename):
    model = VAE(vae_opt)
        
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.weights, 'model', filename),
                            map_location='cpu', weights_only=True)

    print(f'Loading VAE Model {vae_opt.checkpoints_dir}/{vae_opt.weights}/{filename}')
    model.load_state_dict(ckpt["vae"], strict=False)
    
    return model

if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    setup_ddp(opt.rank, opt.world_size)

    # model
    if opt.weights:
        weights_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, "opt.txt")
        weights_opt = get_opt(weights_opt_path, opt.device)
        net =  load_vae(weights_opt, "net_best_fid.tar").to(opt.device)
    else:
        net = VAE(opt).to(opt.device)
    net = DDP(net, device_ids=[opt.rank], output_device=opt.rank, find_unused_parameters=False)

    if opt.rank == 0:
        # num_params = sum(param.numel() for param in net.parameters())
        # print('Total trainable parameters of all models: {}M'.format(num_params/1_000_000))
        print(summary(net))

    # evaluation setup
    wrapper_opt = get_opt(opt.dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(opt.dataset_opt_path, 32, 'val', device=opt.device)

    # dataset & dataloader
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file)

    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers, shuffle=True, pin_memory=True)

    train_loader, _ = get_ddp_generator(train_dataset, opt, drop_last=True, shuffle=True, pin_memory=True)
    val_loader, _ = get_ddp_generator(val_dataset, opt, drop_last=True, shuffle=True, pin_memory=True)

    # train
    trainer = VAETrainer(opt, net)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, None) # plot_t2m

    cleanup_ddp()

## Usage
## python train_vae.py --name vae_silu_kl2e-2_noposloss --vae_type vae --lambda_kl 2e-2 --activation silu