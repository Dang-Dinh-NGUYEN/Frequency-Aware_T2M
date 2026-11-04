import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        rank=rank, 
        world_size=world_size
    )

def cleanup_ddp():
    dist.destroy_process_group()

def get_ddp_generator(dataset, opt, drop_last=True, shuffle=True, pin_memory=True):
    sampler = DistributedSampler(
        dataset, 
        num_replicas=opt.world_size, 
        rank=opt.rank, 
        shuffle=shuffle 
    )
    
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=opt.num_workers, # Use the value from your parser
        pin_memory=pin_memory,
        prefetch_factor=4,
        drop_last=drop_last
    )
    return loader, sampler

def reduce_loss(loss):
    rt = loss.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)   # AVG is supported on NCCL
    return rt