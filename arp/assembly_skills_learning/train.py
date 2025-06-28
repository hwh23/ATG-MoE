# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import os
import wandb
import os.path as osp
import torch
from copy import copy
from tqdm import tqdm
import logging
from time import time
import sys, shlex
from utils import configurable, DictConfig, config_to_dict
import torch.multiprocessing as mp
from utils.dist import find_free_port, find_free_port_for_tensorboard
import torch.distributed as dist
from arp.assembly_skills_learning.dataset import TransitionDataset
from utils.structure import ASSEMBLY_TASKS
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from runstats import Statistics
from torch.utils.tensorboard import SummaryWriter
import logging
from tensorboard import program
# Suppress TensorBoard logs
logging.getLogger('tensorboard').setLevel(logging.ERROR)
from omegaconf import OmegaConf
import hydra

def train(agent, dataloader: DataLoader, logger, device: int, freq: int = 30, rank: int = 0, save_freq: int = 6000, start_step=0, use_wandb=False,
          writer: SummaryWriter=None):
    start = time()
    run_stats = {}
    steps = start_step
    print(f"Training on rank {rank} with {len(dataloader)} batches, start step: {start_step}, freq: {freq}, save_freq: {save_freq}")
    for i, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
        print('' * 20, f"Rank {rank} batch {i} at step {steps}")
        batch = {k:v.to(device) for k,v in batch.items()}
        loss_dict = agent.update(batch)

        for k, v in loss_dict.items():
            if 'loss' in k and k not in run_stats:
                run_stats[k] = Statistics()
        stat_dict = copy(loss_dict)
        if use_wandb and rank == 0: wandb.log(stat_dict)
        for k in run_stats:
            run_stats[k].push(loss_dict[k])
            stat_dict[k] = run_stats[k].mean()
        if writer:# add loss dict to tensorboard
            # writer.add_scalars("losses", loss_dict, steps)
            for k, v in loss_dict.items():
                if k=="v1_norm" or k=="v2_norm" or k=="exponential_weight.rot-z.ce_loss":
                    writer.add_scalar(f"stat/{k}", v, steps) 
                else:                  
                    writer.add_scalar(f"loss/{k}", v, steps) 
            
        if i % freq == 0 and rank == 0:
            logger(f"[step:{str(steps).zfill(8)} time:{time()-start:.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(stat_dict.items())]),
                printer=tqdm.write)
        if rank == 0 and i != 0 and i % save_freq == 0:
            logger(f"checkpoint to {agent.log_dir} at step {steps} and reset running metrics", printer=tqdm.write)
            agent.save(steps)
            run_stats = {}
        steps += 1
    agent.save(steps)   


def main_single(rank: int, cfg: DictConfig, port: int, log_dir:str):
    if cfg.wandb.use_wandb and rank == 0:
        if cfg.wandb.use_proxy:
            os.environ.update({
            "HTTP_PROXY": cfg.wandb.proxy_settings.http_proxy,
            "HTTPS_PROXY": cfg.wandb.proxy_settings.https_proxy,
            "WANDB_INIT_TIMEOUT": cfg.wandb.proxy_settings.wandb_init_timeout,
            "WANDB_HTTP_TIMEOUT": cfg.wandb.proxy_settings.wandb_http_timeout,        
        })
        wandb.init(project=cfg.wandb.project, name='/'.join(log_dir.split('/')[-2:]), config=config_to_dict(cfg))

    world_size = cfg.train.num_gpus
    assert world_size > 0
    ddp, on_master = world_size > 1, rank == 0
    print(f'Rank - {rank}, master = {on_master}  port = {port}, world_size = {world_size}, log_dir = {log_dir}')
    if ddp:
        # os.environ["MASTER_ADDR"] = "localhost" # BUG 服务器多卡会无法通讯
        os.environ["MASTER_ADDR"] = "0.0.0.0"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = rank
    if on_master:
        logfile = open(osp.join(log_dir, 'log.txt'), "w")

    def log(msg, printer=print):
        if on_master:
            print(msg, file=logfile, flush=True)
            printer(msg)

    env_cfg = cfg.env
    if env_cfg.tasks == 'all':
        tasks = ASSEMBLY_TASKS
    else:
        tasks = env_cfg.tasks.split(',')

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    cfg.model.hp.lr *= (world_size * cfg.train.bs)
    cfg.model.hp.cos_dec_max_step = cfg.train.epochs * cfg.train.num_transitions_per_epoch // cfg.train.bs // world_size

    py_module = cfg.py_module
    from importlib import import_module
    MOD = import_module(py_module)
    Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)
    if ddp:
        net = DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True)
    agent = Policy(net, cfg.model.hp, log_dir=log_dir)
    agent.build(training=True, device=device)

    start_step = 0
    if cfg.model.weights:
        start_step = agent.load(cfg.model.weights)
        log(f"Resuming from step {start_step}")
    if ddp: dist.barrier()

    total_batch_num = cfg.train.num_transitions_per_epoch * cfg.train.epochs // cfg.train.bs #(cfg.train.bs * world_size)
    total_batch_num -= (start_step * world_size)
    dataset = TransitionDataset(cfg.train.demo_folder, tasks, cameras=env_cfg.cameras,
            batch_num=total_batch_num, batch_size=cfg.train.bs, scene_bounds=env_cfg.scene_bounds,
            voxel_size=env_cfg.voxel_size, rotation_resolution=env_cfg.rotation_resolution,
            cached_data_path=cfg.train.cached_dataset_path, time_in_state=cfg.env.time_in_state,
            episode_length=cfg.env.episode_length, k2k_sample_ratios=cfg.train.k2k_sample_ratios, 
            origin_style_state=cfg.env.origin_style_state,
            shuffle=True)

    log("Begin Training...")
    dataloader, sampler = dataset.dataloader(num_workers=cfg.train.num_workers, 
                                             pin_memory=False, distributed=ddp)
    log(f"Total number of batches: {len(dataloader)}")

    if ddp: sampler.set_epoch(0)
    if cfg.train.eval_mode:
        agent.eval()
        torch.set_grad_enabled(False)
    else:
        agent.train()

    # BUG 多进程很卡，Create a TensorBoard program instance
    # tb = program.TensorBoard()
    # Launch TensorBoard
    # tb.configure(argv=[None, '--logdir', cfg.output_dir, '--port', str(find_free_port_for_tensorboard())])
    # print(f"\n==================================" + \
    #       f" TensorBoard is running at {tb.launch()} " + \
    #       f"==================================\n")

    # train(agent, dataloader, log, device, freq=cfg.train.disp_freq, rank=rank, save_freq=cfg.train.save_freq, 
    #     start_step=start_step, use_wandb=cfg.wandb and rank == 0,
    #     writer=SummaryWriter(log_dir=cfg.output_dir))    
    print(f"Training started, logs are saved to {log_dir}")
    train(agent, dataloader, log, device, freq=cfg.train.disp_freq, rank=rank, save_freq=cfg.train.save_freq, 
        start_step=start_step, use_wandb=cfg.wandb.use_wandb and rank == 0,
        writer=False)
    print(f"Training finished, logs are saved to {log_dir}")
    

# @hydra.main(config_path="./configs", config_name="arp_plus.yaml",version_base="1.1")
@configurable() # 多卡无法解析output dir
def main(cfg: DictConfig):
    if cfg.train.num_gpus <= 1:
        main_single(0, cfg, -1, cfg.output_dir)
    else:
         # 预解析 output_dir（此时 hydra 插值是有效的）
        # 解析所有Hydra插值
        # OmegaConf.resolve(cfg)
        # output_dir = str(cfg.output_dir)
        port = find_free_port()
        mp.spawn(main_single, args=(cfg, port, cfg.output_dir),  nprocs=cfg.train.num_gpus, join=True)
        print(f"Training finished, logs are saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()
