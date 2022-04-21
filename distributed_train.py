from datasets import datasets
from models import models
from opt import config_parser
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torch
import sys
from evaluation import evaluation

def train(args):

    sys.stdout.write(
            (f"Cuda Availabel : {torch.cuda.is_available()}\n"
                f"GPU Number : {torch.cuda.device_count()}\n"
                )
        )
    torch.cuda.empty_cache()

    # https://zhuanlan.zhihu.com/p/86441879
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    torch.cuda.set_device(args.local_rank)  # before your code runs
    device = torch.device("cuda", args.local_rank)

    # make dataset
    train_dataset = datasets[args.dataset_name](args.train_dataset_file, args.train_dataset_info_file, information = args.dataset_information)
    valid_dataset = datasets[args.dataset_name](args.valid_dataset_file, args.valid_dataset_info_file, information = args.dataset_information)
    train_size = len(train_dataset)
    valid_size = len(valid_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size//torch.cuda.device_count(), drop_last=True)
    valid_sampler=torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_batch_sampler = torch.utils.data.BatchSampler(valid_sampler, args.batch_size * valid_size // (torch.cuda.device_count() * train_size), drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = train_batch_sampler,
            num_workers = 1, pin_memory = True
            )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_sampler = valid_batch_sampler,
            num_workers = 1, pin_memory = True
            )
    dataset_discription = f"Dataset :\nTrain item num : {train_size}; Train batch size : {args.batch_size}." +\
            f"\nValid item number : {valid_size}; Valid batch size : {int(args.batch_size*valid_size/train_size)}\n"
    sys.stdout.write(dataset_discription)

    # make model
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        kwargs = ckpt['kwargs']
        model = models[args.model_name](init_weights = False, **kwargs)
        model.load(ckpt)
    else:
        model = models[args.model_name](args.in_channels, args.out_channels)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank)

    # define optimizer
    optimizer = torch.optim.Adam(
            model.parameters(), lr = args.lr
            )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        final_div_factor=1e2,
        epochs = args.epoch, steps_per_epoch=len(train_loader))

    # define loss
    # mse_loss = torch.nn.MSELoss(reduction = "mean")
    loss = torch.nn.SmoothL1Loss(beta = 1)

    # init log file
    log_folder = f"{args.base_dir}/{args.model_name.lower()}/{args.exp_name}"
    if args.local_rank == 0:
        os.makedirs(log_folder, exist_ok = True)
        os.makedirs(f"{log_folder}/model", exist_ok = True)
        summary_writer = SummaryWriter(log_folder)

    # start train
    train_losses = []
    valid_losses = []
    save_per_steps = int((train_size/args.batch_size)/10)
    global_iteration = 0
    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)
        refresh_rate = 20
        pbar = tqdm(zip(train_loader, valid_loader), miniters = refresh_rate, file = sys.stdout)
        for train_item, valid_item in pbar:
            # train the model
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            train_data, train_target, _ = train_item
            train_data, train_target = train_data.to(device), train_target.to(device)
            train_pred = model(train_data)
            train_loss = loss(train_target, train_pred)
            train_loss.backward()
            optimizer.step()
            train_loss = train_loss.detach().item()
            train_losses.append(train_loss)
            if args.local_rank == 0:
                summary_writer.add_scalar("train/train_lr",
                        optimizer.param_groups[0]["lr"], global_step = global_iteration)
            scheduler.step()    # update learning

            # valid the model
            model.eval()
            valid_data, valid_target, _ = valid_item
            valid_data, valid_target = valid_data.to(device), valid_target.to(device)
            valid_pred = model(valid_data)
            valid_loss = loss(valid_target, valid_pred).detach().item()
            valid_losses.append(valid_loss)

            # update log
            if args.local_rank == 0:
                summary_writer.add_scalar("train/train_mse_loss",
                        train_loss, global_step = global_iteration)
                summary_writer.add_scalar("valid/valid_mse_loss",
                        valid_loss, global_step = global_iteration)
                if global_iteration%refresh_rate == 0:
                    pbar.set_description(
                            f'Iteration {global_iteration:05d}:'
                            + f' train loss = {train_loss:.2f}'
                            + f' valid loss = {valid_loss:.2f}'
                            )
                global_iteration += 1

                # check and save the model
                if epoch > 3 and global_iteration % save_per_steps == 0:
                    now_valid = np.mean(valid_losses[-save_per_steps:])
                    pass_valid = np.mean(valid_losses[-2*save_per_steps:-save_per_steps])
                    if now_valid < pass_valid:
                        # save the model
                        model.module.save(f"{log_folder}/model/model.ckpt")
                        sys.stdout.write(f"Write model to {log_folder}/model/model.ckpt")
    sys.stdout.write("DNN Model Train Finished!")

if __name__ == "__main__":

    # get config
    args = config_parser()
    sys.stdout.write(str(args) + "\n")

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
