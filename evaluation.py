import torch
from datasets import datasets
from models import models
from opt import config_parser
import numpy as np
import sys
from utils.coordinates import xyz2rthetaphi
import pandas as pd
from tqdm import tqdm

@torch.no_grad()
def evaluation(model, valid_dataloader, device):
    true_xyz = []
    true_edep = []
    pred_xyz = []
    for valid_item in tqdm(valid_dataloader):
        valid_data, valid_target, valid_edep = valid_item
        pred_target = model(valid_data.to(device))
        true_xyz.append(valid_target.to(device))
        pred_xyz.append(pred_target)
        true_edep.append(valid_edep)
    true_xyz = torch.concat(true_xyz, dim = 0)
    true_edep = torch.concat(true_edep, dim = 0)
    pred_xyz = torch.concat(pred_xyz, dim = 0)
    return true_xyz, true_edep, pred_xyz

if __name__ == "__main__":

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        sys.stdout.write(f"Use Cuda\n")

    # get config
    args = config_parser()
    sys.stdout.write(str(args) + "\n")

    # load dataset
    test_dataset = datasets[args.dataset_name](args.test_dataset_file, args.test_dataset_info_file)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size = args.batch_size, shuffle = True, num_workers = 1
        )
    dataset_discription = f"Dataset :" +\
            f"\nTest item number : {len(test_loader)}; Test batch size : {args.batch_size}\n"
    sys.stdout.write(dataset_discription)

    # make model
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        kwargs = ckpt['kwargs']
        model = models[args.model_name](init_weights = False, **kwargs)
        model.load(ckpt)
        sys.stdout.write(f"Load model from {args.ckpt}!\n")
    else:
        model = models[args.model_name](args.in_channels, args.out_channels)
    model.to(device)
    model.eval()

    # evaluation ...
    true_xyz, true_edep, pred_xyz = evaluation(model, test_loader, device)
    true_xyz, true_edep, pred_xyz = true_xyz.cpu().numpy(), true_edep.cpu().numpy(), pred_xyz.cpu().numpy()
    true_rthetaphi = xyz2rthetaphi(true_xyz)
    pred_rthetaphi = xyz2rthetaphi(pred_xyz)

    # make DataFrame and save it
    df = pd.DataFrame({
        "fGdLSEdepX" : true_xyz[:,0],
        "fGdLSEdepY" : true_xyz[:,1],
        "fGdLSEdepZ" : true_xyz[:,2],
        "fGdLSEdepR" : true_rthetaphi[:,0],
        "fGdLSEdepTheta" : true_rthetaphi[:,1],
        "fGdLSEdepPhi" : true_rthetaphi[:,2],
        "fGdLSEdep" : true_edep,
        "fRecX" : pred_xyz[:,0],
        "fRecY" : pred_xyz[:,1],
        "fRecZ" : pred_xyz[:,2],
        "fRecR" : pred_rthetaphi[:,0],
        "fRecTheta" : pred_rthetaphi[:,1],
        "fRecPhi" : pred_rthetaphi[:,2]
    }).to_csv(args.eval_output, index = None)
