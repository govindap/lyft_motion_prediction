import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet34
from torch import Tensor
from typing import Dict

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from pathlib import Path
import pandas as pd
import os
import random
import time
import gc, psutil
from lyft_common import *
from lyft_CNN import *

DIR_INPUT = "F:\cs230\lyft-motion-prediction-autonomous-vehicles/"
set_seed(42)

def buildTrainingData():
    # Train DATASET
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    train_cfg = cfg["train_data_loader"]

    # Rasterizer
    rasterizer = build_rasterizer(cfg, dm)

    # Train dataset/dataloader
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=train_cfg["shuffle"],
                                  batch_size=train_cfg["batch_size"],
                                  num_workers=train_cfg["num_workers"])

    return train_dataloader

def lyft_train():
    train_dataloader = buildTrainingData()
    print('data loaded')
    # ==== INIT MODEL=================
    model_name = cfg["model_params"]["model_name"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = LyftCNNModel(cfg)

    # load weight if there is a pretrained model
    weight_path = f"{model_name}_final.pth"
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    # load optimizer if there is a pretrained model
    opt_weight_path = f"{model_name}_optimizer_final.pth"
    if opt_weight_path:
        optimizer.load_state_dict(torch.load(opt_weight_path))

    print(f'device {device}')
    tr_it = iter(train_dataloader)
    n_steps = cfg["train_params"]["steps"]
    progress_bar = range(1, 1 + n_steps)
    losses = []
    iterations = []
    metrics = []
    memorys = []
    times = []
    model_name = cfg["model_params"]["model_name"]
    update_steps = cfg['train_params']['update_steps']
    checkpoint_steps = cfg['train_params']['checkpoint_steps']
    t_start = time.time()
    torch.set_grad_enabled(True)

    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        model.train()

        # Backward pass
        optimizer.zero_grad()
        if cfg["train_params"]["precision"] == True:
            scaler = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast():
                loss, _, _ = forward(data, model, device)

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()
        else:

            loss, _, _ = forward(data, model, device)
            loss.backward()
            optimizer.step()

        loss_v = loss.item()
        losses.append(loss_v)

        if i % update_steps == 0:
            mean_losses = np.mean(losses)
            timespent = (time.time() - t_start) / 60
            print('i: %5d' % i,
                  'loss: %10.5f' % loss_v, 'loss(avg): %10.5f' % mean_losses,
                  '%.2fmins' % timespent, end=' | ')
            mem = memory()
            if i % checkpoint_steps == 0:
                torch.save(model.state_dict(), f'{model_name}_{i}.pth')
                torch.save(optimizer.state_dict(), f'{model_name}_optimizer_{i}.pth')
            iterations.append(i)
            metrics.append(mean_losses)
            memorys.append(mem)
            times.append(timespent)

    torch.save(model.state_dict(), f'{model_name}_final.pth')
    torch.save(optimizer.state_dict(), f'{model_name}_optimizer_final.pth')
    results = pd.DataFrame({
        'iterations': iterations,
        'metrics (avg)': metrics,
        'elapsed_time (mins)': times,
        'memory (GB)': memorys,
    })
    results.to_csv(f'train_metrics_{model_name}_{n_steps}.csv', index=False)
    print(f'Total training time is {(time.time() - t_start) / 60} mins')

if __name__ == "__main__":
    print("start")
    lyft_train()
    print("complete")
