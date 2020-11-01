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

def buildTestData():
    # Train DATASET
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    # Test dataset
    test_cfg = cfg["test_data_loader"]

    # Rasterizer
    rasterizer = build_rasterizer(cfg, dm)

    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset, shuffle=test_cfg["shuffle"],
                                 batch_size=test_cfg["batch_size"], num_workers=test_cfg["num_workers"])
    return test_dataloader

def lyftTest():
    test_dataloader = buildTestData()
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
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []
    memorys_pred = []
    t0 = time.time()
    times_pred = []
    iterations_pred = []

    for i, data in enumerate(test_dataloader):

        _, preds, confidences = forward(data, model, device, compute_loss=False)

        preds = torch.einsum('bmti,bji->bmtj',
                             preds.double(),
                             data["world_from_agent"].to(device)[:, :2, :2]).cpu().numpy()

        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

        if i % 5 == 0:
            t = ((time.time() - t0) / 60)
            print('%4d' % i, '%6.2fmins' % t, end=' | ')
            mem = memory()
            iterations_pred.append(i)
            memorys_pred.append(mem)
            times_pred.append(t)

        # if i > 20:
        #     break

    results = pd.DataFrame({
        'iterations': iterations_pred,
        'elapsed_time (mins)': times_pred,
        'memory (GB)': memorys_pred,
    })
    results.to_csv(f'test_metrics_{model_name}.csv', index=False)
    print(f'Total test time is {(time.time() - t0) / 60} mins')

    print('Total timespent: %6.2fmins' % ((time.time() - t0) / 60))
    pred_path = 'submission.csv'
    write_pred_csv(
        pred_path,
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),
        coords=np.concatenate(future_coords_offsets_pd),
        confs=np.concatenate(confidences_list),
    )

if __name__ == "__main__":
    print("start")
    lyftTest()
    print("complete")