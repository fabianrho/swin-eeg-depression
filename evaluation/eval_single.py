import argparse
import numpy as np

import torch
import sys
sys.path.append(".")

from Training import EEG_Depression_Dectection

import pandas as pd




training = EEG_Depression_Dectection(evaluation=True, data_folder=f"data/data_6s_0.25overlap", save_folder=f"trained_models/swin_no_pretrain/6s_0.25overlap", model_type="swin", cross_validation=True, resize_to=256, pretrained=True)
metrics_mean, metrics_std = training.evaluation_cv("all", device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu"))

print(metrics_mean)
print(metrics_std)
