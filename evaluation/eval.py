import argparse
import numpy as np

import torch

from Training import EEG_Depression_Dectection

import pandas as pd

results_mean = []

pd.DataFrame(columns=["model_type", "timewindow", "overlap", "accuracy", "precision", "recall", "f1"])
results_std = []

pd.DataFrame(columns=["model_type", "timewindow", "overlap", "accuracy", "precision", "recall", "f1"])


for model_type in ["deprnet", "swin"]:

    if model_type == "swin":
        resize_to = 256
    else:
        resize_to = None

    for timewindow in [4,6]:
        for overlap in [0,0.25,0.5,0.75]:
            print(f"{model_type} {timewindow}s {overlap} overlap")

            training = EEG_Depression_Dectection(evaluation=True, data_folder=f"data/data_{timewindow}s_{overlap}overlap", save_folder=f"trained_models/{model_type}/{timewindow}s_{overlap}overlap", model_type=model_type, cross_validation=True, resize_to=resize_to, pretrained=True)
            metrics_mean, metrics_std = training.evaluation_cv("all", device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu"))

            results_mean.append([model_type, timewindow, overlap, metrics_mean[0], metrics_mean[1], metrics_mean[2], metrics_mean[3]])
            results_std.append([model_type, timewindow, overlap, metrics_std[0], metrics_std[1], metrics_std[2], metrics_std[3]])
                            
            results_mean_pd = pd.DataFrame(results_mean, columns=["model_type", "timewindow", "overlap", "accuracy", "precision", "recall", "f1"])
            results_std_pd = pd.DataFrame(results_std, columns=["model_type", "timewindow", "overlap", "accuracy", "precision", "recall", "f1"])

            # to csv
            results_mean_pd.to_csv("results_mean.csv")
            results_std_pd.to_csv("results_std.csv")
            print("\n")