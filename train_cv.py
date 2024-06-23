import argparse
import numpy as np

import torch

from Training import EEG_Depression_Dectection

# create arguments
parser = argparse.ArgumentParser(description='Train a model on EEG data')


parser.add_argument("--model_type", type=str, default="swin", help="Type of model to train")
parser.add_argument('--timewindow', type=int, help='Window size in seconds')
parser.add_argument('--overlap', type=float, help='Overlap percentage')


args = parser.parse_args()

timewindow = args.timewindow
overlap = args.overlap
model_type = args.model_type

# if overlap is float and 0.0, then it is 0
if overlap == 0.0:
    overlap = int(overlap)

if model_type == "swin":
    resize_to = 256
else:
    resize_to = None


training = EEG_Depression_Dectection(data_folder=f"data/data_{timewindow}s_{overlap}overlap", save_folder=f"trained_models/{model_type}_no_pretrain/{timewindow}s_{overlap}overlap", model_type=model_type, cross_validation=True, resize_to=resize_to, pretrained=False)
# training = EEG_Depression_Dectection(data_folder=f"data/data_{timewindow}s_{overlap}overlap", save_folder=f"trained_models/{model_type}/{timewindow}s_{overlap}overlap", model_type=model_type, cross_validation=True, resize_to=None, pretrained=True)


training.train_cv("data/folds.txt", epochs = 100, early_stopping_threshold = 10, device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))




