from models import *
from datasets import Combined_Model_Dataset
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()


def load_weights(model):
    best_model_path = './checkpoints/frame_prediction.pth'
    # best_model_path = './../../checkpoint_frame_predictione13.pth'
    if os.path.isfile(best_model_path):
        print('frame prediction weights found')
        model.module.frame_prediction_model.load_state_dict(torch.load(best_model_path))

    best_model_path = './checkpoints/image_segmentation.pth'
    # best_model_path = './../../image_segmentation_good.pth'
    if os.path.isfile(best_model_path):
        print('image segmentation weights found')
        model.module.image_segmentation_model.load_state_dict(torch.load(best_model_path))


# Create Val DataLoader
hidden = True
batch_size = 8
num_val_videos = 1000
val_data = Combined_Model_Dataset(num_val_videos, evaluation_mode=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# create hidden data loader
num_hidden_videos = 5000
hidden_data = Combined_Model_Dataset(num_hidden_videos, evaluation_mode='hidden')
hidden_loader = DataLoader(hidden_data, batch_size=batch_size)

model = CombinedModel(device)
model = nn.DataParallel(model)
load_weights(model)

val_loss = []
model.eval()

with torch.no_grad():
    if not hidden:
        val_pbar = tqdm(val_loader)
        preds = []
        total_y = []
        for batch_x, batch_y in val_pbar:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            batch_out = torch.argmax(out.detach().cpu(), dim=1)
            preds.append(batch_out)
            total_y.append(batch_y)
        preds = torch.cat(preds, dim=0)
        ground_truth = torch.cat(total_y, dim=0)
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        final_iou = jaccard(preds, ground_truth)
        print(preds.shape, ground_truth.shape)
        print("Final iou on val", final_iou)
    else:
        hidden_pbar = tqdm(hidden_loader)
        preds = []
        for batch_x in hidden_pbar:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            batch_out = torch.argmax(out.detach().cpu(), dim=1)
            preds.append(batch_out)

        preds = torch.cat(preds, dim=0)
        print(preds.shape)
        torch.save(preds, 'leaderboard_1_team_6.pt')
