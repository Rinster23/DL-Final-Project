import os
from models import *
from utils import *
from datasets import Segmentation_Mask_Dataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.cuda.empty_cache()


def load_weights(model):
    best_model_path = './checkpoints/image_segmentation.pth'
    if os.path.isfile(best_model_path):
        print('image segmentation weights found')
        model.load_state_dict(torch.load(best_model_path))


def save_weights(model):
    if 'checkpoints' not in os.listdir():
        os.mkdir('checkpoints')
    torch.save(model.state_dict(), './checkpoints/image_segmentation.pth')
    print('model weights saved successfully')


# Create Train DataLoader
batch_size = 16
num_videos_train = 1000
num_videos_val = 1000
num_frames_per_video = 22

all_frames_train = [[[i, j] for j in range(num_frames_per_video)] for i in range(num_videos_train)]

all_frames_train = np.array(all_frames_train).reshape(-1, 2)  # 22000 * 2
all_frames_train = all_frames_train.tolist()
train_data = Segmentation_Mask_Dataset(all_frames_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

all_frames_val = np.array([[[i, j] for j in range(num_frames_per_video)] for i in
                               range(num_videos_train, num_videos_train + num_videos_val)]).reshape(-1,2)
all_frames_val = all_frames_val[all_frames_val[:, 0] != 1370]
all_frames_val = all_frames_val.tolist()  # 22000 * 2
val_data = Segmentation_Mask_Dataset(all_frames_val, evaluation_mode=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# Hyperparameters:
num_epochs = 20
lr = 0.0001

model = UNet(n_channels=3, n_classes=49, bilinear=False)
model = model.to(device)
model = nn.DataParallel(model)
load_weights(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

train_losses = []

target = 0.01

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        batch_x, batch_y = get_blurry_images(batch_x, batch_y)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss += dice_loss(
            F.softmax(pred_y, dim=1).float(),
            F.one_hot(batch_y, model.module.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
        )
        train_loss.append(loss.item())
        if loss.item() < target:
            target = loss.item()
            save_weights(model)
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        val_score = evaluate(model, val_loader, device)
        scheduler.step(val_score)

    train_loss = np.mean(train_loss)
    print(f"Average train loss {train_loss}")
    train_losses.append(train_loss)
    save_weights(model)
