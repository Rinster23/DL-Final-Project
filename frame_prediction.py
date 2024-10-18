import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models import *
from datasets import Frame_Prediction_Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

batch_size = 8

train_data = Frame_Prediction_Dataset()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = Frame_Prediction_Dataset(evaluation_mode=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


def load_weights(model):
    best_model_path = './checkpoints/frame_prediction.pth'
    if os.path.isfile(best_model_path):
        print('frame prediction weights found')
        model.load_state_dict(torch.load(best_model_path))


def save_weights(model):
    if 'checkpoints' not in os.listdir():
        os.mkdir('checkpoints')
    torch.save(model.state_dict(), './checkpoints/frame_prediction.pth')
    print('model weights saved successfully')


model = SimVP(shape_in=(11, 3, 160, 240), hid_S=64, hid_T=512, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=4)
model = nn.DataParallel(model)
model = model.to(device)

load_weights(model)

num_epochs = 20
lr = 0.001
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader),
                                                epochs=num_epochs)

# Training Loop:
target = 0.001
for epoch in range(num_epochs):
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        train_loss.append(loss.item())
        if loss.item() < target:
            target = loss.item()
            save_weights(model)
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss = np.average(train_loss)
    print(f"Average train loss {train_loss}")
    save_weights(model)

    model.eval()
    val_pbar = tqdm(val_loader)

    with torch.no_grad():
        if epoch % 2 == 0:
            val_loss = []
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                val_loss.append(loss.item())
                val_pbar.set_description('val loss: {:.4f}'.format(loss.item()))
                torch.cuda.empty_cache()
        val_loss = np.average(val_loss)
        print(f'Average val loss {val_loss}')
