import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

#######################################
#######################################
# MAKE CHANGES TO BASE DIRECTORY HERE:
base_dir = '../Dataset/'


class Segmentation_Mask_Dataset(Dataset):
    def __init__(self, all_frames, evaluation_mode=False):
        self.frames = all_frames
        self.evaluation_mode = evaluation_mode

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        global base_dir
        if self.evaluation_mode:
            mode = 'val'
        else:
            mode = 'train'
        i, j = self.frames[idx]
        if len(str(i)) < 5:
            temp = '0' * (5 - len(str(i))) + str(i)
        else:
            temp = str(i)
        file_path = f"{base_dir}{mode}/video_{temp}/image_{j}.png"
        frame = torch.tensor(plt.imread(file_path)).permute(2, 0, 1)

        file_path = f"{base_dir}{mode}/video_{temp}/mask.npy"
        mask = np.load(file_path)[j]
        return frame, mask


class Frame_Prediction_Dataset(Dataset):
    def __init__(self, evaluation_mode=False):
        if evaluation_mode:
            self.vid_indexes = list(range(1000, 2000))
        else:
            self.vid_indexes = list(range(1000)) + list(range(2000, 15000))

        self.evaluation_mode = evaluation_mode

    def __getitem__(self, idx):
        global base_dir

        num_visible_frames = 11
        num_total_frames = 22
        x = []
        y = []
        i = self.vid_indexes[idx]
        if self.evaluation_mode:
            mode = 'val'
        elif i >= 1000:
            mode = 'unlabeled'
        else:
            mode = 'train'
        if len(str(i)) < 5:
            temp = '0' * (5 - len(str(i))) + str(i)
        else:
            temp = str(i)
        filepath = f'{base_dir}{mode}/video_{temp}/'
        # obtain x values.
        for j in range(num_visible_frames):
            x.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)
        for j in range(num_visible_frames, num_total_frames):
            y.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        y = torch.stack(y, 0)
        return x, y

    def __len__(self):
        vid_len = len(self.vid_indexes)
        return vid_len


class Combined_Model_Dataset(Dataset):
    def __init__(self, num_of_vids=1000, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        if self.evaluation_mode is True:
            self.mode = 'val'
            start_num = 1000
        elif self.evaluation_mode == 'hidden':
            self.mode = 'hidden'
            start_num = 15000
        else:
            self.mode = 'train'
            start_num = 0
        self.vid_indexes = [i for i in range(start_num, num_of_vids + start_num)]
        if 1370 in self.vid_indexes:
            self.vid_indexes.remove(1370)
        self.num_of_vids = num_of_vids

    def __len__(self):
        return self.num_of_vids

    def __getitem__(self, idx):
        global base_dir

        num_visible_frames = 11
        x = []
        if idx >= len(self.vid_indexes):
            raise StopIteration

        i = self.vid_indexes[idx]
        if len(str(i)) < 5:
            temp = '0' * (5 - len(str(i))) + str(i)
        else:
            temp = str(i)
        filepath = f'{base_dir}{self.mode}/video_{temp}/'

        # obtain x values.
        for j in range(num_visible_frames):
            x.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)

        if self.evaluation_mode == 'hidden':
            return x

        file_path = f"{base_dir}{self.mode}/video_{temp}/mask.npy"
        y = np.load(file_path)[-1]  # last frame.
        return x, y
