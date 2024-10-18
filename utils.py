from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


def get_blurry_images(x_batch, y_batch, noise_level=0.1):
    x_random_noise = x_batch.clone()
    noise = torch.randn_like(x_random_noise) * noise_level
    x_random_noise += noise

    kernels = [5, 9]
    sigmas = [1, 5]
    x_gaussian_blurring = []
    x_g_k_s = x_batch.clone()
    for kernel_size in kernels:
        for sigma in sigmas:
            blur_transform = GaussianBlur(kernel_size, sigma)
            x_gaussian_blurring.append(blur_transform(x_g_k_s))
    x_gaussian_blurring = torch.cat(x_gaussian_blurring, 0)

    x_augmented = torch.cat([x_batch, x_random_noise, x_gaussian_blurring], 0)

    n_y = 1 + x_random_noise.shape[0] // int(x_batch.shape[0]) + x_gaussian_blurring.shape[0] // int(x_batch.shape[0])
    y_augmented = torch.cat([y_batch for i in range(n_y)], 0)
    return x_augmented, y_augmented


def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # Unet output is of shape [B, C, H, W], where C is the number of classes
    # 这里只考虑mask仅由0,1组成，即num_classes=2
    # reduce_batch_first=True 即单个mask的dice系数，否则是所有batch的平均
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first  # 不可能既是单个mask，又是batch
    # 对于一个形状为[N, H, W]的四维张量，如果sum_dim是(-1, -2)，这意味着函数将在最后两个维度（H和W）
    # 上进行求和

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False,
                          epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    # maximize the dice coefficient, min 1-dice
    return 1 - fn(input, target, reduce_batch_first=True)


@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    dice_score = 0
    num_used = 10
    cnt = 0
    for x_batch, y_batch in dataloader:
        if cnt == num_used:
            break
        x_batch, y_batch = x_batch.to(device), y_batch.to(device).long()
        mask_pred = net(x_batch)
        assert y_batch.min() >= 0 and y_batch.max() < net.module.n_classes, 'True mask indices should be in [0, n_classes]'
        y_batch = F.one_hot(y_batch, net.module.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
        # ignore the background class
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], y_batch[:, 1:], reduce_batch_first=False)
        cnt += 1
    net.train()
    return dice_score / num_used
