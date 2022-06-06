import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def get_image_tensor(image):
    if isinstance(image, str):
        image = cv2.imread(image)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512), cv2.INTER_NEAREST)
    image = (torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2) / 255) * 2 - 1
    return image


def display_tensor(result, mask=None, ax=None):
    result = result.detach().cpu().numpy()
    result = (result + 1) / 2
    if mask is not None:
        result *= mask.detach().cpu().numpy()
    result = (result[0].transpose((1, 2, 0))) * 255
    if ax:
        ax.imshow(result.clip(0, 255).astype(np.uint8))
    else:
        plt.imshow(result.clip(0, 255).astype(np.uint8))
        plt.show()


def get_mask_tensor(mask):
    if isinstance(mask, str):
        mask = cv2.imread(mask)

    mask = cv2.resize(mask, (512, 512), cv2.INTER_NEAREST)
    if len(mask.shape) == 2:
        mask = mask[..., None]
    mask = 1 - np.all(mask != [0, 0, 0], axis=-1).astype(int)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    return mask
