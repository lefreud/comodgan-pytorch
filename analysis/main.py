import os
import random
from inpainting.comodgan_inpainter import ComodganInpainter
from utils import *
from coco_utils import get_image_pairs
from tqdm import tqdm

SEED = 0
IMAGE_COUNT = 20
DEVICE = torch.device("cuda")


def main():
    seed_all(SEED)

    inpainter = ComodganInpainter(DEVICE)

    assert IMAGE_COUNT % 2 == 0
    fig, axs = plt.subplots(IMAGE_COUNT // 2, 4, figsize=(20, IMAGE_COUNT * 3))
    
    image_pairs = get_image_pairs("datasets/coco/annotations/instances_val2017.json", erosion_radius=10)
    for i in tqdm(range(IMAGE_COUNT)):
        image, mask = next(image_pairs)
        image = get_image_tensor(image).to(DEVICE)
        mask = get_mask_tensor(mask).to(DEVICE)
        display_tensor(image, mask, ax=axs[i // 2, (i % 2) * 2])

        result = inpainter.inpaint(image, mask)
        display_tensor(result, ax=axs[i // 2, (i % 2) * 2 + 1])
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/comodgan_inpainter.png")


if __name__ == "__main__":
    main()
