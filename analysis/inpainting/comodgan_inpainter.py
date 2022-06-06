from inpainting.inpainter import Inpainter
from modules.co_mod_gan import Generator
import torch


class ComodganInpainter(Inpainter):
    def __init__(self, device):
        self.device = device

        self.network = Generator()
        self.network.load_state_dict(torch.load("checkpoints/co-mod-gan-places2-050000.pth"))
        self.network.eval()
        self.network = self.network.to(self.device)

    def inpaint(self, image, mask):
        latents = torch.randn(1, 512).to(self.device)
        return self.network(image, mask, [latents], truncation=None)
