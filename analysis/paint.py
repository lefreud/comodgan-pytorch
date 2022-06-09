import tkinter as tk

import numpy as np
from PIL import ImageTk, ImageDraw
import os
import random
from inpainting.comodgan_inpainter import ComodganInpainter
from utils import *
from coco_utils import get_image_pairs
from tqdm import tqdm
from PIL import Image
import cv2

SEED = 0
DEVICE = torch.device("cuda")


class Paint(tk.Tk):
    # Inspired by: https://github.com/zsyzzsoft/co-mod-gan/blob/master/run_demo.py#L22

    def __init__(self):
        super().__init__()
        seed_all(SEED)
        self.canvas = tk.Canvas(self, bg="white", width=512, height=512)
        self.canvas.pack(fill=tk.BOTH, expand=True, anchor="nw")
        self.canvas.focus_set()
        self.canvas.bind("<Button-1>", self.mouse_left_click)
        self.canvas.bind("<B1-Motion>", self.mouse_left_move)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<space>", self.display_result)

        self.brush_size = 30
        self.image_size = (512, 512)
        self.image_pairs = get_image_pairs("datasets/coco/annotations/instances_val2017.json", erosion_radius=10)

        self.inpainter = ComodganInpainter(DEVICE)

        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.reset_image()
        self.display()

    def display_result(self, *args, **kwargs):
        self.mask = np.array(self.mask_image)
        image = get_image_tensor(self.real_image).to(DEVICE)
        mask = get_mask_tensor(self.mask, inverted=False).to(DEVICE)
        result = self.inpainter.inpaint(image, mask).cpu().detach().numpy()[0]
        result = (result + 1) / 2
        result = np.transpose(result, (1, 2, 0))
        result = np.uint8((result * 255).clip(0, 255))
        self.canvas.delete("all")

        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(result))
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

    def reset_image(self):
        # self.mask = np.ones((1, 1, *self.image_size), np.uint8)
        self.real_image = cv2.resize(next(self.image_pairs)[0], self.image_size)
        self.mask = np.ones((self.real_image.shape[0], self.real_image.shape[1]), dtype=np.uint8)
        self.mask_image = Image.fromarray(self.mask)
        self.image_drawer = ImageDraw.Draw(self.mask_image)

    def display(self):
        masked_image = self.mask[:, :, np.newaxis] * self.real_image
        # masked_image = masked_image.resize(self.image_size, Image.ANTIALIAS)
        masked_image = Image.fromarray(np.uint8(masked_image))
        self.tk_image = ImageTk.PhotoImage(masked_image)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

    def mouse_left_click(self, event):
        self.mouse_position = event.x, event.y

    def mouse_left_move(self, event):
        if self.mouse_position is None:
            return
        current_position = event.x, event.y

        # mask_image = Image.fromarray(self.mask)
        self.image_drawer.line([self.mouse_position, current_position], fill=0, width=self.brush_size)
        self.image_drawer.ellipse(
            (current_position[0] - self.brush_size // 2, current_position[1] - self.brush_size // 2,
             current_position[0] + self.brush_size // 2, current_position[1] + self.brush_size // 2),
            fill=0)
        self.canvas.create_line(self.mouse_position[0], self.mouse_position[1], current_position[0],
                                current_position[1],
                                fill="black", width=self.brush_size)
        self.canvas.create_oval(current_position[0] - self.brush_size // 2, current_position[1] - self.brush_size // 2,
                                current_position[0] + self.brush_size // 2, current_position[1] + self.brush_size // 2,
                                fill="black", width=self.brush_size)
        self.mouse_position = current_position

        # self.mask = np.array(mask_image)
        # self.mask = self.mask[:, :, 0]

def main():
    app = Paint()
    app.mainloop()

if __name__ == "__main__":
    main()
