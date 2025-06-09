import cv2
import numpy as np
import os
from tqdm import tqdm

# Configuration
output_dir = "dataset"
classes = {
    "longitudinal": 90,      # vertical
    "transverse": 0,         # horizontal
    "cross_plied": [0, 90],  # horizontal + vertical
    "angle_plied": 45        # diagonal
}

img_size = 224
num_images = 100

def draw_lines(angle, density=10):
    """Generates an image with lines at a given angle."""
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255
    spacing = img_size // density

    if isinstance(angle, list):  # For cross-plied
        for a in angle:
            draw_lines_single(img, a, spacing)
    else:
        draw_lines_single(img, angle, spacing)

    return img

def draw_lines_single(img, angle, spacing):
    theta = np.deg2rad(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    
    for d in range(-img_size, img_size*2, spacing):
        x0 = int(d * cos_a)
        y0 = int(d * sin_a)
        x1 = int((d + img_size) * cos_a)
        y1 = int((d + img_size) * sin_a)
        cv2.line(img, (x0, y0), (x1, y1), (0,), 2)

def generate_images():
    for cls, angle in tqdm(classes.items(), desc="Generating images"):
        class_dir = os.path.join(output_dir, "train", cls)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_images):
            img = draw_lines(angle)
            filename = os.path.join(class_dir, f"{cls}_{i}.png")
            cv2.imwrite(filename, img)

generate_images()

output_dir = os.path.abspath(output_dir)