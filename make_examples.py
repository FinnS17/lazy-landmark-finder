from datasets import load_dataset
from transforms import get_eval_lazy_transform, get_eval_clean_transform
import random
import matplotlib.pyplot as plt
from configs import RESULTS_DIR, IMAGENET_MEAN, IMAGENET_STD
import os
import torch
import numpy as np

dataset = load_dataset("pemujo/GLDv2_Top_51_Categories")
lazy_eval_transform = get_eval_lazy_transform()
clean_eval_transform = get_eval_clean_transform()

mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
std = torch.tensor(IMAGENET_STD).view(3,1,1)

def to_display_img(x):
    x = x*std + mean
    x = x.clamp(0,1)
    x = x.permute(1,2,0).cpu().numpy()
    return x

selected_test_images = []

for x in iter(random.sample(range(100), 4)):
    selected_test_images.append(dataset["test"][x]["image"])

fix, axes = plt.subplots(nrows=len(selected_test_images), ncols=2, figsize=(8, 12))

for i, image in enumerate(selected_test_images):
    clean = clean_eval_transform(image)
    lazy = lazy_eval_transform(image)

    axes[i, 0].imshow(to_display_img(clean))
    axes[i, 0].set_title("Clean")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(to_display_img(lazy))
    axes[i, 1].set_title("Lazy (corrupted)")
    axes[i, 1].axis("off")
plt.tight_layout()
              
path = f"{RESULTS_DIR}/examples"
os.makedirs(path, exist_ok=True)
plt.savefig(f"{path}/clean_vs_lazy_examples.png", dpi=200, bbox_inches="tight")

plt.show()