import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from configs import IMAGENET_MEAN, IMAGENET_STD

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
         device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

# ----- plot and save curves ------
def plot_and_save_curves(x, curves: dict, xlabel, ylabel, title, save_path):
    plt.figure()
    for label, values in curves.items():
        plt.plot(x, values, label=label)

    plt.xlabel(xlabel)
    plt.xticks(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, dpi=200, bbox_inches = "tight")
    print(f"Saved plot to: {save_path}")
    plt.show()
    


# ----- unnormalize picture for visualization of corrupted vs. uncorrupted -------
def unnormalize(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    return x * std + mean

# ---------- pytorch dataset wrapper -----
class HFDatasetTorch:
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    #pytorch needs following methods to automatically work with them

    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        img = self.hf_dataset[idx]["image"]
        x = self.transform(img)
        y = int(self.hf_dataset[idx]["label"])
        return x, y
    
