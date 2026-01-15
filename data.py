from datasets import load_dataset
from urllib.parse import unquote
import torch

from configs import KEEP_NAMES, SEED, BATCH_SIZE
from helpers import HFDatasetTorch

def pretty(name):
    return unquote(name).replace("_", " ")

def filter_and_relabel(split_ds, keep_names = KEEP_NAMES):    
    # only keep certain monuments
    keep_indices = []
    for i, ex in enumerate(split_ds):
        cat = pretty(ex["category"])
        if cat in keep_names:
            keep_indices.append(i)
    filtered_dataset = split_ds.select(keep_indices)

    # relabel examples
    label_mapping = {}
    categories = sorted(list(keep_names))
    for i, cat in enumerate(categories):
        label_mapping[cat] = i

    def relabel_example(ex):
        cat = pretty(ex["category"])
        new_ex = ex.copy()
        new_ex["label"] = label_mapping[cat]
        return new_ex

    return filtered_dataset.map(relabel_example), categories

def prepare_splits(dataset, val_ratio = 0.1):
    """
    Prepare train/val/test splits and category mapping

    Returns: train_split, val_split, test_split, categories
    """

    train, categories = filter_and_relabel(dataset["train"])
    split = train.train_test_split(test_size=val_ratio, seed=SEED)
    train_split = split["train"]
    val_split = split["test"]
    test_split, _ = filter_and_relabel(dataset["test"])

    return train_split, val_split, test_split, categories


def make_loaders(train_split, val_split, test_split, train_transform, eval_transform, lazy_eval_transform, train_robust_transform):
    # create HFDatasetTorch objects -> DataLoader works with that
    train_ds = HFDatasetTorch(train_split, train_transform)
    val_ds = HFDatasetTorch(val_split, eval_transform)
    test_ds = HFDatasetTorch(test_split, eval_transform)
    lazy_test_ds = HFDatasetTorch(test_split, lazy_eval_transform)
    train_robust_ds = HFDatasetTorch(train_split, train_robust_transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    lazy_test_loader = torch.utils.data.DataLoader(lazy_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    robust_train_loader = torch.utils.data.DataLoader(train_robust_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader, lazy_test_loader, robust_train_loader

    
    


    



