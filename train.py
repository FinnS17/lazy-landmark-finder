import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import json
import numpy as np

from configs import TRAIN_CONFIG, SEED, MODELS_DIR, RESULTS_DIR
from datasets import load_dataset
from data import prepare_splits, make_loaders
from transforms import get_eval_clean_transform, get_eval_lazy_transform, get_train_clean_transform, get_train_robust_transform
from model import build_model, freeze_backbone, unfreeze_layer4_and_head, build_optimizer
from helpers import set_seed, get_device, plot_and_save_curves
from engine import train_one_epoch, evaluate

set_seed(SEED)
dataset = load_dataset("pemujo/GLDv2_Top_51_Categories")

train_split, val_split, test_split, categories = prepare_splits(dataset)

train_transform = get_train_clean_transform()
clean_eval_transform = get_eval_clean_transform()
lazy_eval_transform = get_eval_lazy_transform()
train_robust_transform = get_train_robust_transform()
train_loader, val_loader, test_loader, lazy_test_loader, robust_train_loader, lazy_val_loader = make_loaders(train_split, val_split, test_split, train_transform, clean_eval_transform, lazy_eval_transform, train_robust_transform)



# ===============
# SET STAGE and TRAINING
STAGE = "clean_head" # clean_head | clean_finetune | robust
RUN_TRAIN = True
# ===============


path = f"{MODELS_DIR}/{STAGE}.pth"
os.makedirs(MODELS_DIR, exist_ok=True)

criterion = nn.CrossEntropyLoss()
device = get_device()
model = build_model(len(categories))
model.to(device)
print(f"Using device: {device}")

if RUN_TRAIN:
    print(f"Train Stage: {STAGE}")
    cfg = TRAIN_CONFIG.get(STAGE)
    if cfg is None:
        raise ValueError("invalid stage")

    if STAGE == "clean_head":
        freeze_backbone(model)
        num_epochs = cfg["epochs"]
        lr = cfg["lr"]
        optimizer = build_optimizer(model, lr)    

    elif STAGE == "clean_finetune":
        model_loc = f"{MODELS_DIR}/clean_head.pth"
        assert os.path.isfile(model_loc), f"Checkpoint not found {model_loc}"
        state_dict = torch.load(model_loc, map_location=device)
        model.load_state_dict(state_dict)
        unfreeze_layer4_and_head(model)
        num_epochs = cfg["epochs"]
        lr = cfg["lr"]
        optimizer = build_optimizer(model, lr)    

    elif STAGE == "robust":
        model_loc = f"{MODELS_DIR}/clean_finetune.pth"
        assert os.path.isfile(model_loc), f"Checkpoint not found {model_loc}"
        state_dict = torch.load(model_loc, map_location=device)
        model.load_state_dict(state_dict)
        unfreeze_layer4_and_head(model)
        num_epochs = cfg["epochs"]
        lr = cfg["lr"]
        optimizer = build_optimizer(model, lr)


    print("\nStart of Training")
    best_acc = 0.0 # for saving best model
    score = 0.0 # for Stage-Handling

    train_losses = []
    val_losses = []
    val_accs = []
    if STAGE == "robust":
        lazy_val_losses = []
        lazy_val_accs = []

    # ------- Training Loop -------
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, robust_train_loader if STAGE == "robust" else train_loader, criterion, optimizer, device)
        val_loss, acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)
        score = acc

        if STAGE == "robust":
            lazy_val_loss, lazy_val_acc = evaluate(model, lazy_val_loader, criterion, device)
            lazy_val_losses.append(lazy_val_loss)
            lazy_val_accs.append(lazy_val_acc)
            score = lazy_val_acc
            print(f"Epoch: {epoch+1} | Train-Loss: {train_loss:.4f} | Lazy-Val-Loss: {lazy_val_loss:.4f} | Clean-Val-Loss: {val_loss:.4f} | Lazy-Accuracy: {lazy_val_acc:.4f} | Clean-Accuracy: {acc:.4f}")
        else: 
            print(f"Epoch: {epoch+1} | Train-Loss: {train_loss:.4f} | Val-Loss: {val_loss:.4f} | Accuracy: {acc:.4f}")
        if score > best_acc:
            best_acc = score
            torch.save(model.state_dict(), path)
            print(f"Model saved after epoch: {epoch +1}")

    # ------- Loss Curves --------
    stage_dir = os.path.join(RESULTS_DIR, STAGE)
    os.makedirs(stage_dir, exist_ok= True)

    epochs = range(1, len(train_losses) +1) # X-axis

    curves = {"Train Loss": train_losses,"Clean Val Loss": val_losses}
    save_path = os.path.join(stage_dir, f"{STAGE}_loss_curves.png")
    plot_and_save_curves(epochs, curves, "Epoch", "Loss", f"Loss Curves ({STAGE})", save_path)

    curves = {"Val Accuracy": val_accs}
    save_path = os.path.join(stage_dir, f"{STAGE}_validation_accuracy.png")
    plot_and_save_curves(epochs, curves, "Epoch", "Accuracy", f"Validation Accuracy ({STAGE})", save_path)

    if STAGE == "robust":
        curves = {"Clean Val Loss": val_losses, "Lazy Val Loss": lazy_val_losses}
        save_path = os.path.join(stage_dir, "Robustness_loss_curves.png")
        plot_and_save_curves(epochs, curves, "Epoch", "Loss", "Robustness Loss Curves" ,save_path)

        curves = {"Clean Val Accuracy": val_accs, "Lazy Val Accuracy": lazy_val_accs}
        save_path = os.path.join(stage_dir, "clean_vs_lazy_val_accuracy.png")
        plot_and_save_curves(epochs, curves, "Epoch", "Accuracy", "Lazy Validation Accuracy", save_path)

# ------- Evaluation of model on test set
print("Evaluation Stage")
model = build_model(len(categories))
assert os.path.isfile(path), f"Model not found: {path}"
state_dict = torch.load(path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)

print("\nEvaluate best Model on CLEAN test set")
clean_loss, clean_acc = evaluate(model, test_loader, criterion, device)
print(f"Loss: {clean_loss:.4f} | Accuracy: {clean_acc:.4f}")

# --------- Evaluation of model on Lazy (corrupted) Images
print("Evaluate Model on LAZY (corrupted) test set")
lazy_losses = []
lazy_accs = []
for i in range(5):
    lazy_loss, lazy_acc = evaluate(model, lazy_test_loader, criterion, device)
    lazy_losses.append(lazy_loss)
    lazy_accs.append(lazy_acc)
print(f"Loss: {np.mean(lazy_losses):.4f} | Accuracy: {np.mean(lazy_accs):.4f}")

# ---------- Save metrics to json
stage_dir = os.path.join(RESULTS_DIR, STAGE)
os.makedirs(stage_dir, exist_ok=True) # in case TRAIN = False
metrics_path = os.path.join(stage_dir, "metrics.json")
metrics = {
    "stage": STAGE,
    "seed": SEED,
    "checkpoint": path,
    "clean_test": {
        "loss": float(clean_loss),
        "accuracy": float(clean_acc),
    },
    "lazy_test": {
        "loss": float(np.mean(lazy_losses)),
        "accuracy": float(np.mean(lazy_accs)),
    },
    "lazy_drop_abs": float(clean_acc-np.mean(lazy_accs))
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Saved metrics to {metrics_path}")

