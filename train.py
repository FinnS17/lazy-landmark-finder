import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from configs import TRAIN_CONFIG, SEED, MODELS_DIR, RESULTS_DIR
from datasets import load_dataset
from data import prepare_splits, make_loaders
from transforms import get_eval_clean_transform, get_eval_lazy_tranform, get_train_clean_transform, get_train_robust_transform
from model import build_model, freeze_backbone, unfreeze_layer4_and_head, build_optimizer
from helpers import set_seed, get_device, plot_and_save_curves
from engine import train_one_epoch, evaluate

set_seed(SEED)
dataset = load_dataset("pemujo/GLDv2_Top_51_Categories")

train_split, val_split, test_split, categories = prepare_splits(dataset)

train_transform = get_train_clean_transform()
eval_transform = get_eval_clean_transform()
train_loader, val_loader, test_loader = make_loaders(train_split, val_split, test_split, train_transform, eval_transform)


# ===============
# SET STAGE
# ===============
STAGE = "clean_finetune" # clean_head | clean_finetune | robust


print(f"Train Stage: {STAGE}")
path = f"{MODELS_DIR}/{STAGE}.pth"
os.makedirs(MODELS_DIR, exist_ok=True)


criterion = nn.CrossEntropyLoss()
device = get_device()
model = build_model(len(categories))
model.to(device)
print(f"Using device: {device}")


cfg = TRAIN_CONFIG.get(STAGE)
if cfg is None:
    raise ValueError("invalid stage")

if STAGE == "clean_head":
    freeze_backbone(model)
    num_epochs = cfg["epochs"]
    lr = cfg["lr"]
    optimizer = build_optimizer(model, lr)    

elif STAGE == "clean_finetune":
    state_dict = torch.load(f"{MODELS_DIR}/clean_head.pth", map_location=device)
    model.load_state_dict(state_dict)
    unfreeze_layer4_and_head(model)
    num_epochs = cfg["epochs"]
    lr = cfg["lr"]
    optimizer = build_optimizer(model, lr)    

elif STAGE == "robust":
    state_dict = torch.load(f"{MODELS_DIR}/clean_finetune.pth", map_location=device)
    model.load_state_dict(state_dict)
    unfreeze_layer4_and_head(model)
    num_epochs = cfg["epochs"]
    lr = cfg["lr"]
    optimizer = build_optimizer(model, lr)

print("\nStart of Training")
best_acc = 0.0 # for saving best model

train_losses = []
val_losses = []
val_accs = []

# ------- Training Loop -------
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, acc = evaluate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(acc)
    print(f"Epoch: {epoch+1} | Train-Loss: {train_loss:.4f} | Val-Loss: {val_loss:.4f} | Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), path)
        print(f"Model saved after epoch: {epoch +1}")

# ------- Loss Curves --------
stage_dir = os.path.join(RESULTS_DIR, STAGE)
os.makedirs(stage_dir, exist_ok= True)

epochs = range(1, len(train_losses) +1) # X-axis

curves = {"Train Loss": train_losses,"Val Loss": val_losses}
save_path = os.path.join(stage_dir, f"{STAGE}_loss_curves.png")
plot_and_save_curves(epochs, curves, "Epoch", "Loss", f"Loss Curves ({STAGE})", save_path)

curves = {"Val Accuracy": val_accs}
save_path = os.path.join(stage_dir, f"{STAGE}_validation_accuracy.png")
plot_and_save_curves(epochs, curves, "Epoch", "Accuracy", f"Validation Accuracy ({STAGE})", save_path)


# ------- Evaluation of best model on test set
model = build_model(len(categories))
state_dict = torch.load(path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)

print("Evaluate best Model on test set: ")
avg_loss, acc = evaluate(model, test_loader, criterion, device)
print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} ")
