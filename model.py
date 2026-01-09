import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1) # load pretrained model
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ----------- 1.Stage Training -----------
def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

# ----------- 2.Stage Training -----------
def unfreeze_layer4_and_head(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True

# -----------------
def build_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params=params, lr=lr)