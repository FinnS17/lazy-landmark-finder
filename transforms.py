from torchvision import transforms
from configs import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

# ----- for clean training set
def get_train_clean_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean= IMAGENET_MEAN, std= IMAGENET_STD)
    ])

# ----- for robust training set
def get_train_robust_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8,1.25)),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.8, 1.8)),
            transforms.ColorJitter(brightness=(0.2, 0.6), contrast=(0.2, 0.6)),
            ], p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean= IMAGENET_MEAN, std= IMAGENET_STD)])

# ----- for clean validation/testing
def get_eval_clean_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean= IMAGENET_MEAN, std= IMAGENET_STD)
    ])

# ----- for lazy validation/testing
def get_eval_lazy_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 0.9), ratio=(1,1)),
        transforms.GaussianBlur(kernel_size=7, sigma=1.5),
        transforms.ColorJitter(brightness=0.6, contrast=0.6),
        transforms.ToTensor(),
        transforms.Normalize(mean= IMAGENET_MEAN, std= IMAGENET_STD)
    ])