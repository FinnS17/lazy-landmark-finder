from torchvision import transforms
from configs import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

def get_train_clean_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean= IMAGENET_MEAN, std= IMAGENET_STD)
    ])

def get_train_robust_transform():
    pass

def get_eval_clean_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean= IMAGENET_MEAN, std= IMAGENET_STD)
    ])

def get_eval_lazy_tranform():
    pass