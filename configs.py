# ==============
# GENERAL 
# ==============
SEED = 42
BATCH_SIZE = 8

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

MODELS_DIR = "models"
RESULTS_DIR = "results"

# ==============
# DATA
# ==============
KEEP_NAMES = ("Niagara Falls", "Golden Gate Bridge", "Eiffel Tower","Grand Canyon",
              "Lake Como","Masada","Edinburgh Castle","Victoria Memorial, Kolkata",
              "Faisal Mosque", "Jurassic Coast")

# ==============
# TRAINING 
# ==============
TRAIN_CONFIG= {
    "clean_head": {"epochs": 15, "lr": 1e-3} ,
    "clean_finetune": {"epochs": 5, "lr": 1e-4},
    "robust": {"epochs": 5, "lr": 5e-5},
}