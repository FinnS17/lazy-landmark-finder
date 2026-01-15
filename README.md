# Lazy Landmark Finder

A small landmark classification project focused on robustness. The model is trained on a filtered subset of the GLDv2 dataset (10 iconic landmarks) and evaluated on both clean images and "lazy" corruptions (mild, realistic image degradations). The goal is to reduce the accuracy drop under corruptions without hurting clean performance.

## Results (test set)

| Stage | Clean acc | Lazy acc | Abs drop |
| --- | --- | --- | --- |
| clean_finetune | 0.9589 | 0.7412 | 0.2177 |
| robust | 0.9690 | 0.9106 | 0.0584 |

Metrics come from `results/*/metrics.json` in this repo.

## Dataset

Source: Hugging Face `pemujo/GLDv2_Top_51_Categories` (GLDv2 subset). We filter to 10 landmark classes for a focused experiment:

- Niagara Falls
- Golden Gate Bridge
- Eiffel Tower
- Grand Canyon
- Lake Como
- Masada
- Edinburgh Castle
- Victoria Memorial, Kolkata
- Faisal Mosque
- Jurassic Coast

Please follow the dataset license and terms on Hugging Face.

## Approach

- Backbone: ResNet-18 pretrained on ImageNet.
- Training stages:
  - `clean_head`: freeze backbone, train only the final FC layer.
  - `clean_finetune`: unfreeze layer4 + FC and fine-tune on clean data.
  - `robust`: start from clean_finetune and train with mixed "lazy" augmentations.
- Robustness evaluation: compare clean test accuracy with lazy-corrupted test accuracy.

## Setup

Requirements:
- Python 3.9+
- torch, torchvision, datasets, numpy, matplotlib

Install (example):

```bash
python -m pip install torch torchvision datasets numpy matplotlib
```

## Run

Training and evaluation are controlled in `train.py`:

- Set the stage and training flag:
  - `STAGE = "clean_head" | "clean_finetune" | "robust"`
  - `RUN_TRAIN = True` to train, `False` to only evaluate.

Example:

```bash
python train.py
```

Notes:
- `clean_finetune` expects `models/clean_head.pth`.
- `robust` expects `models/clean_finetune.pth`.
- Evaluation expects `models/<STAGE>.pth`.

## Outputs

- Checkpoints: `models/<stage>.pth`
- Plots and metrics: `results/<stage>/`

## Project Structure

- `train.py`: training and evaluation entry point
- `data.py`: dataset filtering and DataLoader creation
- `transforms.py`: clean and lazy augmentation pipelines
- `model.py`: ResNet-18 setup + training stages
- `engine.py`: training/evaluation loops
- `configs.py`: hyperparameters and dataset class list

## Reproducibility

Seeds are set in `helpers.py` and `configs.py`. Note that lazy evaluation uses random transforms, so metrics can vary slightly between runs unless the evaluation corruption is made deterministic.

## References

- Dataset: GLDv2 subset on Hugging Face (`pemujo/GLDv2_Top_51_Categories`)
- Model: ResNet-18 (torchvision)
