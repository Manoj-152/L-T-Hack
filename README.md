# L&T EduTech Hackathon SHAASTRA IITM

Team Name: GiveUsCompute

Team Members: Manoj S, Mukund Varma T

## Introduction

Existing transfer learning methods either choose to finetune all layers, or a seperate linear classifier while freezing the encoder backbone. However, both these methods have certain flaws - the former in the presence of domain drifted data could result in a non-trivial optimum, and the later does not provide us with the best solution. To achieve best performance, one must identify strategies to finetune all the layers such that the intermediate latent distribution remains close to the pretrained model. Towards this end, we propose SDAFT: Self Distill And FineTune - that combines self-distillation and fine-tuning to preserve knowledge from the pretrained model during transfer learning on the downstream task. Our method achieves best results on all three datasets provided in the challenge. 

Please find a more detailed description of our method [here](https://docs.google.com/presentation/d/1q7JkrCpgmlm-cr6tU4qPgVJsrLPTj2a4-o6EXwhv04s/edit?usp=sharing)

## Installation

Clone this repository:

```bash
git clone https://github.com/Manoj-152/L-T-Hack
cd L-T-Hack
```

The code is tested with python 3.8, cuda == 11.6, pytorch == 1.12.1. Additionally dependencies include:

```bash
imgaug
numpy
Pillow
scikit_learn
torchvision
wandb
```

## Datasets

We use the datasets provided by the challenge organizers. All datasets must be downloaded and reorganized into the `torchvision.datasets.ImageFolder` format:
```bash
├──dataset1/
    ├──train/
        ├──image1.png
        ├──image2.png
    ├──val(or test)/
        ├──image1.png
        ├──image2.png
├──dataset2/
...
```

## Usage

### Training

```bash
python3 train_distill.py --use_teacher --dset <crack/ps2/ship> --data_root <dataset-dir> --out_dir <out-dir> --train_epochs 20 --autoaug

# example:
python3 train_distill.py --use_teacher --dset crack --data_root concrete --out_dir concrete_nologits_autoaug --train_epochs 20 --autoaug
python3 train_distill.py --use_teacher --dset ps2 --data_root transformer --out_dir transformer_nologits_autoaug --train_epochs 20 --autoaug
python3 train_distill.py --use_teacher --dset ship --data_root ship --out_dir ship_nologits_autoaug --train_epochs 20 --autoaug
```

### Evaluation

```bash
python3 train_distill.py --use_teacher --dset <crack/ps2/ship> --data_root <dataset-dir> --out_dir <out-dir> --train_epochs 20 --autoaug --use_best --test --resume

# example:
python3 train_distill.py --use_teacher --dset crack --data_root concrete --out_dir concrete_nologits_autoaug --train_epochs 20 --autoaug --use_best --test --resume
python3 train_distill.py --use_teacher --dset ps2 --data_root transformer --out_dir transformer_nologits_autoaug --train_epochs 20 --autoaug --use_best --test --resume
python3 train_distill.py --use_teacher --dset ship --data_root ship --out_dir ship_nologits_autoaug --train_epochs 20 --autoaug --use_best --test --resume
```

## Results

### Dataset I (Concrete Crack Detection)

```bash
metrics
-------------------------
precision: 1.0
recall: 1.0
f1_score: 1.0
kappa_score: 1.0
```

### Dataset II (Power Substation Classification)

```bash
metrics
-------------------------
precision: 0.9444444444444444
recall: 0.9444444444444444
f1_score: 0.9444444444444444
kappa_score: 0.9166666666666666
```

### Dataset III (Ship Classification)

```bash

```
