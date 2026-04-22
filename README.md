# IDS705 ML Final Project — Group 10
Investigating the vulnerability of medical imaging AI to adversarial attacks.

## Project Overview
This project studies how adversarial attacks affect deep learning models used in medical image classification (PneumoniaMNIST via MedMNIST). We evaluate two architectures — ResNet18 and ResNet50 — under FGSM-based attacks, and explore the robustness of DASYNet before and after fine-tuning.

## Repository Structure

```
notebooks/
├── 01_experimental_attacks/     # Exploratory attack methods (brightness/contrast, Gaussian blur, up/downscaling)
├── 02_adversarial_fgsm_attack/
│   ├── resnet18/                # FGSM attack study on ResNet18
│   └── resnet50/                # FGSM attack study on ResNet50
├── 03_dasynet/
│   ├── training/                # DASYNet training and baseline attack evaluation
│   └── weights/                 # dasynet_pneumonia.pth
└── 04_fine_tuning/
    ├── full_image/              # Full-image DASYNet fine-tuning and attack study
    └── grid/                    # Grid-based DASYNet fine-tuning and attack study

model weights/                   # All trained model weights (.pth)
image prep/                         # Utility scripts (corrupted image generation)
```

## Models
| File | Description |
|---|---|
| `dasynet_pneumonia.pth` | DASYNet trained on PneumoniaMNIST |
| `dasynet_finetuned.pth` | DASYNet fine-tuned (full image) |
| `dasynet_finetuned_grid.pth` | DASYNet fine-tuned (grid attack) |

## Key Findings
- FGSM attacks significantly degrade classification accuracy even at low epsilon values
- DASYNet shows improved robustness after adversarial fine-tuning
- Grid-based attacks expose localized vulnerabilities not captured by global perturbation methods

## Team
Group 10 — IDS 705, Duke University
