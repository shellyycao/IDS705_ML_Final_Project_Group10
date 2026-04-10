# IDS705_ML_Final_Project_Group10
# Vulnerability of Medical AI: Evaluating Adversarial Attacks on Diagnostic Imaging
### A study on the robustness of Convolutional Neural Networks using MedMNIST
**IDS 705 — Machine Learning | Duke University | Group 10**

---

## Team Members

| Name | Responsibility |
|------|---------------|
| Shelly Cao |  |
| Arvind Kandala | TBD |
| Diwas Puri | TBD |
| Sebine Scaria | TBD |

---

## Project Overview

Deep learning models have shown strong performance in medical imaging tasks such as disease classification from chest X-rays. However, these models are typically evaluated on clean, curated datasets that do not reflect real-world conditions. In practice, medical images may undergo compression, resizing, or degradation due to storage systems, transmission pipelines, or preprocessing steps.

This project evaluates how the performance of medical image classification models degrades under realistic compression and corruption conditions, and compares robustness across different model architectures.

**Research questions:**
- Which model is more robust overall — ResNet-18 or a small CNN?
- Which corruption type leads to the largest performance degradation?
- Does performance decline gradually or sharply as corruption severity increases?
- Are models more sensitive to compression or noise-based transformations?

---

## Dataset

**PneumoniaMNIST+** from the MedMNIST v2 collection at 224×224 resolution.

| Split | Images |
|-------|--------|
| Train | 4,708  |
| Val   | 524    |
| Test  | 624    |

- **Task**: Binary classification (Normal vs Pneumonia)
- **Source**: Chest X-ray images (grayscale)
- **Zenodo**: [records/10519652](https://zenodo.org/records/10519652) — download `pneumoniamnist_224.npz`

---

## Model

**ResNet-18** pretrained by the MedMNIST team (224×224).

- **Weights Zenodo**: [records/7782114](https://zenodo.org/records/7782114) — download `weights_pneumoniamnist.zip`
- **File used**: `resnet18_224_1.pth`
- **Architecture**: Standard torchvision ResNet-18, `num_classes=2`, CrossEntropyLoss

---

## Corruptions Evaluated

Each corruption is applied at 5 severity levels on the test set only. The model is trained on clean data throughout.

| Corruption | Severity Levels | Status |
|------------|----------------|--------|
| Gaussian Blur | k=3→21, σ=0.5→5.0 | |
| JPEG Compression |  | |
| Downsample → Upsample |  |  |
| Gaussian / Poisson Noise |  |  |
| Brightness / Contrast |  | |

---

## Sample Images

### Original vs Corrupted Test Images

| Corruption | Level 1 | Level 3 | Level 5 |
|------------|---------|---------|---------|
| Gaussian Blur | ![](link) | ![](link) | ![](link) |
| JPEG Compression | ![](link) | ![](link) | ![](link) |
| Downsample → Upsample | ![](link) | ![](link) | ![](link) |
| Gaussian / Poisson Noise | ![](link) | ![](link) | ![](link) |
| Brightness / Contrast | ![](link) | ![](link) | ![](link) |

### Original (Clean)
| Normal | Pneumonia |
|--------|-----------|
| ![](link) | ![](link) |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `model_loader.ipynb` | Downloads and verifies the pretrained ResNet-18 |
| `gaussian_blur.ipynb` | Applies Gaussian blur at 5 severity levels and evaluates robustness |
| `jpeg_compression.ipynb` | TBD |
| `downsample_upsample.ipynb` | TBD |
| `noise.ipynb` | TBD |
| `brightness_contrast.ipynb` | TBD |
| `small_cnn_train.ipynb` | Trains the small CNN baseline from scratch |
| `comparison.ipynb` | Compares ResNet-18 vs small CNN across all corruptions |

---

## Key Findings

### Gaussian Blur
- Clean baseline accuracy: ~0.91, AUROC: ~0.97
- Level 5 blur (k=21, σ=5.0) accuracy: ~0.86, AUROC: ~0.95
- **Finding**: ResNet-18 is highly robust to Gaussian blur. Recall stays near 1.0 across all severity levels, suggesting the model rarely misses pneumonia cases even under heavy blurring. This may be because pneumonia diagnosis relies on coarse features that survive blurring.

*More findings will be added as experiments are completed.*

---

## How to Run

### Requirements
```bash
pip install medmnist scikit-learn torch torchvision matplotlib pandas
```

### Steps
1. Clone this repo
```bash
git clone https://github.com/shellyycao/IDS705_ML_Final_Project_Group10.git
```
2. Open any notebook in Google Colab
3. Run all cells top to bottom — dataset and weights download automatically
4. No manual setup required

> **Note**: A GPU runtime is recommended in Colab (`Runtime → Change runtime type → T4 GPU`)

---

## Data Sources

- **Dataset**: Yang et al., "MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", *Scientific Data*, 2023. [DOI](https://doi.org/10.5281/zenodo.10519652)
- **Weights**: MedMNIST Experiments. [DOI](https://doi.org/10.5281/zenodo.7782114)

---

## Ethical Considerations

- This project is for **research and educational purposes only**
- The dataset is publicly available, anonymized, and licensed under CC BY 4.0
- Results should **not** be used for clinical decision-making
- Findings are presented as controlled experimental results, not claims of clinical readiness

---

## License
This project is licensed under the MIT License.
