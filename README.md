# NeuroStackNet 🧠

**A Hybrid Stacking Ensemble Model for Brain Tumor Detection Using MRI Images**

An undergraduate thesis project (B.Sc. in Computer Science & Engineering, East Delta University) that compares standalone deep learning architectures against a proposed stacking-based ensemble — **NeuroStackNet** — for multi-class brain tumor classification from MRI scans.


---

## Table of Contents

- [Overview](#overview)
- [Authors](#authors)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Brain tumor diagnosis from MRI scans is traditionally performed through manual visual inspection by radiologists — a process that is slow, subjective, and prone to human error. This project automates that process using deep learning, and proposes **NeuroStackNet**, a hybrid stacking ensemble that combines the predictions of multiple base learners through a trainable meta-learner to boost classification accuracy and robustness.

Four MRI classes are classified:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

## Authors

| Name | Student ID |
|---|---|
| Md Kamrul Hassan | 231002222E |


**Supervisor:** Antu Chowdhury, Lecturer, Dept. of Computer Science & Engineering
**Institution:** East Delta University, Chattogram, Bangladesh — May 2026

## Problem Statement

Manual MRI-based diagnosis doesn't scale well against the growing volume of medical imaging data, and single deep learning models (e.g., a standalone CNN) can overfit or misclassify visually similar tumor types (particularly Glioma vs. Meningioma). Stacking-based ensemble approaches — which combine several base learners under a meta-learner — remain underexplored in this domain. This project investigates whether a stacking ensemble can outperform individual CNN-based architectures for multi-class brain tumor detection, while also providing model interpretability via Explainable AI.

## Dataset

- **Total images:** 7,200 MRI scans, evenly split across 4 classes (1,800 each)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Split ratio:** 70% Train / 10% Validation / 20% Test

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| Train | 1,260 | 1,260 | 1,260 | 1,260 | 5,040 |
| Validation | 180 | 180 | 180 | 180 | 720 |
| Test | 360 | 360 | 360 | 360 | 1,440 |
| **Total** | **1,800** | **1,800** | **1,800** | **1,800** | **7,200** |

Preprocessing steps: image resizing to 128×128, pixel normalization (rescale 1/255), and data augmentation (horizontal flip, rotation, zoom, translation, contrast jitter, and Gaussian noise) applied only to the training split.

> **Note:** The raw dataset (images) is not included in this repository. The notebook expects the images in a directory structure of `Training/<class>/` and `Testing/<class>/`, which are then re-split 70/10/20 into `split_70_10_20/{Train,Validation,Test}/<class>/`. See [How to Run](#how-to-run) for setup.

## Methodology

The end-to-end pipeline follows these stages:

1. **Dataset Preparation** — load and label MRI images from four class folders.
2. **Preprocessing & Splitting** — resize, normalize, augment, and re-split data into Train/Validation/Test (70/10/20) while preserving class balance.
3. **Base Model Training** — train CNN, ResNet50, VGG16, and InceptionV3 independently with hyperparameter tuning.
4. **NeuroStackNet Construction** — stack the prediction probabilities of two base learners (a pretrained **InceptionV3** and a **custom CNN**) into a combined feature vector, then train a fully-connected **meta-learner** on top of it.
5. **Evaluation** — compare all models using Accuracy, Precision, Recall, F1-score, ROC curves, and AUC.
6. **Explainability** — apply SHAP and LIME to interpret NeuroStackNet's predictions.

## Model Architectures

| Model | Type | Notes |
|---|---|---|
| **CNN** | Custom baseline | 3 Conv2D + MaxPooling blocks → Dense(128) → Dropout → Softmax(4) |
| **ResNet50** | Transfer learning | ImageNet-pretrained, frozen base + GAP → Dense(128) → Dropout(0.5) → Softmax |
| **VGG16** | Transfer learning | ImageNet-pretrained, frozen base + GAP → Dense(128) → Dropout(0.5) → Softmax |
| **InceptionV3** | Transfer learning | ImageNet-pretrained, frozen base + GAP → Dense(128) → Dropout(0.5) → Softmax |
| **NeuroStackNet** *(proposed)* | Stacking ensemble | Base learners: InceptionV3 + custom CNN → concatenated probability vectors → meta-learner (Dense(64) → Dropout(0.3) → Softmax(4)) |

## Results

### Individual Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| CNN | 93.06% | 0.93 | 0.93 | 0.93 |
| ResNet50 | 74.65% | 0.74 | 0.75 | 0.74 |
| VGG16 | 87.50% | 0.87 | 0.88 | 0.87 |
| InceptionV3 | 87.29% | 0.87 | 0.87 | 0.87 |
| **NeuroStackNet** | **94.93%** | **0.95** | **0.95** | **0.95** |

### NeuroStackNet — Class-wise Performance

| Class | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|
| Glioma | 0.95 | 0.91 | 0.93 | 0.991 |
| Meningioma | 0.92 | 0.92 | 0.92 | 0.989 |
| No Tumor | 0.97 | 0.97 | 0.97 | 0.997 |
| Pituitary | 0.95 | 1.00 | 0.98 | 0.999 |

NeuroStackNet outperformed every individual base model on the held-out test set, with the biggest gains coming from combining InceptionV3's global feature extraction with the custom CNN's locally learned filters. The overall test error rate was **5.00%** (72 / 1,440 misclassified), concentrated mostly in Glioma ↔ Meningioma confusions — tumor types that are also visually similar to human radiologists.

## Explainable AI (XAI)

To make NeuroStackNet's decisions interpretable (important for clinical trust), two XAI techniques were applied on top of the trained meta-learner's inputs (the stacked base-model probability outputs):

- **SHAP (SHapley Additive exPlanations)** — global feature attribution showing how much each base model's class-probability output contributes to the final decision.
- **LIME (Local Interpretable Model-agnostic Explanations)** — local, per-instance explanations showing which base-model outputs pushed a specific prediction toward or away from a class.



## Repository Structure

```
.
├── neurostack.ipynb       # Full implementation: data pipeline, all models, training, evaluation
└── README.md               # This file
```

> The notebook was authored and run in **Google Colab**; it mounts Google Drive and extracts the dataset from a `.zip` archive. Paths will need to be adjusted for local execution (see below).

## Requirements

The notebook relies on the following core libraries:

```
tensorflow>=2.x
numpy
pandas
matplotlib
scikit-learn
Pillow
```

Install them with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn Pillow
```

For the Explainable AI analysis (SHAP / LIME plots referenced in the report), also install:

```bash
pip install shap lime
```

## How to Run

1. **Get the dataset.** This project uses a 4-class brain tumor MRI dataset (Glioma / Meningioma / Pituitary / No Tumor) organized into `Training/<class>/` and `Testing/<class>/` folders.
    - Dataset Link- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Training     
2. **Adapt the paths.** The notebook was built for Google Colab and references Google Drive paths (e.g. `/content/drive/MyDrive/...`) and `google.colab` calls (`drive.mount`, `files.download`, `files.upload`). To run locally or on another platform:
   - Replace `zip_file_path` / `extracted_folder_path` with your local dataset location.
   - Remove or replace the `google.colab` import/mount/upload/download cells with standard file I/O.
3. **Run the pipeline in order:**
   - Load & preprocess data → EDA (class distribution) → 70/10/20 re-split
   - Train the baseline **CNN** (includes a small grid search over learning rate and dropout)
   - Train **ResNet50**, **VGG16**, and **InceptionV3** (transfer learning, frozen base)
   - Train **NeuroStackNet** (InceptionV3 + CNN base learners → meta-learner)
4. **Evaluate & visualize** — each model section outputs a classification report, confusion matrix, and ROC/AUC curves.
5. **Run inference on a single image** using the provided `predict()` helper / upload cells to classify a new MRI scan.



## Limitations

- Dataset size and diversity are limited (single source, no multi-center validation).
- ResNet50 underperformed relative to expectations, likely due to insufficient fine-tuning of the frozen pretrained layers.
- No external dataset validation was performed.
- The stacking ensemble adds computational overhead compared to a single model.
- Only SHAP and LIME were explored for interpretability; no clinical validation was conducted.

## Future Work

- Train on larger, multi-institution MRI datasets for better generalization.
- Explore Vision Transformers (ViT) and hybrid CNN-Transformer architectures.
- Fine-tune pretrained backbones (rather than freezing them) and expand hyperparameter search.
- Deploy as a web/mobile decision-support tool integrated with hospital systems.
- Extend to tumor segmentation and severity grading.
- Validate with domain experts (radiologists/clinicians) and external datasets.

## Acknowledgments

This project was completed in partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Science and Engineering at **East Delta University**, under the supervision of **Antu Chowdhury** (Lecturer, Dept. of CSE).

## License

No license has been specified for this repository. All rights are reserved by the authors unless a license file is added. If you intend to reuse or build on this work, please reach out to the authors or add an appropriate open-source license (e.g., MIT, Apache 2.0).
