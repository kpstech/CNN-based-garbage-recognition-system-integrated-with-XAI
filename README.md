# Garbage Classification System using EfficientNetB0

An AI-powered waste classification system that automatically identifies and categorizes garbage images using deep learning, complete with explainable AI visualizations and a desktop GUI.

## Overview

Manual waste sorting is slow, inconsistent, and labor-intensive. This project uses transfer learning to automatically classify waste images into distinct categories, helping enable smarter, faster, and more consistent recycling workflows.

The model is built on **EfficientNetB0**, fine-tuned on the **TrashNet** dataset, and achieves **~90–92% test accuracy**. To make the system transparent and trustworthy, three explainability techniques — **Grad-CAM**, **LIME**, and **SHAP** — are integrated to visually show which parts of an image influenced each prediction.

## Features

- **Deep Learning Classification** – EfficientNetB0 (transfer learning) fine-tuned for waste category prediction
- **High Accuracy** – ~90–92% accuracy on the TrashNet dataset
- **Explainable AI (XAI)** – Grad-CAM, LIME, and SHAP visualizations for model interpretability
- **Desktop GUI** – Built with CustomTkinter for real-time image upload and classification
- **Multi-class Support** – Classifies waste into categories such as plastic, paper, metal, glass, cardboard, and trash

## Dataset

- **Source:** [TrashNet](https://github.com/garythung/trashnet)
- **Classes:** Cardboard, Glass, Metal, Paper, Plastic, Trash
- Images were preprocessed (resized, normalized, augmented) before training

## Tech Stack

| Component | Technology |
|---|---|
| Model Architecture | EfficientNetB0 (Transfer Learning) |
| Framework | TensorFlow / Keras |
| Explainability | Grad-CAM, LIME, SHAP |
| GUI | CustomTkinter |
| Language | Python |

## Project Structure

```
garbage-classification/
├── dataset/                 # TrashNet dataset (train/val/test splits)
├── models/                  # Saved trained model weights
├── notebooks/                # Training & experimentation notebooks
├── explainability/           # Grad-CAM, LIME, SHAP scripts
├── gui/                       # CustomTkinter desktop app
│   └── app.py
├── src/
│   ├── train.py               # Model training script
│   ├── preprocess.py          # Data preprocessing utilities
│   └── predict.py             # Inference script
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd garbage-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python src/train.py
```

**Run predictions on a single image:**
```bash
python src/predict.py --image path/to/image.jpg
```

**Launch the desktop GUI:**
```bash
python gui/app.py
```

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~90–92% |
| Model | EfficientNetB0 |
| Dataset | TrashNet |

## Explainability

The system uses three complementary techniques to explain predictions:

- **Grad-CAM** – Highlights the regions of the image most responsible for the predicted class
- **LIME** – Perturbs image segments to explain local model behavior
- **SHAP** – Assigns contribution scores to image regions based on game-theoretic principles


## Acknowledgements

- [TrashNet Dataset](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang
- EfficientNet architecture by Tan & Le (Google Research)

## 📜 License

This project is for academic purposes. Add a license of your choice here (e.g., MIT, Apache 2.0).
