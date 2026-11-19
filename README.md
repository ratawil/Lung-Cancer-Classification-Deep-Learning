# Lung Cancer Histopathology Classification (CNN vs. Vision Transformer)

A deep learning computer vision pipeline built with TensorFlow and Keras to classify lung tissue histopathological images into benign (`lung_n`) and squamous cell carcinoma (`lung_scc`) categories. The notebook benchmarks standard pre-trained Convolutional Neural Networks (CNNs) against Vision Transformer (ViT) architectures.

## Overview

Automated classification of histopathological slides assists in early, accurate pathology screening. This project initializes a GPU-accelerated Google Colab workflow that loads and extracts image datasets from Google Drive, applies image preprocessing, and trains multiple deep learning architectures for comparative evaluation.

## Dataset

This pipeline uses the **LC25000** (Lung and Colon Cancer Histopathological Image Dataset) lung subset:
* **`lung_n`**: Benign lung tissue
* **`lung_scc`**: Lung squamous cell carcinoma

*Note: Update the Google Drive mount path (`/content/drive/MyDrive/dataset/...`) in the notebook to match your local Drive directory structure before running.*

## Model Architectures

The environment imports and configures the following model families for transfer learning and custom feature extraction:
* **Convolutional Neural Networks (CNNs):**
  * `VGG16`
  * `ResNet50`
  * `MobileNetV3Large`
* **Vision Transformers (ViT):**
  * Pre-trained ViT models via `vit-keras`
  * Custom Transformer encoder layers (`MultiHeadAttention`, `LayerNormalization`, `Embedding`)

## Tech Stack & Dependencies

* **Deep Learning:** TensorFlow, Keras, `vit-keras`, `tensorflow-addons`
* **Image Processing:** OpenCV (`cv2`), Pillow
* **Data Manipulation & Evaluation:** NumPy, Pandas, Scikit-Learn (`train_test_split`, `classification_report`, `confusion_matrix`)
* **Visualization:** Matplotlib, Seaborn

## Getting Started

1. **Open in Google Colab:** Upload the `.ipynb` notebook to Google Colab and enable GPU acceleration (`Runtime` > `Change runtime type` > `T4 GPU` or higher).
2. **Prepare Data:** Place the zipped LC25000 dataset in your Google Drive and verify the extraction path in the data setup cell.
3. **Install Dependencies:** Run the initial setup cell to install `vit-keras` and `tensorflow-addons`:
   ```bash
   pip install vit-keras tensorflow-addons
