# Cats vs. Dogs Image Classification (CNN)

A deep learning binary image classification pipeline built with **TensorFlow/Keras** and **OpenCV** to classify images of cats and dogs. The project covers end-to-end data ingestion via `kagglehub`, convolutional feature extraction, training/validation performance tracking, and custom image inference.

---

## Key Features
* **Direct Kaggle Integration:** Pulls and caches the `dogsvscats` dataset directly in Google Colab using `kagglehub` without manual zip uploads.
* **CNN Architecture:** Multi-layer Convolutional Neural Network trained on `(256, 256, 3)` RGB image tensors with `binary_crossentropy` loss and sigmoid probability output.
* **Training Diagnostics:** Visualizes epoch-by-epoch training vs. validation accuracy curves using Matplotlib to monitor convergence and generalization.
* **Custom Image Inference:** Includes an OpenCV (`cv2`) preprocessing pipeline handling BGR-to-RGB color conversion, resizing, tensor reshaping, and probability scoring on unseen test images.

---

## Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision & Data:** OpenCV (`cv2`), NumPy, KaggleHub
* **Visualization:** Matplotlib
* **Environment:** Google Colab / Jupyter Notebook

---

## Project Structure
```text
├── cats_vs_dogs_cnn.ipynb         # Main training, evaluation, and inference notebook
├── test_1_dog_cnn_project.jpg     # Sample test image for custom inference
├── sample_data/                   # Lightweight sample subset for quick testing
└── README.md
