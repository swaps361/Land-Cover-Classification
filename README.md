# Satellite Image Classification Using AlexNet

This project is a Satellite Image Classification System built using the EuroSAT dataset. The goal of the project is to classify satellite images into 10 distinct land cover categories using a deep learning model. The model is based on AlexNet, a widely-used Convolutional Neural Network (CNN) architecture.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

The project leverages the EuroSAT dataset, which includes satellite images with 10 land cover classes such as Annual Crop, Forest, Residential, and more. The classification model is built using the AlexNet architecture with modifications to suit the nature of the dataset. The project utilizes techniques like data augmentation, early stopping, and learning rate adjustments to improve model performance.

## Dataset

The dataset consists of satellite images divided into 10 classes, with the following distribution:

- **Training Set**: 17,280 images
- **Validation Set**: 4,320 images
- **Testing Set**: 5,400 images

The dataset is balanced across these 10 classes. The images are preprocessed and augmented to enhance model generalization.

## Model Architecture

The model used for image classification is based on **AlexNet**. Key features of the architecture are:

- **Input Layer**: Accepts images with shape (224x224x3).
- **Convolutional Layers**: Multiple convolutional layers to extract features.
- **Batch Normalization**: Improves training stability.
- **Max Pooling**: Reduces spatial dimensions while retaining important features.
- **Fully Connected Layers**: Connects the extracted features to a dense output layer.
- **Dropout**: Applied to reduce overfitting during training.

## Training and Evaluation

- **Optimizer**: Adam optimizer with learning rate adjustments.
- **Loss Function**: Categorical cross-entropy loss function.
- **Callbacks**: EarlyStopping and ReduceLROnPlateau to prevent overfitting and adjust the learning rate.
- **Metrics**: Training accuracy of 94% and validation accuracy of 90%.

The model was trained on a high-performance GPU to handle the large dataset effectively.

## Grad-CAM Visualization

The project also implements **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the regions of the image that contribute most to the model's decision. This provides insights into how the model makes predictions and enhances interpretability.

### Grad-CAM Insights:
- Helps visualize the important areas of the image that led to classification.
- Provides a better understanding of the model's decision-making process, especially for complex images.

## Results

- **Training Accuracy**: 94%
- **Validation Accuracy**: 90%
- The model demonstrates high classification accuracy and generalizes well to unseen data.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- Seaborn
- NumPy
- Pandas
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/swaps361/land-cover-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd land-cover-classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the EuroSAT dataset and organize the images in the correct directory structure.
2. Run the training script:
    ```bash
    python train_model.py
    ```
3. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```
4. Generate Grad-CAM visualizations for specific images:
    ```bash
    python grad_cam.py --image_path path_to_image
    ```

## Acknowledgments

- [EuroSAT Dataset](https://github.com/phelber/eurosat) for providing the satellite image dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for providing the deep learning framework.
- [Grad-CAM](https://arxiv.org/abs/1610.02391) for the class activation mapping technique.

