# MNIST Digit Recognition with a Convolutional Neural Network (CNN)

This project demonstrates how to build, train, evaluate, and deploy a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The notebook covers essential steps in a typical machine learning workflow, from data loading and preprocessing to model evaluation and making predictions on custom images.

## Project Overview

- **Data Loading & Visualization**: Loads the MNIST dataset and visualizes sample images.
- **Data Preprocessing**: Normalizes pixel values and reshapes images for CNN input.
- **CNN Model Architecture**: Defines a simple yet effective CNN model using TensorFlow/Keras, including `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
- **Model Training**: Trains the CNN model on the preprocessed MNIST training data.
- **Model Evaluation**: Assesses the model's performance using test data and visualizes training history (accuracy and loss curves).
- **Prediction & Custom Image Testing**: Demonstrates how to use the trained model to predict digits from new, custom uploaded images, including robust preprocessing for various image formats.
- **Confusion Matrix**: Visualizes model performance on the test set using a confusion matrix to identify common misclassifications.
- **Model Saving & Loading**: Shows how to save the trained model for later use and load it back into memory.

## Getting Started

To run this notebook and replicate the results:

1.  **Open in Google Colab**: Click the "Open in Colab" badge (if available) or upload the `.ipynb` file to Google Colab.
2.  **Run Cells**: Execute all cells in sequential order.
3.  **Upload Custom Images**: When prompted, you can upload your own `.png` images of handwritten digits (e.g., `digit0.png`, `digit1.png`) to test the model's predictions.

## Dependencies

This project primarily uses:

- `tensorflow`
- `numpy`
- `matplotlib`
- `Pillow` (PIL)
- `scikit-learn` (for confusion matrix)
- `seaborn` (for confusion matrix visualization)

All necessary libraries are typically pre-installed or easily installable in a Google Colab environment.

## How to Use Custom Digit Images

1.  **Prepare your image**: Ensure your image is a grayscale `.png` file containing a single handwritten digit. White digits on a black background (like MNIST) are preferred for optimal results with the current preprocessing.
2.  **Upload**: When you encounter the `files.upload()` cell, select your digit image file.
3.  **Run Prediction Cells**: Execute the subsequent cells that process the uploaded image and generate a prediction.

Enjoy experimenting with handwritten digit recognition!
