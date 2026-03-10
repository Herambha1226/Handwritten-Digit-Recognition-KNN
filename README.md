# Handwritten Digit Recognition using KNN

This project implements a handwritten digit recognition system using the K-Nearest Neighbors (KNN) algorithm and the MNIST dataset. The system can recognize digits in real time using a webcam.

## Features

- Train a KNN classifier on the MNIST dataset
- Visualize handwritten digits
- Real-time digit detection using OpenCV
- Image preprocessing pipeline
- Prediction of digits from live camera input
- Voice output for predicted digits

## Technologies Used

- Python
- NumPy
- Scikit-learn
- OpenCV
- Matplotlib
- pyttsx3

## Dataset

The project uses the MNIST dataset, which contains 70,000 handwritten digit images (28x28 pixels).

## Project Workflow

1. Load MNIST dataset
2. Train KNN model
3. Evaluate accuracy
4. Capture camera frames
5. Extract Region of Interest (ROI)
6. Preprocess image
7. Resize to 28x28
8. Predict digit using KNN
9. Display prediction and speak digit

## Installation

Clone the repository:

