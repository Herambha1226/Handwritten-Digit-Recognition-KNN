# Handwritten Digit Recognition using KNN

This project implements a **Handwritten Digit Recognition system** using the **K-Nearest Neighbors (KNN)** machine learning algorithm and the **MNIST dataset**. The system can recognize handwritten digits in **real time using a webcam** by preprocessing the captured image and predicting the digit using a trained machine learning model.

Handwritten digit recognition is one of the most popular beginner machine learning problems and is widely used for learning computer vision and classification algorithms. The MNIST dataset contains thousands of handwritten digit images used for training and testing digit recognition systems. :contentReference[oaicite:1]{index=1}

---

# Project Features

- Train a **KNN classification model**
- Use the **MNIST handwritten digits dataset**
- Visualize digit images using **Matplotlib**
- Perform **image preprocessing using OpenCV**
- Real-time digit recognition using **webcam**
- Display predicted digit on screen
- Voice output for predicted digit

---

# Technologies Used

- Python
- NumPy
- Scikit-learn
- OpenCV
- Matplotlib
- pyttsx3 (Text to Speech)

---

# Dataset

This project uses the **MNIST dataset**.

Dataset Details:

- Total Images: 70,000
- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Features per image: 784

Each digit image is converted into a **784-dimensional feature vector** by flattening the 28×28 pixel image.

---

# Machine Learning Algorithm

This project uses the **K-Nearest Neighbors (KNN)** classification algorithm.

KNN works by:

1. Calculating distance between input sample and training samples
2. Selecting the **K nearest neighbors**
3. Taking the **majority class label**
4. Predicting the digit based on nearest neighbors

Distance metric used:

- **Euclidean Distance**

---

# Project Workflow

1. Load MNIST dataset
2. Split dataset into training and testing data
3. Train KNN classification model
4. Evaluate prediction accuracy
5. Capture frames using webcam
6. Extract **Region of Interest (ROI)**
7. Convert image to grayscale
8. Apply preprocessing (blur + threshold)
9. Resize image to **28×28**
10. Flatten image to **784 features**
11. Predict digit using trained KNN model
12. Display prediction and voice output

---

# Installation

Clone the repository
```
git clone https://github.com/Herambha1226/Handwritten-Digit-Recognition-KNN.git
```


---

# Train the Model

Before running the real-time digit recognition system, you must train the KNN model.

Run the following command:
```
python dataset/load-dataset.py
```

This will:

- Load MNIST dataset
- Train KNN classifier
- Evaluate model accuracy
- Save trained model

---

# Run Real-Time Digit Recognition

After training the model, start the webcam digit recognition system:

use command:
```
python src/realtime_camera.py
```

---
# Developed By

Herambha Karthikeya Guptha

---

