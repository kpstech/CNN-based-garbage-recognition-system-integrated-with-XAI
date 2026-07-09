Garbage Classification System using Deep Learning
Project Overview

This project focuses on developing an image classification system for garbage waste categories using deep learning techniques. The model automatically classifies waste images into different categories, helping in smart waste management and recycling automation.

The system uses a Convolutional Neural Network (CNN) to analyze garbage images and predict the correct waste class.

Objectives

Automatically classify garbage images into predefined categories.

Reduce manual waste sorting effort.

Support smart waste management systems.

Improve recycling efficiency.

Dataset

The dataset used in this project is sourced from Kaggle Garbage Classification Dataset.

Dataset Characteristics

Total Images: ~2500+

Classes: 6 Waste Categories

Waste Categories

Cardboard

Glass

Metal

Paper

Plastic

Trash

Dataset Link

https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

Project Architecture
Workflow

Dataset Collection
Data Preprocessing
Image Augmentation
Model Training
Model Evaluation
Prediction & Testing

Input Image
     ↓
Preprocessing
     ↓
CNN Model
     ↓
Feature Extraction
     ↓
Classification Layer
     ↓
Predicted Garbage Class
Technologies Used
Technology	Purpose
Python	Programming Language
TensorFlow / Keras	Deep Learning Model
OpenCV	Image Processing
NumPy	Numerical Computations
Matplotlib	Visualization
Scikit-Learn	Model Evaluation
Streamlit (optional)	Web App Interface
Model Used

The project uses a Convolutional Neural Network (CNN) for image classification.

Model Features

Multiple convolution layers

ReLU activation

Max pooling layers

Fully connected layers

Softmax output layer

Model Performance

Example performance metrics:

Metric	Value
Accuracy	~90%
Loss	Low
Precision	High
Recall	High

(Replace with your actual results)

Installation
Step 1: Clone the Repository
git clone https://github.com/yourusername/garbage-classification.git
Step 2: Navigate to Project Folder
cd garbage-classification
Step 3: Install Dependencies
pip install -r requirements.txt
Run the Project
Train the Model
python train_model.py
Run Prediction
python predict.py
Run Web Application (Optional)
streamlit run app.py
Example Prediction

Input Image → Model → Output Class

Example:

Input: plastic_bottle.jpg
Output: Plastic
Future Improvements

Improve model accuracy with larger datasets.

Deploy model using Flask / Streamlit Web App.

Integrate with IoT smart garbage bins.

Real-time waste detection using cameras.

Author

Krishna Prasad
M.Tech first Year Student

License

This project is open-source and available under the MIT License.
