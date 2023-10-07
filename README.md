# Breast Cancer Detection

## Overview
This project aims to detect breast cancer using machine learning algorithms. It involves importing, preprocessing, and analyzing breast cancer data to train and evaluate various classification models. The dataset used in this project contains information about breast cancer biopsies.

## Table of Contents
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)

## Dataset
The breast cancer dataset used in this project can be found in the `data.csv` file. It contains various features such as radius, texture, smoothness, and more, which are used to predict the diagnosis (Malignant or Benign) of breast cancer cases.

## Getting Started
To get started with this project, follow the steps below:

### Prerequisites
- Python 3.x
- Jupyter Notebook (optional for running the code interactively)

### Installation
1. Clone the repository to your local machine:
``` 
git clone https://github.com/Anik-et/breast-cancer-detection-project.git
```

2. Navigate to the project directory:
```
cd breast-cancer-detection
```

3. Install the required Python libraries:
```
!pip install numpy
!pip install pandas
!pip install matplotlib
```
 
## Usage
1. Upload the `data.csv` file containing the breast cancer dataset to the project directory.
2. Open and run the Jupyter Notebook or Python script (`breast_cancer_detection.ipynb` or `breast_cancer_detection.py`) to perform the following tasks:
- Load and preprocess the dataset.
- Map class string values to numbers.
- Split the dataset into training and testing sets.
- Scale the features for better model performance.
- Evaluate and compare the performance of different machine learning models.
- Train a selected model with the highest accuracy.
- Make predictions on the test data.

## Models
This project evaluates the following machine learning models:
- Logistic Regression
- Linear Discriminant Analysis
- K-Nearest Neighbors
- Decision Tree Classifier
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

## Results
The project provides accuracy scores for each model based on a 10-fold cross-validation. The model with the highest accuracy is selected for training and prediction.

## Contributing
Contributions to this project are welcome. Feel free to fork the repository, make improvements, and submit pull requests.





