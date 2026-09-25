# Titanic Survival Prediction

A machine learning classification project that predicts whether a passenger survived the Titanic disaster based on passenger-related features.

The project focuses on data preprocessing, feature encoding, train-test splitting, feature scaling, and comparison of multiple classification algorithms.

## Overview

The Titanic dataset is used to build and evaluate several machine learning classification models.

The following models are implemented and compared:

* Logistic Regression
* K-Nearest Neighbours (KNN)
* Gaussian Naive Bayes
* Decision Tree
* Support Vector Classifier (SVC)

After evaluating the models using accuracy, confusion matrices, and classification reports, the SVC model was selected for this dataset based on the results obtained in the project.

## Dataset

The project uses the Titanic dataset.

The dataset contains information about passengers, including features such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Port of embarkation

The target variable is:

* `survived` — whether the passenger survived

## Data Preprocessing

The following preprocessing steps were performed:

### Removing Unnecessary Columns

The following columns were removed:

```text
deck
embark_town
alive
class
who
adult_male
```

### Handling Missing Values

Missing values in the `age` column were replaced with the mean age.

Rows with missing values in the `embarked` column were removed.

### Encoding Categorical Features

The categorical columns `sex` and `embarked` were converted into numerical values using `LabelEncoder`.

### Feature and Target Separation

The dataset was divided into:

```text
X = Features
Y = Target (survived)
```

## Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

```python
train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

## Machine Learning Models

### 1. Logistic Regression

A Logistic Regression model was trained as one of the baseline classification models.

### 2. K-Nearest Neighbours

A KNN classifier was implemented with:

```python
n_neighbors=5
```

Feature scaling was performed using `StandardScaler` before training the KNN model.

### 3. Gaussian Naive Bayes

A Gaussian Naive Bayes classifier was trained on the processed dataset.

### 4. Decision Tree

A Decision Tree classifier was implemented using:

```python
random_state=42
```

### 5. Support Vector Classifier

A Support Vector Classifier (`SVC`) was trained using the scaled feature data.

According to the results obtained in the notebook, SVC performed the best among the models tested and was selected as the final model for this dataset.

## Model Evaluation

The models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

## Project Workflow

```text
Titanic Dataset
       |
       v
Data Exploration
       |
       v
Remove Unnecessary Features
       |
       v
Handle Missing Values
       |
       v
Encode Categorical Features
       |
       v
Train-Test Split
       |
       v
Feature Scaling
       |
       v
Train Multiple Classification Models
       |
       v
Model Evaluation
       |
       v
Compare Results
       |
       v
Select SVC
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

## Project Structure

```text
Titanic-Survival-Classification/
│
├── Titanic survival (Classification learning project).ipynb
├── titanic.csv
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Titanic-Survival-Classification
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Open the Notebook

Open the following file using Jupyter Notebook or JupyterLab:

```text
Titanic survival (Classification learning project).ipynb
```

### 4. Run the Notebook

Run the cells sequentially to:

1. Load the dataset
2. Preprocess the data
3. Train the classification models
4. Generate predictions
5. Evaluate the models
6. Compare their performance

## Conclusion

This project demonstrates the implementation and comparison of several supervised machine learning classification algorithms on the Titanic dataset.

The models were evaluated using accuracy, confusion matrices, and classification reports. Based on the results obtained in the notebook, **Support Vector Classifier (SVC)** was selected as the model used for the final prediction.

This project is part of my Machine Learning journey, focused on understanding classification algorithms, preprocessing techniques, model evaluation, and practical machine learning workflows.
