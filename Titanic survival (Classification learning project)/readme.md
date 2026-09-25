# Titanic Survival Prediction

A machine learning classification project that predicts whether a passenger survived the Titanic disaster based on passenger-related features.

The project focuses on data exploration, data preprocessing, categorical feature encoding, train-test splitting, feature scaling, implementation of multiple classification algorithms, model evaluation, and model selection.

## Overview

The Titanic dataset is used to build and evaluate multiple supervised machine learning classification models.

The following classification algorithms are implemented and compared:

* Logistic Regression
* K-Nearest Neighbours (KNN)
* Gaussian Naive Bayes
* Decision Tree
* Support Vector Classifier (SVC)

The models are evaluated using accuracy score, confusion matrix, and classification report. Based on the results obtained in the project, SVC achieved the highest accuracy and was selected as the final model.

## Dataset

The project uses the Titanic dataset.

The dataset contains information about passengers, including:

* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Port of embarkation

The target variable is:

```text
survived
```

where:

* `0` represents a passenger who did not survive
* `1` represents a passenger who survived

## Data Exploration

Initial data exploration was performed to understand:

* Dataset shape
* Column names
* Data types
* Missing values
* Numerical and categorical features
* Feature distributions
* Relationships between variables
* Survival patterns

Visualisation was performed using Matplotlib and Seaborn.

## Data Preprocessing

Several preprocessing steps were performed before training the machine learning models.

### Removing Unnecessary Columns

The following columns were removed because they were not required for the prediction task:

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

Rows containing missing values in the `embarked` column were removed.

### Encoding Categorical Features

Categorical features were converted into numerical values using `LabelEncoder`.

The following categorical columns were encoded:

```text
sex
embarked
```

### Feature and Target Separation

The dataset was divided into independent features and the target variable:

```python
X = df.drop("survived", axis=1)
Y = df["survived"]
```

where:

* `X` contains the input features.
* `Y` contains the target variable `survived`.

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

The `random_state=42` parameter was used to ensure reproducibility.

## Feature Scaling

Feature scaling was performed using `StandardScaler`.

Scaling was particularly important for models that are sensitive to the scale of input features, such as KNN and SVC.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

The scaled feature data was then used for the relevant models.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression was implemented as a baseline classification algorithm.

It is a commonly used classification model for predicting binary outcomes.

### 2. K-Nearest Neighbours

K-Nearest Neighbours (KNN) was implemented using:

```python
n_neighbors=5
```

Feature scaling was performed before training the KNN model.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes was implemented as a probabilistic classification algorithm.

The model was trained using the processed Titanic dataset.

### 4. Decision Tree

A Decision Tree classifier was implemented using:

```python
random_state=42
```

The model uses decision rules based on the input features to classify passengers.

### 5. Support Vector Classifier

A Support Vector Classifier (`SVC`) was implemented using scaled feature data.

SVC attempts to find an optimal decision boundary for separating the different classes.

## Model Selection

All five classification models were trained and evaluated using the same train-test split.

The accuracy results obtained in the project were:

| Model                           |   Accuracy |
| ------------------------------- | ---------: |
| Logistic Regression             |     80.34% |
| K-Nearest Neighbours            |     77.53% |
| Gaussian Naive Bayes            |     77.53% |
| Decision Tree                   |     76.97% |
| Support Vector Classifier (SVC) | **82.58%** |

Based on the evaluation results, **SVC achieved the highest accuracy of 82.58%** among the tested models.

Therefore, **SVC was selected as the final model** for the Titanic Survival Prediction project.

## Model Evaluation

The trained models were evaluated using multiple classification metrics.

### Accuracy Score

Accuracy measures the proportion of correctly classified passengers out of all predictions.

```python
accuracy_score(y_test, y_pred)
```

### Confusion Matrix

A confusion matrix was used to examine the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

```python
confusion_matrix(y_test, y_pred)
```

### Classification Report

A classification report was generated for each model.

It provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(y_test, y_pred)
```

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
Separate Features and Target
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
Generate Predictions
       |
       v
Evaluate Models
       |
       v
Compare Model Performance
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


### The classification models achieved the following accuracy scores:

| Model                |   Accuracy |
| -------------------- | ---------: |
| Logistic Regression  |     80.34% |
| KNN                  |     77.53% |
| Gaussian Naive Bayes |     77.53% |
| Decision Tree        |     76.97% |
| **SVC**              | **82.58%** |

SVC produced the highest accuracy among the models tested in this project.

## Conclusion

This project demonstrates a complete machine learning classification workflow using the Titanic dataset.

The project covers:

* Data exploration
* Data cleaning
* Missing value handling
* Categorical feature encoding
* Feature scaling
* Train-test splitting
* Multiple classification algorithms
* Model evaluation
* Model comparison
* Final model selection

Five classification algorithms were evaluated, with **Support Vector Classifier (SVC)** achieving the highest accuracy of **82.58%**. Therefore, SVC was selected as the final model for the project.

This project is part of my Machine Learning journey and focuses on understanding classification algorithms, preprocessing techniques, model evaluation, and practical machine learning workflows.
