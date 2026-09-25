# Heart Disease Prediction

A machine learning classification project that predicts the likelihood of heart disease based on patient-related health features.

The project covers exploratory data analysis, data cleaning, preprocessing, feature encoding, feature scaling, training and comparison of multiple classification algorithms, model selection, model serialization, and deployment using Streamlit.

## Overview

The Heart Disease Prediction project uses a machine learning classification approach to predict whether a patient is likely to have heart disease.

The complete workflow includes:

* Exploratory Data Analysis
* Data cleaning and preprocessing
* Handling invalid values
* Categorical feature encoding
* Train-test splitting
* Feature scaling
* Training multiple classification models
* Model evaluation
* Model comparison and selection
* Saving the trained model
* Building a Streamlit application
* Deploying the application using Streamlit Community Cloud

## Dataset

The project uses the Heart Disease dataset containing **918 records and 12 columns**.

The dataset contains the following features:

| Feature          | Description                      |
| ---------------- | -------------------------------- |
| `Age`            | Age of the patient               |
| `Sex`            | Sex of the patient               |
| `ChestPainType`  | Type of chest pain               |
| `RestingBP`      | Resting blood pressure           |
| `Cholesterol`    | Cholesterol level                |
| `FastingBS`      | Fasting blood sugar              |
| `RestingECG`     | Resting electrocardiogram result |
| `MaxHR`          | Maximum heart rate achieved      |
| `ExerciseAngina` | Exercise-induced angina          |
| `Oldpeak`        | ST depression                    |
| `ST_Slope`       | Slope of the ST segment          |
| `HeartDisease`   | Target variable                  |

The target variable is:

```text
HeartDisease
```

where:

```text
0 = No Heart Disease
1 = Heart Disease
```

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and characteristics of the dataset.

The analysis included:

* Inspecting dataset shape
* Checking column names
* Checking data types
* Generating descriptive statistics
* Checking duplicate records
* Checking missing values
* Analysing target distribution
* Analysing numerical feature distributions
* Studying categorical features
* Analysing relationships between features and the target
* Creating correlation heatmaps

Visualisations were created using Matplotlib and Seaborn.

## Data Preprocessing

### Handling Invalid Cholesterol Values

The dataset contained cholesterol values of `0`, which are not meaningful for this feature.

The mean cholesterol value calculated from the non-zero observations was used to replace the zero values.

```python
ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()

df['Cholesterol'] = df['Cholesterol'].replace(
    0,
    ch_mean
)

df['Cholesterol'] = df['Cholesterol'].round(2)
```

### Handling Invalid Resting Blood Pressure Values

Similarly, zero values in `RestingBP` were replaced using the mean resting blood pressure calculated from the non-zero observations.

```python
resting_bp_mean = df.loc[
    df['RestingBP'] != 0,
    'RestingBP'
].mean()

df['RestingBP'] = df['RestingBP'].replace(
    0,
    resting_bp_mean
)

df['RestingBP'] = df['RestingBP'].round(2)
```

### Categorical Feature Encoding

Categorical features were converted into numerical features using one-hot encoding.

```python
df_encode = pd.get_dummies(df)
df_encode = df_encode.astype(int)
```

This converted categorical variables such as:

```text
Sex
ChestPainType
RestingECG
ExerciseAngina
ST_Slope
```

into numerical features suitable for machine learning models.

## Feature and Target Separation

The dataset was divided into independent features and the target variable.

```python
X = df_encode.drop('HeartDisease', axis=1)
Y = df_encode['HeartDisease']
```

where:

* `X` contains the input features.
* `Y` contains the target variable.

## Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

The `random_state=42` parameter was used to ensure reproducibility.

## Feature Scaling

Feature scaling was performed using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then used to transform both training and testing data.

The fitted scaler was later saved and used by the Streamlit application so that new user inputs receive the same preprocessing as the training data.

## Machine Learning Models

Five classification algorithms were implemented and compared:

### 1. Logistic Regression

Logistic Regression was used as one of the classification models for predicting the presence or absence of heart disease.

### 2. K-Nearest Neighbours

K-Nearest Neighbours (KNN) was implemented to classify observations based on nearby observations in the feature space.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes was implemented as a probabilistic classification algorithm.

### 4. Decision Tree

A Decision Tree classifier was implemented to classify patients using feature-based decision rules.

### 5. Support Vector Machine

A Support Vector Classifier (`SVC`) was implemented to find a decision boundary between the two classes.

## Model Selection Process

The model was not selected arbitrarily.

All five classification models were trained using the same scaled training data and evaluated on the same test data.

The process was:

```text
Preprocessed Dataset
        |
        v
Train-Test Split
        |
        v
StandardScaler
        |
        v
Train 5 Classification Models
        |
        v
Generate Predictions
        |
        v
Calculate Accuracy and F1 Score
        |
        v
Compare Model Performance
        |
        v
Select Best Performing Model
        |
        v
Save Selected Model
```

The models were evaluated using:

* Accuracy Score
* F1 Score
* Classification Report
* Confusion Matrix

## Model Comparison

The results obtained from the notebook were:

| Model                   |   Accuracy |   F1 Score |
| ----------------------- | ---------: | ---------: |
| **Logistic Regression** | **86.41%** | **87.92%** |
| KNN                     |     84.78% |     86.41% |
| Naive Bayes             |     84.24% |     85.57% |
| Decision Tree           |     80.43% |     81.63% |
| SVM                     |     85.33% |     87.08% |

### Final Model Selection

Logistic Regression achieved the highest performance among the tested models:

```text
Accuracy : 86.41%
F1 Score : 87.92%
```

Therefore, **Logistic Regression was selected as the final model** for the Heart Disease Prediction application.

The selected model was then saved using Joblib for use in the Streamlit application.

## Model Evaluation

### Accuracy Score

Accuracy was used to measure the proportion of correctly classified observations.

```python
accuracy_score(Y_test, Y_pred)
```

### F1 Score

F1 score was used to evaluate the balance between precision and recall.

```python
f1_score(Y_test, Y_pred)
```

### Confusion Matrix

A confusion matrix was used to analyse correct and incorrect classifications.

It provides information about:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

## Saving the Model

After selecting Logistic Regression as the final model, the trained model and preprocessing objects were saved using Joblib.

```python
import joblib

joblib.dump(
    models['Logistic Regression'],
    "Logistic Regression_heart.pkl"
)

joblib.dump(
    scaler,
    "scalar.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "columns.pkl"
)
```

The following files are used by the application:

```text
Logistic Regression_heart.pkl
scalar.pkl
columns.pkl
```

### `Logistic Regression_heart.pkl`

Contains the trained Logistic Regression model.

### `scalar.pkl`

Contains the fitted `StandardScaler`.

### `columns.pkl`

Contains the feature column order expected by the trained model.

These files allow the deployed application to use the same model and preprocessing pipeline that were used during training.

## Streamlit Application

A Streamlit web application was created to provide an interactive interface for the trained machine learning model.

The application allows users to enter:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

The application uses Streamlit input components such as sliders, select boxes, and number inputs.

## Prediction Workflow

When the user clicks the **Predict** button, the application performs the following steps:

```text
User Input
    |
    v
Create Input DataFrame
    |
    v
Add Missing Expected Columns
    |
    v
Arrange Columns in Expected Order
    |
    v
Apply Saved StandardScaler
    |
    v
Load Logistic Regression Model
    |
    v
Generate Prediction Probabilities
    |
    v
Display Heart Disease Probability
```

The application loads the trained model, scaler, and expected feature columns using Joblib.

```python
model = joblib.load('Logistic Regression_heart.pkl')
scaler = joblib.load('scalar.pkl')
expected_columns = joblib.load('columns.pkl')
```

The input data is then arranged according to the expected feature columns before scaling.

```python
input_df = input_df[expected_columns]

input_scaled = scaler.transform(input_df)
```

The model generates prediction probabilities using:

```python
probabilities = model.predict_proba(input_scaled)[0]
```

The probability for each class is then converted into a percentage.

## Prediction Output

The application displays two probabilities:

```text
Heart Disease Probability
No Heart Disease Probability
```

The prediction threshold used by the application is:

```python
if disease_probability >= 0.5:
```

If the disease probability is at least 50%, the application displays the corresponding heart disease probability and no-heart-disease probability.

Otherwise, it displays the no-heart-disease probability first.

## Streamlit Deployment

The application was deployed using **Streamlit Community Cloud**.

### Live Application

```text
https://heart-disease-prediction-by-ac.streamlit.app/
```

The deployed application provides a web-based interface where users can enter patient information and obtain predictions from the trained Logistic Regression model.

## Deployment Files

The Streamlit deployment requires the application code, trained model, preprocessing objects, and dependencies.

```text
Heart-Disease-Prediction/
│
├── app.py
├── requirements.txt
├── Logistic Regression_heart.pkl
├── scalar.pkl
├── columns.pkl
├── heart.csv
└── README.md
```

## `app.py`

The `app.py` file contains the Streamlit frontend and prediction logic.

It:

* Creates the user interface
* Collects user inputs
* Creates the input DataFrame
* Matches the expected feature columns
* Loads the saved scaler
* Scales the input
* Loads the trained Logistic Regression model
* Generates prediction probabilities
* Displays the prediction results

The application uses the saved model and preprocessing files rather than retraining the model when the application starts.

## `requirements.txt`

The deployment dependencies are specified in `requirements.txt`.

```text
streamlit
pandas
numpy
joblib
scikit-learn==1.6.1
```

The scikit-learn version is explicitly specified to maintain compatibility with the saved model and preprocessing objects.

## Deployment Workflow

```text
Heart Disease Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Data Cleaning
        |
        v
Handle Invalid Values
        |
        v
One-Hot Encoding
        |
        v
Train-Test Split
        |
        v
Feature Scaling
        |
        v
Train Multiple Models
        |
        v
Evaluate Models
        |
        v
Compare Accuracy and F1 Score
        |
        v
Select Logistic Regression
        |
        v
Save Model, Scaler and Columns
        |
        v
Build Streamlit Application
        |
        v
Create requirements.txt
        |
        v
Push Files to GitHub
        |
        v
Connect Repository to Streamlit
        |
        v
Deploy on Streamlit Community Cloud
        |
        v
Live Prediction Application
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Google Colab
* GitHub
* Streamlit Community Cloud

## Libraries Used

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
```

## Results

The final model comparison from the project was:

| Model                   |   Accuracy |   F1 Score |
| ----------------------- | ---------: | ---------: |
| **Logistic Regression** | **86.41%** | **87.92%** |
| KNN                     |     84.78% |     86.41% |
| Naive Bayes             |     84.24% |     85.57% |
| Decision Tree           |     80.43% |     81.63% |
| SVM                     |     85.33% |     87.08% |

Logistic Regression achieved the highest accuracy and F1 score among the models tested and was therefore selected as the final model.

## Conclusion

This project demonstrates an end-to-end machine learning classification workflow for heart disease prediction.

The project covers:

* Exploratory Data Analysis
* Data cleaning
* Handling invalid values
* Categorical feature encoding
* Feature scaling
* Train-test splitting
* Training multiple classification models
* Model evaluation
* Model comparison
* Model selection
* Model serialization
* Streamlit application development
* GitHub integration
* Cloud deployment

Five classification models were evaluated using the same train-test split and scaled feature data.

**Logistic Regression achieved the highest accuracy of 86.41% and F1 score of 87.92%, and was selected as the final model.**

The trained model, scaler, and expected feature columns were saved using Joblib and integrated into a Streamlit application.

The application was then deployed using Streamlit Community Cloud.

## Live Demo

**Heart Disease Prediction Application:**

https://heart-disease-prediction-by-ac.streamlit.app/

## Disclaimer

This project is intended for educational and machine learning demonstration purposes only.

The predictions generated by the application should not be considered medical advice, diagnosis, or a substitute for consultation with a qualified healthcare professional.
