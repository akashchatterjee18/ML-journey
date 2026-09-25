# Football Player Market Value Prediction

A machine learning project that predicts the estimated market value of a football player based on their player attributes and preferred position.

The project includes data preprocessing, feature engineering, statistical feature selection, feature scaling, model training, evaluation, and a Streamlit web application for making predictions.

## Live Demo

The trained model is deployed as an interactive Streamlit application where users can enter player attributes and receive an estimated market value.

Live application: [Football Player Market Value Prediction — Streamlit App](https://footballer-market-value-prediction-by-ac.streamlit.app/

## Project Overview

Football player market value depends on several factors, including age, overall rating, potential, physical attributes, technical abilities, attacking and defending capabilities, goalkeeping attributes, and playing position.

This project uses these attributes to build a regression model capable of estimating a player's market value.

The application allows users to enter player information through an interactive interface and generates a predicted market value in euros.

## Features

The model uses a combination of player profile, physical, technical, attacking, defending, goalkeeping, and positional features.

### Player Profile

* Age
* Overall Rating
* Potential
* Special
* Elite status

### Physical Attributes

* Acceleration
* Sprint Speed
* Agility
* Balance
* Jumping
* Stamina
* Strength

### Ball Control

* Ball Control
* Dribbling
* Composure
* Reactions

### Passing

* Vision
* Short Passing
* Long Passing
* Crossing
* Curve
* Free Kick Accuracy

### Attacking

* Finishing
* Shot Power
* Long Shots
* Positioning
* Volleys
* Heading Accuracy
* Penalties

### Defending

* Aggression
* Interceptions
* Marking
* Standing Tackle
* Sliding Tackle

### Goalkeeping

* GK Diving
* GK Handling
* GK Kicking
* GK Positioning
* GK Reflexes

### Preferred Position

The application supports multiple football positions, including:

* GK
* CB
* LB
* RB
* LWB
* RWB
* CDM
* CM
* CAM
* LM
* RM
* LW
* RW
* LF
* RF
* CF
* ST
* LS
* RS
* LCB
* RCB
* LCM
* RCM
* LDM
* RDM
* LAM
* RAM

The application converts the selected preferred position into binary features before making a prediction.

## Machine Learning Workflow

The project follows the following pipeline:

```text
Raw Dataset
     |
     v
Data Cleaning
     |
     v
Missing Value Handling
     |
     v
Feature Engineering
     |
     v
Position Encoding
     |
     v
Statistical Feature Selection
     |
     v
Feature Scaling
     |
     v
Train-Test Split
     |
     v
Linear Regression
     |
     v
Model Evaluation
     |
     v
Model Serialization
     |
     v
Streamlit Deployment
```

## Data Preprocessing

The dataset contains football player information along with their market value and wage.

### Missing Values

Missing positional attributes are filled with `0`, while missing club information is represented as `Free Agent`.

### Monetary Conversion

Market value and wage values originally containing formats such as `€M` and `€K` are converted into numerical values.

For example:

```text
€10M -> 10000000
€500K -> 500000
```

### Date and Numeric Cleaning

Some attributes contain values represented using formats such as:

```text
11-Oct
```

The project includes preprocessing logic to convert these values into numerical representations before modelling.

## Feature Engineering

An `Elite` feature is created based on the player's overall rating:

```python
df['Elite'] = (df['Overall'] >= 90).astype(int)
```

This creates a binary feature indicating whether a player has an overall rating of 90 or higher.

Preferred positions are also transformed into binary features.

For example:

```text
CAM_preferred_position
ST_preferred_position
LW_preferred_position
CB_preferred_position
```

This allows the regression model to use positional information numerically.

## Feature Selection

Two statistical approaches were used during feature selection.

### Pearson Correlation

Pearson correlation was calculated between numerical features and the target variable, `Value`.

Features with an absolute correlation below `0.10` were removed from the modelling dataset.

### Chi-Square Test

Categorical features were evaluated using a chi-square test.

The target market value was divided into four groups, after which categorical features were tested for their relationship with market value categories.

The significance level used was:

```text
α = 0.05
```

Features that did not meet the statistical significance criterion were removed.

## Feature Scaling

`StandardScaler` from Scikit-learn is used to standardise numerical features before training the model.

```python
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])
```

The fitted scaler is saved and reused by the Streamlit application so that user inputs are transformed consistently with the training data.

## Model

The project uses **Linear Regression** for predicting football player market value.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

The dataset is divided into training and testing sets using an 80/20 split with `random_state=42`.

## Model Performance

The model achieved the following results on the test data:

| Metric             | Result |
| ------------------ | -----: |
| R² Score           | 0.7961 |
| Adjusted R²        | 0.7928 |
| Training R²        | 0.7539 |
| Testing R²         | 0.7961 |
| Generalization Gap | 0.0422 |

These values are from the evaluation output in the training notebook/script.

An R² score of approximately `0.7961` indicates that the model explains a substantial portion of the variation in the test-set market values.

## Streamlit Application

The Streamlit application provides an interactive interface for entering player attributes.

Users can specify:

* Player age
* Overall rating
* Potential
* Physical attributes
* Ball control attributes
* Passing attributes
* Attacking attributes
* Defending attributes
* Goalkeeping attributes
* Preferred position

The application automatically calculates the `Elite` feature from the overall rating and creates the required positional features.

After the input is processed and scaled, the trained model generates the predicted market value.

```text
Predicted Market Value: €...
```

The application loads the trained model, scaler, and expected feature columns from serialized files.

## Project Structure

```text
football-market-value-prediction/
│
├── app.py
├── Footballer MV Prediction.py
├── footballers.csv
│
├── football_market_value_model.pkl
├── scalar.pkl
├── columns.pkl
│
├── requirements.txt
└── README.md
```

### File Description

| File                              | Description                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| `app.py`                          | Streamlit application used for deployment                                                    |
| `Footballer MV Prediction.py`     | Data preprocessing, feature engineering, statistical analysis, model training and evaluation |
| `footballers.csv`                 | Football player dataset                                                                      |
| `football_market_value_model.pkl` | Trained Linear Regression model                                                              |
| `scalar.pkl`                      | Saved StandardScaler used during preprocessing                                               |
| `columns.pkl`                     | Saved list of expected model input columns                                                   |
| `requirements.txt`                | Python dependencies                                                                          |
| `README.md`                       | Project documentation                                                                        |

## Model Deployment

The trained model and preprocessing objects are saved using Joblib:

```python
joblib.dump(model, "football_market_value_model.pkl")
joblib.dump(scaler, "scalar.pkl")
joblib.dump(expected_columns, "columns.pkl")
```

The Streamlit application loads these files when it starts.

## Requirements

The project uses:

```text
streamlit
pandas
numpy
joblib
scikit-learn==1.6.1
```

These dependencies are specified in the project's `requirements.txt`.

## Prediction Process

When a user clicks **Predict Market Value**, the application:

1. Collects the player's attributes.
2. Calculates the `Elite` feature.
3. Creates preferred-position features.
4. Matches the input columns with the training columns.
5. Applies the saved StandardScaler.
6. Sends the processed data to the trained Linear Regression model.
7. Displays the predicted market value in euros.

The implementation follows this preprocessing and prediction flow directly in the Streamlit application.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Joblib
* Streamlit
* GitHub

## Key Machine Learning Concepts

This project demonstrates practical implementation of:

* Exploratory Data Analysis
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* One-Hot Encoding
* Pearson Correlation
* Chi-Square Testing
* Feature Selection
* Standardisation
* Train-Test Splitting
* Linear Regression
* R² Score
* Adjusted R²
* Generalisation Gap
* Model Serialization
* Streamlit Deployment

## Limitations

The predicted value should be treated as a model-based estimate rather than an official football transfer valuation.

Actual player market values can depend on factors that are not represented in the model, including:

* Contract duration
* Transfer demand
* Club finances
* League
* Recent performance
* Injury history
* Transfer market conditions
* Reputation
* Age-related market trends
* Negotiation factors

The model's predictions are therefore dependent on the features and data used during training.


## Project Purpose

This project was developed as part of a practical machine learning journey to explore the complete workflow of a regression problem, from raw football player data to a deployed prediction application.

---
