# Insurance Cost Prediction

A machine learning project focused on analysing medical insurance data, performing exploratory data analysis, feature engineering, statistical feature selection, and predicting insurance charges using Linear Regression.

## Project Overview

This project uses an insurance dataset to understand the factors associated with medical insurance charges and build a regression model for predicting those charges.

The workflow covers:

* Exploratory Data Analysis
* Data cleaning
* Duplicate removal
* Categorical data encoding
* Feature engineering
* Feature selection
* Standardisation
* Correlation analysis
* Chi-square testing
* Linear Regression
* Model evaluation
* Overfitting and underfitting analysis

## Dataset

The dataset contains information about individuals and their medical insurance costs.

The main features include:

| Feature    | Description                        |
| ---------- | ---------------------------------- |
| `age`      | Age of the individual              |
| `sex`      | Sex of the individual              |
| `bmi`      | Body Mass Index                    |
| `children` | Number of children/dependants      |
| `smoker`   | Whether the individual is a smoker |
| `region`   | Residential region                 |
| `charges`  | Medical insurance charges          |

The target variable is:

```text
charges
```

## Exploratory Data Analysis

The project performs several EDA operations to understand the dataset:

* Dataset shape
* First and last records
* Data types
* Statistical summary
* Missing-value analysis
* Column inspection
* Distribution analysis
* Count plots
* Box plots
* Correlation heatmap

Histograms are used to analyse numerical variables such as:

```text
age
bmi
children
charges
```

Categorical distributions are analysed for:

```text
sex
smoker
children
```

Box plots are also used to identify the distribution and potential outliers in numerical variables.

## Data Cleaning and Preprocessing

A copy of the original dataset is created before preprocessing.

Duplicate records are removed from the dataset.

```python
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)
```

The `sex` and `smoker` columns are converted into binary numerical features.

```text
sex
male   -> 0
female -> 1

smoker
no  -> 0
yes -> 1
```

The columns are subsequently renamed:

```text
sex    -> is_female
smoker -> is_smoker
```

## Categorical Encoding

The `region` feature contains multiple categories, so one-hot encoding is used.

The resulting features include:

```text
region_northeast
region_northwest
region_southeast
region_southwest
```

## Feature Engineering

BMI is converted into categorical groups:

| BMI Range    | Category    |
| ------------ | ----------- |
| Below 18.5   | Underweight |
| 18.5–24.9    | Normal      |
| 25–29.9      | Overweight  |
| 30 and above | Obese       |

These categories are then converted into numerical features using one-hot encoding.

The resulting features include:

```text
bmi_category_Underweight
bmi_category_Normal
bmi_category_Overweight
bmi_category_Obese
```

## Feature Scaling

StandardScaler is used to standardise the following numerical features:

```text
age
bmi
children
```

The transformation is performed using:

```python
from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']

scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
```

## Statistical Analysis

### Pearson Correlation

Pearson correlation is calculated between the selected features and the target variable, `charges`.

The analysis includes:

```text
age
bmi
children
is_female
is_smoker
region_northeast
region_northwest
region_southeast
region_southwest
bmi_category_Underweight
bmi_category_Normal
bmi_category_Overweight
bmi_category_Obese
```

The correlation coefficients are used to understand the linear relationship between the features and insurance charges.

### Chi-Square Test

A chi-square test is performed for categorical features.

The target variable `charges` is divided into four quantile-based groups before performing the test.

The significance level is:

```text
alpha = 0.05
```

The test is used as part of the feature-selection process.

## Final Feature Set

After the feature-selection process, the final dataframe contains:

```text
age
is_female
bmi
children
is_smoker
charges
region_southeast
bmi_category_Obese
```

The target variable is:

```text
charges
```

## Machine Learning Model

The project uses **Linear Regression** to predict insurance charges.

The dataset is divided into training and testing sets using an 80/20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

The model is then trained using:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

## Model Evaluation

The model is evaluated using the following metrics:

### R² Score

The R² score measures how well the model explains the variation in the target variable.

```python
r2 = r2_score(y_test, y_pred)
```

### Adjusted R²

Adjusted R² is also calculated while accounting for the number of predictors in the model.

```python
adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)
```

### Overfitting and Underfitting Analysis

The project compares training and testing R² scores.

```text
Training R²
Testing R²
Generalization Gap
```

The generalisation gap is calculated as:

```python
abs(train_r2 - test_r2)
```

This provides an indication of how closely the model's performance on unseen data compares with its performance on training data.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Jupyter Notebook / Python environment

## Python Libraries

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

## Project Workflow

```text
Dataset
   |
   v
Data Loading
   |
   v
Exploratory Data Analysis
   |
   v
Data Cleaning
   |
   v
Categorical Encoding
   |
   v
Feature Engineering
   |
   v
Feature Scaling
   |
   v
Pearson Correlation
   |
   v
Chi-Square Test
   |
   v
Feature Selection
   |
   v
Train-Test Split
   |
   v
Linear Regression
   |
   v
Prediction
   |
   v
Model Evaluation
   |
   v
Overfitting / Underfitting Analysis
```


## Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Exploratory Data Analysis
* Data preprocessing
* Label encoding
* One-hot encoding
* Feature engineering
* Feature scaling
* Pearson correlation
* Chi-square statistical testing
* Feature selection
* Train-test splitting
* Linear Regression
* R² evaluation
* Adjusted R²
* Model generalisation
* Overfitting analysis


This project is part of my machine learning learning journey, focusing on practical implementation of data analysis, feature engineering, statistics, and machine learning.
