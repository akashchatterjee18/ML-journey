# Ford Car Price Prediction

A machine learning regression project that predicts the price of Ford cars based on vehicle-related features.

The project focuses on exploratory data analysis, categorical feature encoding, feature scaling, train-test splitting, Linear Regression, model comparison, R² evaluation, adjusted R², and overfitting/underfitting analysis.

## Overview

The Ford car price dataset contains information about different Ford vehicles along with their prices.

The project explores the relationship between vehicle characteristics and car prices and builds a Linear Regression model to predict the price of a vehicle.

Two different preprocessing approaches were implemented and compared:

* One-Hot Encoding
* Label Encoding

Both approaches were used to train separate Linear Regression models, and their performance was compared using regression evaluation metrics.

## Dataset

The project uses a Ford car dataset containing information about vehicles and their prices.

The dataset includes the following features:

```text
model
year
transmission
mileage
fuelType
tax
mpg
engineSize
price
```

The target variable is:

```text
price
```

The target represents the price of the Ford vehicle.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure, distribution, and relationships within the dataset.

The analysis included:

* Inspecting the first few rows
* Checking column names
* Checking dataset shape
* Checking data types
* Generating descriptive statistics
* Checking duplicate records
* Analysing the distribution of car prices
* Analysing numerical feature relationships
* Studying categorical features
* Examining relationships between individual features and price
* Creating correlation visualisations

## Data Visualisation

Several visualisations were created using Matplotlib and Seaborn.

### Price Distribution

A histogram with KDE was used to visualise the distribution of car prices.

```python
sns.histplot(df['price'], kde=True)
plt.show()
```

### Correlation Heatmap

A correlation heatmap was created to analyse relationships between numerical variables.

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
plt.show()
```

### Feature vs Price Analysis

Box plots and scatter plots were used to analyse the relationship between price and different vehicle features.

The project visualised:

* Year vs Price
* Mileage vs Price
* Engine Size vs Price
* Transmission vs Price
* Fuel Type vs Price
* Model vs Price
* Tax vs Price
* MPG vs Price

## Feature and Target Separation

The dataset was divided into independent features and the target variable.

```python
x = df.drop('price', axis=1)
y = df['price']
```

where:

* `x` contains the vehicle features.
* `y` contains the vehicle price.

## Categorical Feature Encoding

The categorical features in the dataset were:

```text
model
transmission
fuelType
```

Two different encoding techniques were implemented to compare their effect on the regression model.

## One-Hot Encoding

One-Hot Encoding was applied to the categorical features using Pandas.

```python
x_1hot = pd.get_dummies(
    x,
    columns=['model', 'transmission', 'fuelType'],
    dtype=int
)
```

This converts the categorical variables into separate binary columns.

For example, different transmission types or fuel types are represented as individual numerical columns.

## Label Encoding

A second version of the dataset was created using Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()

columns = [
    'model',
    'transmission',
    'fuelType'
]

x_label = x

for i in columns:
    x_label[i] = encode.fit_transform(x_label[i])
```

Each category is assigned a numerical label.

This approach was implemented separately so that its performance could be compared with the One-Hot Encoding approach.

## Feature Scaling

Standardisation was applied using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

### One-Hot Encoded Dataset

The numerical columns:

```text
year
mileage
tax
mpg
engineSize
```

were standardised.

```python
numerical_colms = [
    'year',
    'mileage',
    'tax',
    'mpg',
    'engineSize'
]

x_1hot[numerical_colms] = scaler.fit_transform(
    x_1hot[numerical_colms]
)
```

### Label Encoded Dataset

The numerical and encoded features were also standardised before training the Label Encoded model.

```python
x_label[
    [
        'model',
        'year',
        'transmission',
        'mileage',
        'fuelType',
        'tax',
        'mpg',
        'engineSize'
    ]
] = scaler.fit_transform(
    x_label[
        [
            'model',
            'year',
            'transmission',
            'mileage',
            'fuelType',
            'tax',
            'mpg',
            'engineSize'
        ]
    ]
)
```

## Machine Learning Model

Linear Regression was used as the regression algorithm.

The project trained two Linear Regression models using different feature encoding approaches.

### Model 1 — One-Hot Encoding

The One-Hot Encoded dataset was divided into training and testing sets using an 80:20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_1hot,
    y,
    test_size=0.2,
    random_state=42
)
```

A Linear Regression model was then trained:

```python
model = LinearRegression()

model.fit(
    x_train,
    y_train
)
```

Predictions were generated on the test data:

```python
y_pred = model.predict(x_test)
```

## Model 2 — Label Encoding

The Label Encoded dataset was also divided into training and testing sets using the same 80:20 split and `random_state=42`.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_label,
    y,
    test_size=0.2,
    random_state=42
)
```

A second Linear Regression model was trained:

```python
model = LinearRegression()

model.fit(
    x_train,
    y_train
)
```

Predictions were then generated using the test data.

## Model Selection Process

The project did not simply choose an encoding technique without comparison.

Two different preprocessing approaches were tested using Linear Regression:

```text
                    Ford Dataset
                         |
                         v
                Feature Preparation
                         |
             +-----------+-----------+
             |                       |
             v                       v
      One-Hot Encoding        Label Encoding
             |                       |
             v                       v
      Feature Scaling          Feature Scaling
             |                       |
             v                       v
      Linear Regression       Linear Regression
             |                       |
             v                       v
       Model Evaluation        Model Evaluation
             |                       |
             +-----------+-----------+
                         |
                         v
                  Compare Results
                         |
                         v
                 Select Better Model
```

The models were compared using:

* R² Score
* Adjusted R² Score
* Training R²
* Testing R²
* Generalisation Gap

## Model Evaluation

### R² Score

R² score was used to measure how well the regression model explains the variation in the target variable.

```python
from sklearn.metrics import r2_score

r2 = r2_score(
    y_test,
    y_pred
)
```

A higher R² indicates that the model explains a greater proportion of the variance in the target variable.

## Adjusted R²

Adjusted R² was also calculated to account for the number of predictors in the model.

```python
n = x_test.shape[0]
p = x_test.shape[1]

adjusted_r2 = 1 - (
    ((1-r2) * (n-1)) /
    (n-p-1)
)
```
