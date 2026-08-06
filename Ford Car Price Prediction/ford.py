import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('ford.csv')
print(df.head())

# EDA
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.duplicated().sum())


sns.histplot(df['price'],kde = True)
plt.show()

sns.heatmap(df.corr(numeric_only = True),annot = True)
plt.show()

sns.boxplot(data = df, x = 'year', y = 'price')
plt.xticks(rotation = 90)
plt.show()

sns.scatterplot(data = df, x = 'mileage',y = 'price')
plt.show()

sns.boxplot(data = df, x = 'engineSize', y = 'price')
plt.show()

sns.boxplot(data = df, x = 'transmission',y = 'price')
plt.show()

sns.boxplot(data = df, x = 'fuelType', y = 'price' )
plt.show()

sns.boxplot(data = df, x = 'model',y = 'price')
plt.xticks(rotation = 90)
plt.show()

sns.boxplot(data = df, x = 'tax', y = 'price' )
plt.xticks(rotation = 90)
plt.show()

sns.boxplot(data = df, x = 'mpg', y = 'price' )
plt.xticks(rotation = 90)
plt.show()


x = df.drop('price',axis = 1)
y = df['price']


x_1hot = pd.get_dummies(x,columns = ['model','transmission','fuelType'],dtype= int)

from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()

columns = ['model','transmission','fuelType']
x_label = x

for i in columns:
    x_label[i] = encode.fit_transform(x_label[i])

from sklearn.preprocessing import StandardScaler
numerical_colms = ['year','mileage','tax','mpg','engineSize']
scaler = StandardScaler()

x_1hot[numerical_colms] = scaler.fit_transform(x_1hot[numerical_colms])

x_label[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg','engineSize']] = scaler.fit_transform(x_label[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg','engineSize']])


# ML Model for 1-hot encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


x_train, x_test, y_train, y_test = train_test_split(x_1hot,y,test_size =0.2, random_state =42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

r2 = r2_score(y_test,y_pred)
print(f'R2 value of 1-hot encoded model : {r2}')

n = x_test.shape[0]
p = x_test.shape[1]
adjusted_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
print(f'Adjusted R2 value of 1-hot encoded model : {adjusted_r2}')


# Overfitting / Underfitting check

# Training predictions
y_train_pred = model.predict(x_train)

# Testing predictions
y_test_pred = model.predict(x_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training R² of 1-hot encoded model :", train_r2)
print("Testing R² of 1-hot encoded model :", test_r2)
print(f"Generalization Gap of 1-hot encoded model : {abs(train_r2 - test_r2)}")


# ML model for label encoding
x_train, x_test, y_train, y_test = train_test_split(x_label,y,test_size =0.2, random_state =42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

r2 = r2_score(y_test,y_pred)
print(f'R2 value of label encoded model : {r2}')

n = x_test.shape[0]
p = x_test.shape[1]
adjusted_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
print(f'Adjusted R2 value of label encoded model : {adjusted_r2}')


# Overfitting / Underfitting check

# Training predictions
y_train_pred = model.predict(x_train)

# Testing predictions
y_test_pred = model.predict(x_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training R² of label encoded model :", train_r2)
print("Testing R² of label encoded model :", test_r2)
print(f"Generalization Gap of label encoded model : {abs(train_r2 - test_r2)}")


## here model 2 (label encoded) have less accurance than model 1 (1-hot encoded)

# thus first model was better

