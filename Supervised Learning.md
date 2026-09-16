~~~~
Regression
~~~~
Regression is a supervised machine learning algorithm used to predict continuous numerical values.
Example of Regression : Insurance Prediction, Footballers Market Value Prediction

Linear Regression is a supervised machine learning algorithm that models the relationship between independent variables and a continuous dependent variable by fitting the best-fit linear equation.

The best-fit line is the straight line that minimises the sum of the squared differences (residuals) between the actual values and the predicted values, providing the best linear relationship between the input features and the target variable.

y = mx + b

y = prediction of target

m = slope

x = datapoint to be used to predict y

b = y-intercept

How to find the perfect value of m and b?
Here comes the concept of residual error...

Residual Error is the difference between the actual value and the value predicted(Best Fit Line) by the regression model for a data point.

Residual Error = Ya - Yp

Mean Squared Error (MSE) is the average of the squared differences between the actual values and the predicted values used to measure the accuracy of a regression model.

<img width="321" height="124" alt="image" src="https://github.com/user-attachments/assets/19bcc9af-55cc-4e51-9e4d-e92a22acef18" />

Cost Function is a mathematical function that measures the average prediction error of a machine learning model. The objective of linear regression is to minimise the cost function to obtain the best-fit line.

<img width="355" height="121" alt="image" src="https://github.com/user-attachments/assets/d0d8ac5e-f953-41f1-b36a-a6ab638694f4" />

b = y-intercept

m = slope

Relation Between J and MSE

<img width="148" height="77" alt="image" src="https://github.com/user-attachments/assets/53b48e7f-3eea-4947-9df8-6e65471e504f" />

We have to minimise the cost function by changing the value of b and m by using the concept of Gradient Descent.
Gradient Descent is an optimization algorithm used to minimize the cost function by iteratively updating the model parameters in the direction of the steepest decrease in cost.

How Gradient Descent Works?
* Initialize random values for b and m.
* Compute the cost function J(b,m).
* Calculate the gradients with respect to b and m.
* Update b and m using the update equations.
* Repeat until the cost function reaches its minimum.

Repeat Convergence Theorem

Gradient Descent converges to the global minimum for a convex cost function if the learning rate is chosen appropriately.

<img width="549" height="399" alt="image" src="https://github.com/user-attachments/assets/9b786fb2-a96e-4826-9c98-1cae9160491f" />


Why use a low learning rate?
* Large learning rates can jump over the minimum point repeatedly and never converge.
* Small updates allow the algorithm to gradually approach the minimum cost.
* Smaller steps help the model settle closer to the true minimum.

Now what to do for multiple input and output variable?

Multiple Linear Regression comes into play.

<img width="652" height="454" alt="image" src="https://github.com/user-attachments/assets/4f706703-959c-4bbb-b868-da0ff1b24c1a" />

Hyperplane in Multiple Linear Regression

In Multiple Linear Regression, a hyperplane is the best-fit flat surface that represents the relationship between multiple independent variables and a single dependent variable.

# Evaluation Metrics

## R² (R-Squared)

**R²**, also known as the **coefficient of determination**, measures how well the independent variables explain the variation in the dependent variable.

It represents the proportion of the variance in the target variable that is explained by the regression model.

### Formula

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Where:

* $SS_{res}$ = Residual Sum of Squares
* $SS_{tot}$ = Total Sum of Squares

More explicitly:

$$
SS_{res} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

$$
SS_{tot} = \sum_{i=1}^{n}(y_i-\bar{y})^2
$$

Where:

* $y_i$ = Actual value
* $\hat{y}_i$ = Predicted value
* $\bar{y}$ = Mean of the actual target values
* $n$ = Number of observations

### Interpretation

R² generally ranges from **0 to 1** in ordinary regression with an intercept, although it can be negative on unseen/test data.

* **R² = 1** → The model explains 100% of the variation in the target.
* **R² = 0** → The model explains none of the variation beyond predicting the mean.
* **Higher R²** → The model explains more of the variation in the target variable.

For example, if:

$$
R^2 = 0.80
$$

then approximately **80% of the variation** in the target variable is explained by the model, while the remaining **20%** is not explained by the model.

> **Important:** A high R² does not necessarily mean that the model is good or that it will generalise well to unseen data. R² should be considered along with test-set performance and other evaluation metrics.

---

## Adjusted R²

**Adjusted R²** is a modified version of R² that takes the **number of predictors (features)** and the **number of observations** into account.

Unlike R², adjusted R² introduces a penalty for adding unnecessary predictors to the model.

### Formula

$$
Adjusted\ R^2 =
1-\left(\frac{(1-R^2)(n-1)}{n-p-1}\right)
$$

Where:

* $R^2$ = R-Squared
* $n$ = Number of observations
* $p$ = Number of independent variables/predictors

### Why Adjusted R² is Used

R² has an important limitation:

> **R² never decreases when additional predictors are added to a regression model.**

Even if a new feature contributes very little useful information, R² may increase slightly.

Adjusted R² addresses this problem by applying a **penalty for additional predictors**.

Therefore:

* If a new predictor provides meaningful information → Adjusted R² may increase.
* If a new predictor provides little or no useful information → Adjusted R² may decrease.
* If the additional predictor does not improve the model enough to justify its inclusion → Adjusted R² penalises the model.

---

## R² vs Adjusted R²

| Metric                                                         | R²            | Adjusted R²   |
| -------------------------------------------------------------- | ------------- | ------------- |
| Measures explained variance                                    | Yes           | Yes           |
| Considers number of predictors                                 | No            | Yes           |
| Penalises unnecessary predictors                               | No            | Yes           |
| Can decrease when a feature is added                           | No            | Yes           |
| Useful for comparing models with different numbers of features | Less suitable | More suitable |

### Example

Suppose a model has:

$$
R^2 = 0.85
$$

and:

$$
Adjusted\ R^2 = 0.84
$$

This means the model explains approximately **85% of the variation** in the target variable, while after accounting for the number of predictors, the adjusted measure is approximately **84%**.

If another feature is added and the results become:

$$
R^2 = 0.86
$$

but:

$$
Adjusted\ R^2 = 0.83
$$

then the increase in R² does not necessarily indicate an improvement in the model. The decrease in adjusted R² suggests that the additional predictor may not provide enough useful information relative to the complexity it adds.

---

## Key Difference

The main difference can be summarised as:

> **R² tells us how much variation the model explains, whereas Adjusted R² tells us how much variation the model explains while accounting for the number of predictors used.**

For regression models containing many features, **Adjusted R² can provide a more informative measure of model fit than R² alone.**

~~~~
Classification
~~~~
Classification is a supervised machine learning algorithm used to predict categorical class labels.
Example of Classification : Heart Disease Prediction (Yes/No)

