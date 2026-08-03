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


Classification is a supervised machine learning algorithm used to predict categorical class labels.
Example of Classification : Heart Disease Prediction (Yes/No)

