# Model Tuning

### What is Model Tuning?

Model tuning is the process of finding the **best set of hyperparameters** for a machine learning model so that it performs well on **unseen data**.

The first model we train is usually not necessarily the best version. We can improve its performance by changing its hyperparameters and evaluating the resulting models.

But if we just use random hyperparameters — our model might underfit, or overfit, or simply perform poorly.

So the goal of Model Tuning is:

The goal of model tuning is:

1. To find the best combination of hyperparameters.
2. To improve the model's performance on unseen data.
3. To reduce problems such as overfitting and underfitting.
4. To obtain a model that generalises well to new data.


In short — tuning helps you squeeze out the best possible performance from your model — and that’s why it’s an essential step before finalizing any ML model.

And this is also called **hyper parameter tuning**.

# Cross-Validation

**Cross-validation** is a model evaluation technique used to estimate how well a machine learning model will perform on **unseen data**.

Instead of relying on a single train-validation split, the training dataset is divided into multiple parts called **folds**.

---

### K-Fold Cross-Validation

The most common type is **K-Fold Cross-Validation**.

For example, in **5-Fold Cross-Validation**:

```text
Dataset
┌────┬────┬────┬────┬────┐
│ F1 │ F2 │ F3 │ F4 │ F5 │
└────┴────┴────┴────┴────┘

Round 1 → F1 = Validation, F2-F5 = Training
Round 2 → F2 = Validation, F1,F3-F5 = Training
Round 3 → F3 = Validation, F1,F2,F4-F5 = Training
Round 4 → F4 = Validation, F1-F3,F5 = Training
Round 5 → F5 = Validation, F1-F4 = Training
```

The model is trained and evaluated **K times**, with a different fold used for validation each time.

The scores are then usually averaged to obtain the final cross-validation score.

### Example

Suppose the five accuracy scores are:

```text
0.88, 0.91, 0.89, 0.90, 0.92
```

Average CV score:

```text
(0.88 + 0.91 + 0.89 + 0.90 + 0.92) / 5
= 0.90
```

Therefore:

```text
Cross-Validation Score = 90%
```

---

### Why Use Cross-Validation?

- Gives a more reliable estimate of model performance.
- Reduces dependence on a single train-validation split.
- Makes better use of the available training data.
- Helps compare different models.
- Helps evaluate different hyperparameter combinations.
- Is commonly used during **hyperparameter tuning**.

---

## Types of Cross-Validation

### 1. K-Fold Cross-Validation

The dataset is divided into `K` folds.

```text
K = 5  → 5 folds
K = 10 → 10 folds
```

Each fold gets a turn as the validation set.

---

### 2. Stratified K-Fold

Used mainly for **classification**.

It maintains approximately the same proportion of each class in every fold.

For example:

```text
Dataset:
80% Class A
20% Class B

Each fold:
≈ 80% Class A
≈ 20% Class B
```

This is particularly useful when the classes are imbalanced.

---

### 3. Leave-One-Out Cross-Validation (LOOCV)

Each individual observation is used as the validation set once.

For `N` observations:

```text
N rounds of training and validation
```

It can be computationally expensive for large datasets.

---

### Cross-Validation During Hyperparameter Tuning

Cross-validation is commonly used **inside the hyperparameter-tuning process**.

The process is:

```text
Different Hyperparameter Combinations
                ↓
         Cross-Validation
                ↓
        Calculate CV Scores
                ↓
        Compare the Scores
                ↓
       Select Best Parameters
```

## Methods of Hyperparameter Tuning

### 1. Manual Search

Manually selecting different hyperparameter values, training the model, and comparing its performance to find suitable hyperparameters.

### 2. Grid Search CV

Systematically testing **every possible combination** of predefined hyperparameter values to find the best combination.

### 3. Random Search CV

Randomly selecting and testing a specified number of hyperparameter combinations from a predefined search space.

---
# Grid Search CV

**Grid Search CV (GridSearchCV)** is a hyperparameter tuning method that tests **every possible combination** of specified hyperparameter values using **Cross-Validation**.

### How it works
1. Define hyperparameters and their possible values.
2. Create all possible combinations.
3. Evaluate each combination using Cross-Validation.
4. Select the combination with the **best CV score**.

**In short:**  
> Grid Search CV = Grid Search + Cross-Validation

# Random Search CV

**Random Search CV (RandomizedSearchCV)** is a hyperparameter tuning method that randomly tests a specified number of hyperparameter combinations from a given search space using **Cross-Validation**.

### How it works
1. Define hyperparameters and their possible values/ranges.
2. Randomly select a specified number of combinations.
3. Evaluate each combination using Cross-Validation.
4. Select the combination with the **best CV score**.

**In short:**  
> Random Search CV = Random Search + Cross-Validation

# Why Use Random Search CV Over Grid Search CV?

The main reason to use **Random Search CV over Grid Search CV** is **efficiency**, especially when there are many hyperparameters or a large search space.

## Example

Suppose we have 4 hyperparameters:

- `n_estimators` → 5 values
- `max_depth` → 5 values
- `min_samples_split` → 4 values
- `max_features` → 3 values

### Grid Search CV

Grid Search tests every possible combination:

**5 × 5 × 4 × 3 = 300 combinations**

If `cv=5`:

**300 × 5 = 1,500 model fits**

### Random Search CV

Suppose we use `n_iter=30`.

Random Search tests only **30 randomly selected combinations**.

If `cv=5`:

**30 × 5 = 150 model fits**

Therefore, Random Search can explore a large search space with much less computation.

## Key Difference

| Grid Search CV | Random Search CV |
|---|---|
| Tests **every combination** | Tests **random combinations** |
| Can become very expensive | Usually much faster |
| Good for small search spaces | Good for large search spaces |
| Number of combinations grows rapidly | Number of iterations is controlled using `n_iter` |

## Important Point

Random Search is not simply "less accurate".

It can sometimes find a very good combination with far fewer model evaluations because it explores the search space more broadly.

> **Grid Search CV → Exhaustive but expensive**

> **Random Search CV → Controlled and efficient**
