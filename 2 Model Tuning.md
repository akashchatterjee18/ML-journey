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

## Cross-Validation

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

## Cross-Validation During Hyperparameter Tuning

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

For example:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
```

Here:

- `GridSearchCV` → searches for the best hyperparameters.
- `cv=5` → uses 5-fold cross-validation.
- `scoring="accuracy"` → uses accuracy to evaluate each combination.
- `best_params_` → gives the best hyperparameter combination.

---

## Cross-Validation vs Hyperparameter Tuning

They are closely related but serve different purposes.

**Cross-Validation** asks:

> "How well does this model perform across different subsets of the training data?"

**Hyperparameter Tuning** asks:

> "Which hyperparameter settings are best?"

They are commonly used together:

```text
Hyperparameter Combinations
          ↓
   Cross-Validation
          ↓
   Compare CV Scores
          ↓
 Best Hyperparameters
```

---

## Important: Cross-Validation and the Test Set

Cross-validation is generally performed on the **training data**.

The test set should remain separate until the final evaluation.

```text
Dataset
   ↓
Train/Test Split
   ↓
Training Data
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Best Model
   ↓
Final Evaluation
   ↓
Test Data
```

The test set should represent **unseen data**.

If we repeatedly use the test set to select hyperparameters, it is no longer an independent final evaluation.

---

## Advantages

- More reliable estimate of generalisation performance.
- Makes efficient use of available training data.
- Useful for comparing models.
- Useful for hyperparameter tuning.
- Less dependent on a single random train-validation split.

---

## Limitations

- More computationally expensive than a single train-validation split.
- The model must be trained multiple times.
- Large datasets and complex models can make cross-validation time-consuming.
- Standard K-Fold may not be appropriate for every type of data, such as time-series data.

---

## Key Terms

| Term | Meaning |
|---|---|
| **Fold** | One subset of the dataset |
| **K** | Number of folds |
| **Training Fold** | Data used to train the model |
| **Validation Fold** | Data used to evaluate the model during CV |
| **CV Score** | Performance score obtained through cross-validation |
| **K-Fold CV** | Dataset is divided into K folds and evaluated K times |
| **Stratified K-Fold** | K-Fold that preserves class proportions |
| **LOOCV** | Each observation is used as the validation set once |

---

## Key Point

> **Cross-validation = repeatedly train and validate a model on different parts of the training data to estimate its generalisation performance.**
methods of hyper parameter tuning

The main methods of hyperparameter tuning are:

Hyperparameter Tuning
│
├── 1. Manual Search
├── 2. Grid Search
├── 3. Random Search
├── 4. Bayesian Optimization
├── 5. Successive Halving
└── 6. Evolutionary / Genetic Algorithms
1. Manual Search

You manually choose hyperparameter values, train the model, and compare the results.

Try values → Train → Evaluate → Adjust → Repeat

Simple, but inefficient for large search spaces.

2. Grid Search

Tests every possible combination from a predefined set of values.

param_grid = {
    "max_depth": [3, 5, 10],
    "n_estimators": [50, 100, 200]
}

Total combinations:

3 × 3 = 9

Implemented using GridSearchCV.

3. Random Search

Randomly selects a specified number of combinations instead of testing every combination.

Implemented using RandomizedSearchCV.

Useful when the search space is large.

4. Bayesian Optimization

Uses the results from previous trials to intelligently decide which hyperparameters to try next.

Try → Evaluate → Learn from result
                  ↓
            Choose next values
                  ↓
                Try again

More efficient than blindly searching a huge space.

5. Successive Halving

Starts with many configurations, evaluates them with limited resources, and progressively eliminates poor-performing configurations.

Many configurations
        ↓
  Evaluate cheaply
        ↓
Remove poor ones
        ↓
Evaluate remaining ones more
        ↓
Remove more
        ↓
Best configurations

Scikit-learn provides HalvingGridSearchCV and HalvingRandomSearchCV.

6. Evolutionary / Genetic Algorithms

Uses ideas inspired by biological evolution.

Population
    ↓
Evaluate
    ↓
Select better configurations
    ↓
Crossover + Mutation
    ↓
New population
    ↓
Repeat

Useful for complex search spaces, but more advanced.

For your current ML notes

I'd organise it as:

Model Tuning / Hyperparameter Tuning
│
├── Hyperparameters
│
├── Cross-Validation
│
└── Methods of Hyperparameter Tuning
    ├── Manual Search
    ├── Grid Search
    ├── Random Search
    ├── Bayesian Optimization
    ├── Successive Halving
    └── Evolutionary / Genetic Algorithms

For your current level, focus particularly on Grid Search, Random Search, and Cross-Validation first. Bayesian optimisation and the others can come afterwards.

provide the notes in .md format. proper note like

also it should be as short as possible with all required info
secondly include only the first 3
thirdly dont explain all of them now

just provide the definitions then we can study one by one

## Methods of Hyperparameter Tuning

### 1. Manual Search

Manually selecting different hyperparameter values, training the model, and comparing its performance to find suitable hyperparameters.

### 2. Grid Search

Systematically testing **every possible combination** of predefined hyperparameter values to find the best combination.

### 3. Random Search

Randomly selecting and testing a specified number of hyperparameter combinations from a predefined search space.
