# Ensemble Learning

## What is Ensemble Learning?

**Ensemble Learning** is a machine learning technique where we combine the predictions of **multiple machine learning models** to make a final prediction.

## Basic Idea

Suppose we have a classification problem and train multiple models on the same dataset.

```text
                    Machine Learning

Train
Test ──→ M₁ ──→ M₂ ──→ M₃ ──→ M₄
         LR      SVM     NB      KNN
```

Where:

- **M₁ → Logistic Regression (LR)**
- **M₂ → Support Vector Machine (SVM)**
- **M₃ → Naive Bayes (NB)**
- **M₄ → K-Nearest Neighbours (KNN)**

Each model makes its own prediction.

The predictions are then **combined** to produce the final prediction.

---

## Example

Suppose we have a test sample and the models make the following predictions:

| Model | Prediction |
|---|---|
| Logistic Regression | A |
| SVM | A |
| Naive Bayes | B |
| KNN | A |

The ensemble receives:

```text
A, A, B, A
```

Using **majority voting**:

```text
A → 3 votes
B → 1 vote
```

Therefore, the final prediction is:

```text
A
```

This is the intuition behind:

> **"We hear the crowd."**

---

## Why Ensemble Learning?

A single model may make mistakes.

Different models may make **different mistakes** because they learn patterns in different ways.

By combining multiple models, we can use the collective information from all of them.

### Main Idea

```text
Multiple Models
      ↓
Multiple Predictions
      ↓
Combine Predictions
      ↓
Final Prediction
```

---

## Types of Ensemble Learnimg

<img width="1918" height="820" alt="image" src="https://github.com/user-attachments/assets/8620c0b2-d1d6-4137-bac1-f8dcdab03edc" />

---

## Stacking

**Stacking (Stacked Generalization)** is an **ensemble learning technique** that combines multiple different machine learning models using another model called a **meta-model**.

Instead of simply averaging or voting the predictions, stacking trains a meta-model to learn **how to combine the predictions of the base models**.


## How Stacking Works

Stacking consists of two levels:

### Base Models

Multiple different models are trained on the same dataset.

Examples:
- Logistic Regression
- KNN
- Decision Tree
- SVM
- Naive Bayes

These models are called **base learners** or **level-0 models**.

### Meta-Model

The predictions made by the base models are used as **features** for another model.

This model learns how to combine the base-model predictions.

It is called the **meta-model**, **meta-learner**, or **level-1 model**.

---

## Basic Structure

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/86c326ec-4196-4e29-8009-913f9874ac41" />

---
## Bagging

**Bagging (Bootstrap Aggregating)** is an **ensemble learning technique** that trains multiple models on different **bootstrap samples** of the training data and combines their predictions.

**Main goal:** Reduce **variance** and **overfitting**.



## How Bagging Works

### 1. Bootstrap Sampling
Create multiple datasets using **random sampling with replacement**.

### 2. Train Models
Train a separate model on each bootstrap sample.

### 3. Aggregate Predictions
Combine the predictions:

- **Classification → Majority Voting**
- **Regression → Averaging**


## Flow

```text
Training Data
      ↓
Bootstrap Samples
      ↓
Multiple Models
      ↓
Predictions
      ↓
Voting / Averaging
      ↓
Final Prediction
```

# Random Forest Classifier

Random Forest Classifier is a **supervised ensemble learning algorithm** used for **classification**.

It combines multiple **Decision Trees** and uses **majority voting** to make the final prediction.

Random Forest is based on **Bagging** and **Random Feature Selection**.

## How does Random Forest work?

1. Create multiple **bootstrap samples** from the training data using sampling with replacement.
2. Train a **Decision Tree** on each bootstrap sample.
3. At each split, only a **random subset of features** is considered.
4. Each Decision Tree makes a prediction.
5. The predictions are combined using **majority voting**.
6. The class with the most votes becomes the final prediction.

## Basic Structure

```text
Training Data
      |
      ↓
Bootstrap Sampling
      |
      ↓
Multiple Decision Trees
      |
      ↓
Random Feature Selection
      |
      ↓
Predictions from Trees
      |
      ↓
Majority Voting
      |
      ↓
Final Prediction
```

## Why Random Feature Selection?

If every tree uses all the features, the trees can become very similar.

Randomly selecting features makes the trees **more diverse**.

This helps reduce **variance and overfitting**.

## Random Forest vs Decision Tree

| Decision Tree | Random Forest |
|---|---|
| Single tree | Multiple trees |
| Higher variance | Lower variance |
| More prone to overfitting | Less prone to overfitting |
| Uses all features at a split | Uses a random subset of features |

## Important Hyperparameters

- **n_estimators** → Number of Decision Trees.
- **max_depth** → Maximum depth of each tree.
- **max_features** → Number of features considered at each split.
- **min_samples_split** → Minimum samples required to split a node.
- **min_samples_leaf** → Minimum samples required in a leaf.

## Key Takeaway

$$
\boxed{\text{Random Forest = Bagging + Random Feature Selection}}
$$

For classification:

$$
\boxed{\text{Final Prediction = Majority Vote of all Trees}}
$$




## Boosting

**Boosting** is an **ensemble learning technique** that combines multiple **weak learners sequentially** to create a strong learner.

Each new model focuses on the **errors made by previous models**.

**Main goal:** Reduce **bias** and improve prediction performance.

---

## How Boosting Works

### 1. Train a Model
Train the first weak learner on the training data.

### 2. Focus on Errors
Identify the observations that were incorrectly predicted.

### 3. Train the Next Model
The next model gives more importance to previous errors.

### 4. Repeat
Multiple models are trained **sequentially**, with each model improving upon the previous ones.

### 5. Combine Predictions
The predictions of all models are combined to produce the final prediction.

---

## Flow

```text
Training Data
      ↓
Weak Learner 1
      ↓
Focus on Errors
      ↓
Weak Learner 2
      ↓
Focus on Errors
      ↓
Weak Learner 3
      ↓
      ...
      ↓
Combine Predictions
      ↓
Final Prediction
```

## Note
```text
In Bagging / Random Forest, it converts a **low-bias, high-variance (overfitting)** model into a more **generalised, low-bias, low-variance** model.

In Boosting, it converts a **high-bias, low-variance (underfitting)** model into a more **generalised, low-bias, low-variance** model.
```

### Formula

$$
F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)
$$

Where:

- $h_m(x)$ = prediction of the $m^{th}$ weak learner
- $\alpha_m$ = weight assigned to the $m^{th}$ learner
- $M$ = number of weak learners
- $F(x)$ = final boosted model
