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
