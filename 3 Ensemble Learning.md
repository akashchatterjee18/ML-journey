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
