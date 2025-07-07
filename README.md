# Machine-Learning 
# Assignment 1: Linear Models in ML

## Overview

This assignment has been designed to demonstrate how linear models work for both regression and classification tasks using `scikit-learn` and `mglearn`.

---

## 1. Linear Regression

A simple linear regression model has been trained on the extended Boston dataset.

**Equation:**

\[
\hat{y} = w_0 + w_1 x_1 + \cdots + w_p x_p
\]

Model coefficients and intercept have been printed to understand the relationship.

---

## 2. Ridge Regression (L2)

Ridge adds a penalty on the squared coefficients:

\[
\text{Loss} = \text{MSE} + \alpha \sum w_j^2
\]

Models have been trained with different `alpha` values to compare performance and coefficient shrinkage.

---

## 3. Lasso Regression (L1)

Lasso encourages sparsity:

\[
\text{Loss} = \text{MSE} + \alpha \sum |w_j|
\]

Models with various `alpha` values have been used to observe feature selection.

---

## 4. Classification (Logistic Regression vs LinearSVC)

Two classifiers have been trained on the Forge dataset and their decision boundaries plotted.

- **Logistic Regression**: Probabilistic
- **LinearSVC**: Margin-based

---

## Plots

- Coefficients for Linear, Ridge, and Lasso regressions
- Decision boundaries for classifiers

---

# Assignment 1: Classification with Linear Models

## Overview

This assignment has been focused on understanding linear classification using `scikit-learn` and `mglearn`. Several classifiers have been trained and visualized, including Logistic Regression and Linear Support Vector Classifier (SVC).

---

## 1. Forge Dataset Classification

The `make_forge()` synthetic dataset has been used to train:
- **Logistic Regression**
- **Linear SVC**

Their decision boundaries have been plotted to show how both models separate classes linearly in 2D space.

---

## 2. Effect of Regularization on Logistic Regression

A logistic regression model has been trained on the **breast cancer dataset** using different `C` values (inverse of regularization strength):
- `C=1` (default)
- `C=100` (less regularization)
- `C=0.01` (more regularization)

**Equation:**
\[
P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}
\]

The magnitude of each feature coefficient has been plotted to show how regularization affects feature importance.

---

## 3. L1 Regularization (Lasso-like Logistic Regression)

Logistic regression with `penalty="l1"` has been used to enforce sparsity in the feature weights:

\[
\text{Loss} = - \text{LogLikelihood} + \alpha \sum |w_j|
\]

Different `C` values have been tested to show how L1 affects the number of non-zero coefficients (feature selection).

---

## 4. Multi-Class Classification with SVM

A synthetic **3-class dataset** from `make_blobs()` has been created, and a **LinearSVC** has been trained.

- Each class has been separated using its own linear decision boundary.
- The equations of the decision lines:
  \[
  w_0x + w_1y + b = 0 \Rightarrow y = -\frac{w_0 x + b}{w_1}
  \]
  have been visualized for all three classes.

Additionally, the SVM prediction regions have been plotted using `mglearn.plots.plot_2d_classification()`.

---

## Summary

- Linear classifiers have been applied to binary and multi-class problems.
- Effects of L2 (`C`) and L1 (`penalty="l1"`) regularization have been analyzed.
- Decision boundaries and coefficients have been visualized.

---

## Requirements

```bash
pip install scikit-learn matplotlib mglearn numpy
