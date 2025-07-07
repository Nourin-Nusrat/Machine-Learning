# Machine Learning  
## Assignment 1: Linear Models in ML

### Overview

This assignment has been designed to demonstrate how linear models work for both regression and classification tasks using `scikit-learn` and `mglearn`.

---

### 1. Linear Regression

A simple linear regression model has been trained on the extended Boston dataset.

**Equation:**  
ŷ = w₀ + w₁·x₁ + w₂·x₂ + ... + wₚ·xₚ

Model coefficients and intercept have been printed to understand the relationship.

---

### 2. Ridge Regression (L2)

Ridge adds a penalty on the squared coefficients.

**Loss Function:**  
Loss = MSE + α * Σ (wⱼ)²

Models have been trained with different `alpha` values to compare performance and coefficient shrinkage.

---

### 3. Lasso Regression (L1)

Lasso encourages sparsity by adding a penalty on the absolute values of the coefficients.

**Loss Function:**  
Loss = MSE + α * Σ |wⱼ|

Different `alpha` values have been tested to observe feature selection and regularization strength.

---

### 4. Classification (Logistic Regression vs LinearSVC)

Two classifiers have been trained on the Forge dataset and their decision boundaries plotted:

- Logistic Regression: Probabilistic model  
- LinearSVC: Margin-based model (SVM)

---

### Plots

- Coefficients for Linear, Ridge, and Lasso regressions  
- Decision boundaries for classifiers using Forge dataset

---

# Assignment 2: Classification with Linear Models

### Overview

This assignment focuses on linear classification using `scikit-learn` and `mglearn`. Several classifiers have been trained and visualized, including Logistic Regression and Linear Support Vector Classifier (SVC).

---

### 1. Forge Dataset Classification

The `make_forge()` dataset has been used to train:

- Logistic Regression  
- LinearSVC  

Decision boundaries have been visualized in 2D space to understand model behavior.

---

### 2. Effect of Regularization on Logistic Regression

A Logistic Regression model has been trained on the breast cancer dataset with different values of C (inverse of regularization strength):

- C = 1 (default)  
- C = 100 (low regularization)  
- C = 0.01 (high regularization)

**Equation:**  
P(y = 1 | x) = 1 / (1 + exp(-(wᵀ·x + b)))

The magnitude of feature coefficients has been plotted to visualize regularization effects.

---

### 3. L1 Regularization (Lasso-like Logistic Regression)

Logistic Regression with L1 penalty (`penalty="l1"`) has been used to enforce sparsity in coefficients.

**Loss Function:**  
Loss = -LogLikelihood + α * Σ |wⱼ|

Different values of `C` have been tested to show how L1 affects the number of non-zero weights (feature selection).

---

### 4. Multi-Class Classification with SVM

A synthetic 3-class dataset created with `make_blobs()` has been used with LinearSVC.

Each class is separated using its own linear decision boundary.

**Equation for decision line:**  
w₀·x + w₁·y + b = 0  ⇒  y = -(w₀·x + b) / w₁

Decision boundaries and classification regions have been plotted using `mglearn.plots.plot_2d_classification()`.

---

### Summary

- Linear models have been applied to binary and multi-class classification.
- The effect of L2 (C) and L1 (penalty) regularization has been visualized.
- Decision boundaries and coefficient magnitudes have been analyzed.

---

### Requirements

To run the code, install the following packages:

```bash
pip install scikit-learn matplotlib mglearn numpy
