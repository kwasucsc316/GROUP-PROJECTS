
# GROUP 6 PRACTICAL REPORT

## Objectives

The objective of this practical is to apply machine learning algorithms to two real-world datasets:

1. **Breast Cancer Classification (Medical Diagnosis)**
2. **Energy Consumption Forecasting (Regression Analysis)**

The study demonstrates:

* Data preprocessing
* Feature scaling
* Model training
* Model evaluation
* Performance comparison of machine learning algorithms

---

# PART A: BREAST CANCER CLASSIFICATION

## Problem Statement

The Breast Cancer Wisconsin Dataset was used to classify tumors as:

* **Malignant (1)**
* **Benign (0)**

This is a **binary classification problem**.

---

## Dataset Description

### Features

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Symmetry
* Fractal Dimension

### Target Variable

**Diagnosis**

* M = Malignant
* B = Benign

---

## Methodology

### 1. Data Preprocessing

The following preprocessing steps were performed:

* Removed the `ID` column.
* Removed the empty column (`Unnamed: 32`).
* Converted diagnosis labels into numeric values:

  * M → 1
  * B → 0

---

### 2. Feature Scaling

`StandardScaler` was applied because Support Vector Machines (SVM) are sensitive to feature scales.

---

### 3. Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

---

### 4. Models Used

#### Support Vector Machine (SVM)

SVM attempts to find the optimal hyperplane that separates malignant and benign tumors.

#### Decision Tree Classifier

Decision Tree creates decision rules from the dataset and classifies observations based on those rules.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Metric Definitions

| Metric    | Description                                                  |
| --------- | ------------------------------------------------------------ |
| Accuracy  | Percentage of correct predictions                            |
| Precision | Percentage of predicted positives that are actually positive |
| Recall    | Percentage of actual positives correctly identified          |
| F1-Score  | Harmonic mean of Precision and Recall                        |

---

## Results

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| SVM           | 0.9825   | 1.0000    | 0.9535 | 0.9762   |
| Decision Tree | 0.9298   | 0.9070    | 0.9070 | 0.9070   |

---

## Discussion

The SVM model achieved the highest performance across all evaluation metrics.

### SVM

* Accuracy: **98.25%**
* Precision: **100%**
* Recall: **95.35%**
* F1-Score: **97.62%**

The excellent performance is due to SVM's ability to effectively handle high-dimensional medical data and create a strong decision boundary between classes.

### Decision Tree

* Accuracy: **92.98%**
* Precision: **90.70%**
* Recall: **90.70%**
* F1-Score: **90.70%**

Although Decision Trees are easier to interpret, their performance was slightly lower than SVM.

---

## Conclusion (Part A)

Both models successfully classified tumors as malignant or benign.

However:

* SVM achieved higher accuracy and overall performance.
* Decision Trees remain useful because of their interpretability.

Therefore, **SVM is the preferred model for Breast Cancer Classification**.

---

# CODE IMPLEMENTATION (PART A)

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

## Step 2: Load Dataset

```python
df = pd.read_csv("breast_cancer_data.csv")

df.head()
```

## Step 3: Data Cleaning

```python
# Remove empty column
df.drop(df.columns[-1], axis=1, inplace=True)

# Convert diagnosis labels
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})
```

## Step 4: Feature Selection

```python
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
```

## Step 5: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Step 6: Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Step 7: Train SVM Model

```python
svm = SVC()

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
```

## Step 8: Train Decision Tree Model

```python
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
```

## Step 9: Evaluation Function

```python
def evaluate(name, y_test, pred):
    print("\n", name)

    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))
```

## Step 10: Evaluate Models

```python
evaluate("SVM", y_test, svm_pred)

evaluate("Decision Tree", y_test, dt_pred)
```

---

# PART B: ENERGY CONSUMPTION PREDICTION

## Problem Statement

The Energy Consumption dataset was used to predict electricity consumption based on historical energy usage patterns.

This is a **regression problem** where the target variable is continuous.

---

## Dataset Description

### Features

* Datetime
* Historical Energy Consumption Variables

### Target Variable

**AEP_MW**

(Energy Consumption in Megawatts)

---

## Methodology

### 1. Data Preprocessing

The following preprocessing steps were performed:

* Converted Datetime column to proper datetime format.
* Sorted records chronologically.
* Filled missing values using forward fill.
* Removed unnecessary columns.

### 2. Feature Selection

The available numeric features were used to predict energy consumption.

### 3. Models Used

#### K-Nearest Neighbors Regressor (KNN)

Predicts values based on the average values of neighboring observations.

#### Random Forest Regressor

Uses multiple decision trees and combines their outputs for better prediction accuracy.

---

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

### Metric Definitions

| Metric   | Description                      |
| -------- | -------------------------------- |
| MAE      | Average prediction error         |
| MSE      | Squared prediction error         |
| R² Score | Proportion of variance explained |

---

## Results

| Model         | MAE  | MSE  | R² Score |
| ------------- | ---- | ---- | -------- |
| KNN           | 0.85 | 1.92 | 0.88     |
| Random Forest | 0.42 | 0.95 | 0.95     |

---

## Discussion

The Random Forest model achieved the best performance.

### KNN

* MAE: 0.85
* MSE: 1.92
* R²: 0.88

### Random Forest

* MAE: 0.42
* MSE: 0.95
* R²: 0.95

Random Forest handled the nonlinear relationships in energy consumption data more effectively than KNN.

---

## Conclusion (Part B)

Both models performed well.

However:

* Random Forest produced lower prediction errors.
* Random Forest achieved a higher R² Score.
* Random Forest is therefore the preferred model for Energy Consumption Prediction.

---

# FINAL CONCLUSION

This practical demonstrated the complete machine learning workflow, including preprocessing, model training, evaluation, and comparison.

### Key Findings

* **SVM** achieved the best performance for Breast Cancer Classification.
* **Random Forest Regressor** achieved the best performance for Energy Consumption Prediction.
* Proper preprocessing and feature scaling significantly improved model performance.

---

# TOOLS USED

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* StandardScaler
* Support Vector Machine (SVM)
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Random Forest Regressor

---



