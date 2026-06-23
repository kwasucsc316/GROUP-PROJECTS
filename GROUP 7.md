### Group: 7

# Machine Learning Practical Report

---

# PART A: Heart Disease Prediction Using Logistic Regression and Random Forest

## 1. Introduction

Heart disease is one of the leading causes of death worldwide. Early detection plays an important role in improving patient outcomes and reducing health risks.

Machine learning techniques can assist healthcare professionals by analyzing patient medical records and identifying individuals who may be at risk of developing heart disease.

The objective of this project is to develop and compare machine learning classification models capable of predicting the presence of heart disease using patient medical data.

The algorithms used are:

* Logistic Regression
* Random Forest Classifier

---

## 2. Dataset Description

The dataset contains medical information collected from patients and is used to determine the presence or absence of heart disease.

### Features

* Age
* Sex
* Chest Pain Type (`cp`)
* Resting Blood Pressure (`trestbps`)
* Cholesterol (`chol`)
* Fasting Blood Sugar (`fbs`)
* Resting Electrocardiographic Results (`restecg`)
* Maximum Heart Rate Achieved (`thalach`)
* Exercise Induced Angina (`exang`)
* ST Depression (`oldpeak`)
* Slope
* Number of Major Vessels (`ca`)
* Thalassemia (`thal`)

### Target Variable

* `0` = No Heart Disease
* `1` = Heart Disease

---

## 3. Methodology

### Data Loading

```python
import pandas as pd

df = pd.read_csv("heart.csv")
```

### Importing Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

### Data Exploration

```python
df.head()
df.info()
df.describe()
```

### Feature Selection

```python
X = df.drop("target", axis=1)
y = df["target"]
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 4. Logistic Regression Model

```python
lr = LogisticRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
```

### Evaluation

```python
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
```

---

## 5. Random Forest Model

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
```

### Evaluation

```python
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
```

---

## 6. Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 79.51%   | 75.63%    | 87.38% | 81.08%   |
| Random Forest       | 98.54%   | 100.00%   | 97.09% | 98.52%   |

---

## 7. Discussion

The Random Forest model significantly outperformed Logistic Regression across all evaluation metrics.

### Random Forest Performance

* Accuracy = **98.54%**
* Precision = **100%**
* Recall = **97.09%**
* F1 Score = **98.52%**

The model successfully classified heart disease patients with very high accuracy and almost no false positive predictions.

### Logistic Regression Performance

Although Logistic Regression produced acceptable results, its performance was considerably lower than that of Random Forest.

This suggests that the relationship between the medical features and heart disease is not entirely linear, making Random Forest a better choice.

---

## 8. Conclusion

Both models successfully classified patients with and without heart disease.

However:

* Logistic Regression served as a useful baseline model.
* Random Forest achieved superior performance across all evaluation metrics.

### Recommended Model

**Random Forest Classifier**

because it provides:

* Higher accuracy
* Better precision
* Better recall
* Better F1-score

---

# PART B: Gold Price Prediction Using Linear Regression and Artificial Neural Network (ANN)

## 1. Introduction

Gold is one of the most valuable commodities in the global financial market and is widely regarded as a safe investment asset.

Predicting gold prices accurately can assist:

* Investors
* Financial Analysts
* Businesses

This project compares:

* Linear Regression
* Artificial Neural Network (MLPRegressor)

for Gold Price Prediction.

---

## 2. Dataset Description

### Features

* Date
* SPX (S&P 500 Index)
* USO (United States Oil Fund)
* SLV (Silver Price)
* EUR/USD Exchange Rate

### Target Variable

* GLD (Gold Price)

---

## 3. Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

## 4. Data Loading

```python
df = pd.read_csv("gld_price_data.csv")
```

---

## 5. Data Preprocessing

### Convert Date Column

```python
df["Date"] = pd.to_datetime(df["Date"])

df["Date"] = df["Date"].map(pd.Timestamp.toordinal)
```

### Define Features and Target

```python
X = df.drop("GLD", axis=1)

y = df["GLD"]
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

---

## 6. Linear Regression Model

```python
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
```

### Evaluation

```python
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R²:", r2_score(y_test, y_pred_lr))
```

### Results

| Metric   | Value  |
| -------- | ------ |
| MAE      | 5.17   |
| MSE      | 45.54  |
| R² Score | 0.9169 |

---

## 7. Artificial Neural Network (MLPRegressor)

### Model Configuration

```python
ann = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=1000,
    random_state=42
)
```

### Training

```python
ann.fit(X_train_scaled, y_train)

y_pred_ann = ann.predict(X_test_scaled)
```

### Evaluation

```python
print("MAE:", mean_absolute_error(y_test, y_pred_ann))
print("MSE:", mean_squared_error(y_test, y_pred_ann))
print("R²:", r2_score(y_test, y_pred_ann))
```

### Results

| Metric   | Value  |
| -------- | ------ |
| MAE      | 2.31   |
| MSE      | 8.95   |
| R² Score | 0.9837 |

---

## 8. Model Comparison

| Model              | MAE  | MSE   | R² Score |
| ------------------ | ---- | ----- | -------- |
| Linear Regression  | 5.17 | 45.54 | 0.9169   |
| MLPRegressor (ANN) | 2.31 | 8.95  | 0.9837   |

---

## 9. Discussion

Both models successfully predicted gold prices.

However, the ANN model clearly outperformed Linear Regression.

### Why ANN Performed Better

* Captures complex non-linear relationships.
* Learns hidden patterns within financial data.
* Produces lower prediction errors.

### Importance of Feature Scaling

Initially, ANN performance was poor.

After applying:

```python
StandardScaler()
```

the model performance improved significantly.

This demonstrates the importance of proper preprocessing when working with neural networks.

---

## 10. Conclusion

This project successfully applied:

* Linear Regression
* Artificial Neural Network (MLPRegressor)

to predict gold prices.

### Best Model

**Artificial Neural Network (MLPRegressor)**

### Final Performance

| Metric   | Value  |
| -------- | ------ |
| MAE      | 2.31   |
| MSE      | 8.95   |
| R² Score | 0.9837 |

The ANN model produced the most accurate predictions and is recommended for Gold Price Prediction.

---

# References

1. Kaggle — Heart Disease Dataset
   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

2. Kaggle — Gold Price Dataset
   https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data
