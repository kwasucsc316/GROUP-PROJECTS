# MACHINE LEARNING PRACTICAL REPORT  
## Comparative Analysis of Classification and Regression Models  

---

## 1. Introduction

This project applies machine learning techniques to solve both classification and regression problems.

- **Classification Task:** Toxic Comment Classification  
- **Regression Task:** California Housing Price Prediction  

The aim is to compare different machine learning algorithms and evaluate their performance using appropriate metrics.

---

# 2. Classification Task: Toxic Comment Detection

## 2.1 Dataset Description

The dataset contains user comments labeled across multiple toxicity categories:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

A binary target was created:

- `1` → Toxic comment  
- `0` → Non-toxic comment  

---

## 2.2 Data Preprocessing

### Load Dataset

```python
import pandas as pd
import numpy as np
import sklearn

print("Machine Learning setup is ready!")
````

```python
toxic = pd.read_csv("Classification/train.csv")
toxic.head()
```

---

### Missing Values Check

```python
toxic.isnull().sum()
```

---

### Target Creation

```python
labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

toxic['target'] = toxic[labels].sum(axis=1)

toxic['target'] = toxic['target'].apply(lambda x: 1 if x > 0 else 0)

toxic[['comment_text', 'target']].head()
```

---

### TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english'
)

X = vectorizer.fit_transform(toxic['comment_text'])
y = toxic['target']
```

---

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 2.3 Model Training

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
```

---

### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=20, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
```

---

## 2.4 Evaluation Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.9565   | 0.9253    | 0.6224 | 0.7442   |
| Decision Tree       | 0.9432   | 0.8916    | 0.5018 | 0.6422   |

---

## 2.5 Discussion

* Logistic Regression performed better due to effectiveness with high-dimensional TF-IDF data.
* Decision Tree underperformed due to overfitting and difficulty handling sparse text vectors.

---

## 2.6 Conclusion (Classification)

**Best Model:** Logistic Regression
It achieved higher accuracy and better balance between precision and recall.

---

# 3. Regression Task: California Housing Price Prediction

## 3.1 Dataset Description

The dataset contains housing features used to predict median house values.

### Features

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity

### Target

* median_house_value

---

## 3.2 Data Preprocessing

### Load Dataset

```python
housing = pd.read_csv("Regression/housing.csv")
housing.head()
```

---

### Handle Missing Values

```python
housing['total_bedrooms'] = housing['total_bedrooms'].fillna(
    housing['total_bedrooms'].median()
)
```

---

### Encode Categorical Data

```python
housing = pd.get_dummies(
    housing,
    columns=['ocean_proximity'],
    drop_first=True
)
```

---

### Train-Test Split

```python
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 3.3 Model Training

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
```

---

### Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

---

## 3.4 Evaluation Results

| Model             | MAE      | MSE              | R² Score |
| ----------------- | -------- | ---------------- | -------- |
| Linear Regression | 50670.74 | 4,908,476,721.16 | 0.6254   |
| Random Forest     | 31639.37 | 2,404,745,975.12 | 0.8165   |

---

## 3.5 Discussion

* Linear Regression struggled with complex relationships in housing data.
* Random Forest performed better by capturing non-linear patterns.

---

## 3.6 Conclusion (Regression)

**Best Model:** Random Forest Regression
It produced lower error and higher R² score.

---

# 4. Final Conclusion

* **Best Classification Model:** Logistic Regression
* **Best Regression Model:** Random Forest

### Key Insight:

Model performance depends heavily on data type:

* Linear models → better for sparse text data
* Tree-based models → better for structured numeric data

---
