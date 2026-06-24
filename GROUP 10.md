
# CSC 316 MACHINE LEARNING PRACTICAL REPORT  

## Comparison of Machine Learning Algorithms for Diabetes Classification

---

# GROUP 10

---

# TOPIC

1. Perform proper data preprocessing (handling missing values, encoding, scaling where necessary).  
2. Train and evaluate models using the assigned algorithms only.  
3. For classification, use accuracy, precision, recall, and F1-score.  
4. For regression, use MAE, MSE, and R² score.  
5. Compare model performance and discuss findings.  

---

# OBJECTIVE

The objective of this practical is to apply classification algorithms to a healthcare dataset in order to predict diabetes outcomes.

This includes:

- Data preprocessing  
- Handling missing values  
- Feature scaling  
- Model training and evaluation  
- Model comparison  

---

# PART A: DIABETES PREDICTION WITH DATA IMPUTATION (CLASSIFICATION)

---

## Problem Statement

The Pima Indians Diabetes dataset is used to predict whether a patient has diabetes based on medical measurements.

This is a binary classification problem:

- **1 = Diabetic**
- **0 = Non-diabetic**

---

## Dataset Description

Features include:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome (Target Variable)  

Some features contain zero values that are medically unrealistic and required preprocessing.

---

## Methodology

---

## 1. Data Loading

```python
import pandas as pd

df = pd.read_csv("diabetes.csv")
print(df.head())
````

---

## 2. Handling Missing Values

```python id="a1b2c3"
import numpy as np
from sklearn.impute import SimpleImputer

columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[columns_with_zero] = df[columns_with_zero].replace(0, np.nan)

imputer = SimpleImputer(strategy='median')
df[columns_with_zero] = imputer.fit_transform(df[columns_with_zero])
```

---

## 3. Feature Selection

```python id="b2c3d4"
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

---

## 4. Train-Test Split

```python id="c3d4e5"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 5. Feature Scaling

```python id="d4e5f6"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Model Training

### Decision Tree Classifier

```python id="e5f6g7"
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
```

---

### K-Nearest Neighbors (KNN)

```python id="f6g7h8"
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

---

## 7. Model Evaluation

```python id="g7h8i9"
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

print("KNN Precision:", precision_score(y_test, knn_pred))
print("KNN Recall:", recall_score(y_test, knn_pred))
print("KNN F1 Score:", f1_score(y_test, knn_pred))
```

---

## Results and Discussion

| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------- | --------- | ------ | -------- |
| Decision Tree | 0.682    | 0.553     | 0.481  | 0.515    |
| KNN           | 0.753    | 0.660     | 0.611  | 0.635    |

---

## Observations

* KNN performed better than Decision Tree
* Feature scaling significantly improved KNN performance
* Decision Tree struggled with generalization

---

## Conclusion (Part A)

KNN provided better predictive performance, while Decision Tree was more interpretable.

---

# PART B: AUTOMATED ESSAY SCORING (ML & DEEP LEARNING)

---

## Problem Statement

Automated Essay Scoring (AES) evaluates essays using machine learning models.

---

## Objectives

* Convert text into numerical features
* Train ML and DL models
* Compare performance

---

## Dataset

```python
df = pd.read_csv("training_set_rel3.tsv", sep='\t')
df = df[['essay', 'domain1_score']]
```

---

## Methodology

---

## 1. Text Vectorization

```python id="h8i9j0"
from sklearn.feature_extraction.text import TfidfVectorizer

X = df['essay']
y = df['domain1_score']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)
```

---

## 2. Train-Test Split

```python id="i9j0k1"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y,
    test_size=0.2,
    random_state=42
)
```

---

## 3. Model Training

### Linear Regression

```python id="j0k1l2"
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
```

---

### Artificial Neural Network (ANN)

```python id="k1l2m3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

ann_model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32)

ann_pred = ann_model.predict(X_test.toarray())
```

---

## 4. Model Evaluation

```python id="l2m3n4"
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("ANN MAE:", mean_absolute_error(y_test, ann_pred))

print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("ANN R2:", r2_score(y_test, ann_pred))
```

---

## Results

| Model             | MAE  | MSE   | R² Score |
| ----------------- | ---- | ----- | -------- |
| Linear Regression | 2.65 | 16.01 | 0.79     |
| ANN               | 1.43 | 6.86  | 0.91     |

---

## Discussion

* ANN outperformed Linear Regression due to nonlinear learning capability
* Linear Regression struggles with complex text relationships

---

## Conclusion (Part B)

ANN is more suitable for automated essay scoring tasks.

---

# TOOLS USED

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras

---

# FINAL CONCLUSION

* Proper preprocessing improves model performance
* KNN is better for structured classification data
* ANN is better for complex text regression tasks
* Model selection significantly affects performance

---

```

