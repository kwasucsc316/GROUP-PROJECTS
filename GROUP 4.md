# CSC 316: MACHINE LEARNING PRACTICAL REPORT  


# GROUP 4 :  COMPARISON OF MACHINE LEARNING ALGORITHMS FOR DIABETES PREDICTION AND INSURANCE COST PREDICTION  

---


# OBJECTIVE

The objective of this practical exercise is to apply machine learning algorithms to real-world datasets in order to solve both classification and regression problems.

This includes:

- Data preprocessing  
- Feature selection  
- Model training  
- Model evaluation  
- Performance comparison  

---

# PART A: DIABETES PREDICTION (CLASSIFICATION)

---

## Problem Statement

The Pima Indians Diabetes dataset is used to predict whether a patient has diabetes based on medical attributes such as:

- Glucose level  
- Blood pressure  
- BMI  
- Age  

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

---

## Methodology

---

## 1. Data Loading

```python
import pandas as pd

df = pd.read_csv("diabetes.csv")
df.head()
````

---

## 2. Data Cleaning

```python id="d1a2b3"
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for col in cols:
    df[col] = df[col].replace(0, df[col].median())
```

---

## 3. Feature Selection

```python id="c2b3a4"
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

---

## 4. Train-Test Split

```python id="e3f4g5"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## 5. Feature Scaling

```python id="f4g5h6"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Model Training

### K-Nearest Neighbors (KNN)

```python id="g5h6i7"
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
```

---

### Support Vector Machine (SVM)

```python id="h6i7j8"
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
```

---

## 7. Model Evaluation

```python id="i7j8k9"
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

knn_results = evaluate(y_test, knn_pred)
svm_results = evaluate(y_test, svm_pred)

print("KNN:", knn_results)
print("SVM:", svm_results)
```

---

## Results and Discussion

| Model | Accuracy | Precision | Recall | F1 Score |
| ----- | -------- | --------- | ------ | -------- |
| KNN   | 0.753    | 0.655     | 0.655  | 0.655    |
| SVM   | 0.766    | 0.686     | 0.636  | 0.660    |

### Observations

* SVM achieved higher accuracy and precision
* KNN is sensitive to scaling and noise
* SVM provides a more stable decision boundary

---

## Conclusion (Part A)

Both models performed well, but **SVM is slightly more effective** due to its ability to find an optimal decision boundary.

---

# PART B: INSURANCE COST PREDICTION (REGRESSION)

---

## Problem Statement

The insurance dataset is used to predict medical insurance charges based on personal and lifestyle attributes.

This is a regression problem with a continuous target variable.

---

## Dataset Description

Features include:

* Age
* Sex
* BMI
* Children
* Smoker
* Region
* Charges (Target)

---

## Methodology

---

## 1. Data Loading

```python
df = pd.read_csv("insurance.csv")
df.head()
```

---

## 2. Data Preprocessing

```python id="j8k9l0"
df = pd.get_dummies(df, drop_first=True)
```

---

## 3. Feature Selection

```python id="k9l0m1"
X = df.drop("charges", axis=1)
y = df["charges"]
```

---

## 4. Train-Test Split

```python id="l0m1n2"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## 5. Model Training

### Linear Regression

```python id="m1n2o3"
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
```

---

### Decision Tree Regressor

```python id="n2o3p4"
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
```

---

## 6. Model Evaluation

```python id="o3p4q5"
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def reg_eval(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred),
        r2_score(y_true, y_pred)
    )

lr_results = reg_eval(y_test, lr_pred)
dt_results = reg_eval(y_test, dt_pred)

print("Linear Regression:", lr_results)
print("Decision Tree:", dt_results)
```

---

## Results and Discussion

| Model             | MAE     | MSE         | R² Score |
| ----------------- | ------- | ----------- | -------- |
| Linear Regression | 4181.19 | 33596915.85 | 0.784    |
| Decision Tree     | 2930.77 | 25831862.60 | 0.834    |

### Observations

* Decision Tree performs better for nonlinear relationships
* Linear Regression is simpler and more interpretable
* Smoking and lifestyle effects are better captured by trees

---

## Conclusion (Part B)

Decision Tree Regression is more effective due to its ability to capture complex nonlinear relationships.

---

# GENERAL CONCLUSION

* Proper preprocessing improves model performance
* Model choice depends on dataset type
* SVM performs well for classification tasks
* Decision Trees perform well for nonlinear regression problems
* Machine learning effectiveness depends on selecting the right model

---


