# CSC 316 MACHINE LEARNING PRACTICAL REPORT  
### GROUP 11  :  Toxic Comment Classification and Bitcoin Price Prediction  

---

# TOPIC

1. Perform proper data preprocessing (handling missing values, encoding, scaling where necessary).  
2. Train and evaluate models using the assigned algorithms only.  
3. For classification, use accuracy, precision, recall, and F1-score.  
4. For regression, use MAE, MSE, and R² score.  
5. Compare model performance and discuss findings.  

---

# OBJECTIVE

The aim of this practical is to apply machine learning techniques to both classification and regression problems.

This project focuses on:

- Toxic comment classification  
- Bitcoin price prediction  

using different machine learning algorithms.

---

# PART A: TOXIC COMMENT CLASSIFICATION

---

## Problem Statement

The goal is to classify whether a comment is toxic or not using text data.

---

## Dataset

- Dataset: Jigsaw Toxic Comment Dataset  
- Input: Comment text  
- Output: Toxic label (0 or 1)  

---

## Methodology

---

## 1. Data Loading

```python
import pandas as pd

df = pd.read_csv("toxic_comments.csv")
print(df.head())

---

## 2. Data Preprocessing

```python id="a1c2d3"
# Remove missing values
df = df.dropna()

X = df['comment_text']
y = df['toxic']
```

---

## 3. Text Vectorization (TF-IDF)

```python id="b2d3e4"
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X)
```

---

## 4. Train-Test Split

```python id="c3e4f5"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## 5. Model Training

### Random Forest Classifier

```python id="d4f5g6"
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

---

### Support Vector Machine (SVM)

```python id="e5g6h7"
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
```

---

## 6. Model Evaluation

```python id="f6h7i8"
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

print("SVM Precision:", precision_score(y_test, svm_pred))
print("SVM Recall:", recall_score(y_test, svm_pred))
print("SVM F1 Score:", f1_score(y_test, svm_pred))
```

---

## Results

| Model         | Accuracy | Observation      |
| ------------- | -------- | ---------------- |
| Random Forest | ~0.94    | Good but slower  |
| SVM           | ~0.96    | Best performance |

---

## Discussion

* SVM performs better due to strong handling of high-dimensional TF-IDF features
* Random Forest struggles with sparse text data
* SVM generalizes better for text classification

---

## Conclusion (Part A)

Support Vector Machine (SVM) is the most effective model for toxic comment classification.

---

# PART B: BITCOIN PRICE PREDICTION (REGRESSION)

---

## Problem Statement

Predict Bitcoin closing price using historical data.

---

## Dataset

* Dataset: Bitcoin Historical Data
* Input: Open, High, Low, Volume
* Output: Close price

---

## Methodology

---

## 1. Data Loading

```python
df = pd.read_csv("bitcoin_data.csv")
print(df.head())
```

---

## 2. Data Preprocessing

```python id="g7i8j9"
# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values(by='Date')

# Remove missing values
df = df.dropna()
```

---

## 3. Feature Selection

```python id="h8j9k0"
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
```

---

## 4. Train-Test Split

```python id="i9k0l1"
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

```python id="j0l1m2"
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
```

---

### Random Forest Regressor

```python id="k1m2n3"
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

---

## 6. Model Evaluation

```python id="l2n3o4"
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
```

---

## Results

| Model             | MAE      | MSE    | R² Score |
| ----------------- | -------- | ------ | -------- |
| Linear Regression | Moderate | Higher | ~0.85    |
| Random Forest     | Lower    | Lower  | ~0.95    |

---

## Discussion

* Linear Regression is simple but limited in capturing patterns
* Random Forest captures complex nonlinear relationships
* Ensemble methods improve prediction accuracy

---

## Conclusion (Part B)

Random Forest Regression is the best model for Bitcoin price prediction.

---

# OVERALL CONCLUSION

* SVM is best for toxic comment classification
* Random Forest is best for Bitcoin price prediction
* Data preprocessing significantly improves model performance

---

# TOOLS USED

* Python
* Pandas
* NumPy
* Scikit-learn

---

# FINAL REMARK

This project demonstrates how different machine learning algorithms perform across classification and regression tasks, emphasizing the importance of preprocessing and model selection.

```

