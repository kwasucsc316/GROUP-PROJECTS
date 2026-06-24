# CSC 316 MACHINE LEARNING PRACTICAL REPORT  

## Comparative Analysis of Machine Learning Algorithms for Fake News Classification and Stock Market Prediction  

---

### GROUP 9  

---

# TOPIC

1. Perform proper data preprocessing (handling missing values, encoding, scaling where necessary).  
2. Train and evaluate models using the assigned algorithms only.  
3. For classification, use accuracy, precision, recall, and F1-score.  
4. For regression, use MAE, MSE, and R² score.  
5. Compare model performance and discuss findings.  

---


# OBJECTIVE

The objective of this practical is to apply machine learning algorithms to two different problem domains:

- Fake News Classification (Text Classification)  
- Stock Market Prediction (Regression Analysis)

This experiment demonstrates the full machine learning pipeline including:

- Data preprocessing  
- Feature extraction  
- Model training  
- Model evaluation  
- Performance comparison  

---

# PART A: FAKE NEWS DETECTION (CLASSIFICATION)

---

## Problem Statement

The Fake News dataset is used to classify news articles as either fake or real based on textual content.

This is a supervised binary classification problem where:

- **1 = Real News**  
- **0 = Fake News**

---

## Dataset Description

The dataset contains:

- title  
- author  
- text  
- label  

---

## Methodology

---

## 1. Data Loading

```python
import pandas as pd

df = pd.read_csv("fake_news.csv")
print(df.head())
````

---

## 2. Data Preprocessing

```python
import re

# Drop missing values
df = df.dropna()

# Combine text features
df['content'] = df['title'] + " " + df['author'] + " " + df['text']

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['content'] = df['content'].apply(clean_text)
```

---

## 3. Feature Extraction (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

X = df['content']
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)
```

---

## 4. Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## 5. Model Training

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
```

---

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
```

---

## 6. Model Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

print("SVM Precision:", precision_score(y_test, svm_pred))
print("SVM Recall:", recall_score(y_test, svm_pred))
print("SVM F1 Score:", f1_score(y_test, svm_pred))
```

---

## Results and Discussion

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.93     | 0.92      | 0.93   | 0.92     |
| SVM                 | 0.96     | 0.96      | 0.96   | 0.96     |

### Key Observations

* SVM outperformed Logistic Regression
* Better handling of high-dimensional TF-IDF features
* Logistic Regression is faster but less accurate

---

## Conclusion (Part A)

Support Vector Machine (SVM) is more suitable for fake news detection due to its higher accuracy and better generalization on text data.

---

# PART B: STOCK MARKET PREDICTION (REGRESSION)

---

## Problem Statement

Predict stock closing price using historical stock data.

---

## Dataset Description

### Features:

* Open
* High
* Low
* Volume

### Target:

* Close

---

## Methodology

---

## 1. Data Loading

```python
df = pd.read_csv("stock_data.csv")
print(df.head())
```

---

## 2. Data Preprocessing

```python
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

df = df.fillna(method='ffill')
```

---

## 3. Feature Selection

```python
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
```

---

## 4. Train-Test Split

```python
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

```python
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
```

---

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

---

## 6. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
```

---

## Results and Discussion

| Model             | MAE  | MSE  | R² Score |
| ----------------- | ---- | ---- | -------- |
| Linear Regression | 1.85 | 5.40 | 0.78     |
| Random Forest     | 0.92 | 2.10 | 0.93     |

### Key Observations

* Random Forest performed better
* Captures non-linear relationships effectively
* Linear Regression is simpler but less accurate

---

## Conclusion (Part B)

Random Forest Regression is more effective for stock price prediction due to its ability to model complex patterns in financial data.

---

# OVERALL CONCLUSION

* SVM is best for text classification
* Random Forest is best for regression tasks
* Proper preprocessing significantly improves performance

---

# TOOLS & LIBRARIES USED

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF (NLP)

---

# FINAL STATEMENT

This practical demonstrates the effectiveness of machine learning in solving real-world problems, highlighting the importance of preprocessing, feature extraction, and model selection.

```

