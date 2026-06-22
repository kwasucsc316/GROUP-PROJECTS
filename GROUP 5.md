# ASSIGNED DATASETS AND ALGORITHMS

* **Classification (Titanic Survival):** Random Forest, Artificial Neural Network (ANN)
* **Regression (Student Performance):** Linear Regression, Random Forest

---

# PART A: TITANIC SURVIVAL PREDICTION (CLASSIFICATION)

## Titanic Survival Prediction Using Random Forest and Artificial Neural Network (ANN) Classification Algorithms

## INTRODUCTION

The Titanic disaster remains one of the most widely studied datasets in machine learning education. The dataset provides demographic and travel information about passengers aboard the RMS Titanic and is commonly used to demonstrate classification techniques.

In this practical, machine learning algorithms were developed to predict whether a passenger survived or not. Random Forest and Artificial Neural Network (ANN) models were trained and evaluated using standard classification metrics.

## OBJECTIVES

1. Load and preprocess the Titanic dataset.
2. Handle missing values and categorical variables.
3. Train Random Forest and ANN classification models.
4. Evaluate the models using Accuracy, Precision, Recall, and F1-Score.
5. Compare both models and determine the better-performing algorithm.

## DATASET DESCRIPTION

The Titanic dataset contains passenger-related information such as:

* PassengerId
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

The target variable is **Survived**, where:

* **1** = Survival
* **0** = Non-survival

## METHODOLOGY

### Data Preprocessing

* Missing **Age** values were replaced using the median age (**28.0**).
* Missing **Embarked** values were filled using the mode.
* **Cabin**, **Ticket**, and **Name** columns were removed.
* **Sex** and **Embarked** were encoded using Label Encoding.

### Dataset Splitting

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

using `train_test_split()`.

### Model Training

#### Random Forest Classifier

An ensemble learning algorithm consisting of **100 decision trees**.

#### Artificial Neural Network (ANN)

Implemented using `MLPClassifier` with:

* One hidden layer
* 100 neurons
* Maximum iterations = 1000

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

## RESULTS

### Random Forest Results

| Metric    | Value    |
| --------- | -------- |
| Accuracy  | 0.821229 |
| Precision | 0.808824 |
| Recall    | 0.743243 |
| F1-Score  | 0.774648 |

### ANN Results

| Metric    | Value    |
| --------- | -------- |
| Accuracy  | 0.715084 |
| Precision | 0.629213 |
| Recall    | 0.756757 |
| F1-Score  | 0.687117 |

## COMPARISON TABLE

| Model         | Accuracy | Precision | Recall   | F1-Score |
| ------------- | -------- | --------- | -------- | -------- |
| Random Forest | 0.821229 | 0.808824  | 0.743243 | 0.774648 |
| ANN           | 0.715084 | 0.629213  | 0.756757 | 0.687117 |

## DISCUSSION

The Random Forest model achieved the highest overall performance across most evaluation metrics. Its ensemble approach enabled it to effectively capture feature interactions and reduce overfitting.

The ANN demonstrated a slightly higher Recall value but lower Precision and Accuracy. This may be attributed to the relatively small size of the dataset and limited hyperparameter tuning.

Overall, Random Forest proved to be the more reliable model for this structured classification problem.

## CODE IMPLEMENTATION

### Step 1: Import Required Libraries

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

### Step 2: Load Dataset

```python
df = pd.read_csv("titanic.csv")
```

### Step 3: Data Cleaning and Preprocessing

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = df.drop(["Cabin", "Ticket", "Name"], axis=1)
```

### Step 4: Feature Selection and Splitting

```python
X = df.drop("Survived", axis=1)
Y = df["Survived"]
```

### Step 5: Train Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Step 6: Train ANN

```python
ann = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=1000,
    random_state=42
)
```

## CONCLUSION

Random Forest achieved superior performance with approximately:

* **82.1% Accuracy**
* **77.5% F1-Score**

Therefore, it is the preferred model for Titanic survival prediction in this practical.

---

# PART B: STUDENT PERFORMANCE PREDICTION (REGRESSION)

## Student Mathematics Score Prediction Using Linear Regression and Random Forest Regression

## INTRODUCTION

Predicting student academic performance is a common application of machine learning in education analytics.

In this practical, Linear Regression and Random Forest Regression models were applied to predict students' mathematics scores using demographic and examination-related features.

## OBJECTIVES

1. Load and preprocess the Student Performance dataset.
2. Encode categorical variables.
3. Train regression models.
4. Evaluate using MAE, MSE, and R² Score.
5. Compare the effectiveness of both algorithms.

## DATASET DESCRIPTION

The dataset contains features such as:

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch Type
* Test Preparation Course
* Reading Score
* Writing Score

The target variable is **Math Score**.

## METHODOLOGY

### Data Preprocessing

* Dataset loaded using Pandas.
* One-hot encoding applied using `get_dummies()`.
* Math Score selected as the target variable.

### Dataset Splitting

* **70% Training Data**
* **30% Testing Data**

### Model Training

#### Linear Regression

A baseline algorithm that models relationships using a linear equation.

#### Random Forest Regression

An ensemble regression algorithm capable of capturing nonlinear relationships.

### Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

## RESULTS

### Linear Regression

| Metric   | Value     |
| -------- | --------- |
| MAE      | 4.174705  |
| MSE      | 27.149815 |
| R² Score | 0.870626  |

### Random Forest

| Metric   | Value     |
| -------- | --------- |
| MAE      | 6.683333  |
| MSE      | 73.210000 |
| R² Score | 0.651141  |

## COMPARISON TABLE

| Model             | MAE      | MSE       | R² Score |
| ----------------- | -------- | --------- | -------- |
| Linear Regression | 4.174705 | 27.149815 | 0.870626 |
| Random Forest     | 6.683333 | 73.210000 | 0.651141 |

## DISCUSSION

Linear Regression significantly outperformed Random Forest across all regression metrics.

The lower MAE and MSE values indicate smaller prediction errors, while the higher R² Score demonstrates a stronger ability to explain variation in mathematics scores.

These results suggest that the underlying relationships between features and the target variable are largely linear, making Linear Regression the most suitable model.

## CODE IMPLEMENTATION

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
```

### Step 2: Load Dataset

```python
df = pd.read_csv("exams.csv")
```

### Step 3: Preprocess Dataset

```python
df = pd.get_dummies(df, drop_first=True)
```

### Step 4: Train Linear Regression

```python
lr = LinearRegression()
```

### Step 5: Train Random Forest Regression

```python
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

## CONCLUSION

Linear Regression delivered the best performance with:

* **R² Score = 0.870626**
* Lower prediction errors

Therefore, it is the preferred model for predicting student mathematics scores.

---

# OVERALL CONCLUSION

This practical successfully implemented machine learning techniques for both classification and regression tasks.

* **Random Forest** was the best-performing model for Titanic Survival Prediction.
* **Linear Regression** achieved superior results for Student Performance Prediction.

The exercise demonstrated the importance of:

* Data preprocessing
* Feature engineering
* Model selection
* Performance evaluation

using appropriate machine learning metrics.

The practical also reinforced the relevance of:

* Python
* Pandas
* Scikit-learn

in solving real-world machine learning problems.
