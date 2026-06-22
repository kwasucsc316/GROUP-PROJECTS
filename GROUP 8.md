
# **Task 1 Report**

## **Spam Detection Using KNN and ANN**

**Dataset:** Spam and Ham Classification Balanced Dataset (Kaggle)

---

# **Background and Definitions**

Classification is a type of machine learning task where the goal is to predict which category an input belongs to. The output is always one of a fixed set of labels.

In this task, the categories are:

* **Spam** → unwanted messages
* **Ham** → legitimate messages

This is a **supervised learning** problem, meaning the model is trained on labelled data where the correct answers are already known.

---

## **Key Terms**

* **Training Set:** Portion of data used to train the model (80%)
* **Test Set:** Unseen data used for evaluation (20%)
* **TF-IDF:** Converts text into numerical values based on word importance so ML models can process it

---

# **Methodology**

The dataset contained **9,989 messages** labelled as spam or ham, with no missing values.

---

## **Preprocessing**

* Labels converted using **Label Encoding**

  * Ham = 0
  * Spam = 1

* Text converted into numerical features using **TF-IDF Vectorization**

  * Top **5,000 words selected**
  * English stop words removed

---

## **Data Splitting**

* Training Set: **80% (7,991 messages)**
* Test Set: **20% (1,998 messages)**
* Random State: **42** (for reproducibility)

---

## **Algorithms Used**

### **K-Nearest Neighbors (KNN)**

* Uses **k = 5 nearest neighbors**
* Classifies messages based on majority vote
* Does not learn patterns; instead memorizes training data

---

### **Artificial Neural Network (ANN)**

Built using `MLPClassifier`:

* Hidden Layers: **128 neurons + 64 neurons**
* Activation Function: **ReLU**
* Optimizer: **Adam**
* Training Iterations: **20**

ANN learns patterns by adjusting internal weights over time.

---

## **Evaluation Metrics**

* **Accuracy:** Overall correctness of predictions
* **Precision:** Correct spam predictions out of all predicted spam
* **Recall:** Correctly identified spam out of all actual spam
* **F1-Score:** Balance between precision and recall

---

# **Results**

| **Metric** | **KNN** | **ANN** |
| ---------- | ------- | ------- |
| Accuracy   | 0.7648  | 0.9745  |
| Precision  | 0.9662  | 0.9674  |
| Recall     | 0.5176  | 0.9787  |
| F1-Score   | 0.6741  | 0.9730  |

---

# **Discussion**

The ANN significantly outperformed KNN across all evaluation metrics.

### **KNN Performance**

* Accuracy: **76%**
* Recall: **0.52**

KNN failed to detect nearly half of all spam messages. This is a major weakness because spam detection requires high recall.

The main issue is the **curse of dimensionality**:

* TF-IDF creates **5,000-dimensional feature space**
* Distance calculations become unreliable
* All messages appear similarly distant

---

### **ANN Performance**

* Accuracy: **97%**
* Recall: **0.98**
* F1-Score: **0.97**

ANN performed exceptionally well because:

* It learns complex patterns in text
* Uses hidden layers to extract meaningful relationships
* Handles high-dimensional data efficiently

---

# **Conclusion**

The ANN is clearly the superior model for spam detection.

* Best performance in **recall (most important metric)**
* Strong overall accuracy and balance
* Better suited for high-dimensional text data

KNN remains simple and fast but is not suitable for this type of problem.

---

# **Task 2 Report**

## **Weather Temperature Prediction Using Decision Tree and Random Forest**

**Dataset:** Daily Temperature of Major Cities (Kaggle)

---

# **Background and Definitions**

Regression is a machine learning task where the goal is to predict a **continuous numerical value**.

In this case, the model predicts:

> **Average daily temperature**

---

## **Key Terms**

* **Training Set:** 80% of dataset
* **Test Set:** 20% of dataset
* **Label Encoding:** Converts categorical values (city, region) into numbers
* **Overfitting:** When a model learns training data too well and performs poorly on new data

---

# **Methodology**

The dataset contained over **2.9 million records**, reduced to **1,450,555 usable entries** after cleaning.

---

## **Preprocessing**

* Removed invalid values (**-99 placeholders**)

* Dropped missing values

* Encoded:

  * Region
  * Country
  * City

* Input Features:

  * Month
  * Day
  * Year
  * Encoded location data

* Target:

  * **AvgTemperature**

---

## **Data Splitting**

* Training Set: **80%**
* Test Set: **20%**
* Random State: **42**

---

## **Algorithms Used**

### **Decision Tree**

* Splits data using conditional rules
* Max depth = **10**
* Prevents overfitting

---

### **Random Forest**

* Ensemble of **100 decision trees**
* Averages predictions for stability
* Reduces overfitting

---

## **Evaluation Metrics**

* **MAE:** Average prediction error
* **MSE:** Penalizes large errors
* **R² Score:** Measures how well model explains variance (0–1)

---

# **Results**

| **Metric** | **Decision Tree** | **Random Forest** |
| ---------- | ----------------- | ----------------- |
| MAE        | 8.1632            | 7.9798            |
| MSE        | 110.7645          | 104.9513          |
| R² Score   | 0.6944            | 0.7104            |

---

## **Feature Importance (Random Forest)**

* **Month:** 0.80 (dominant factor)
* **City:** 0.10 – 0.20
* **Day:** 0.00 – 0.10

---

# **Discussion**

Random Forest performed slightly better than Decision Tree.

* Lower MAE and MSE
* Higher R² score (0.71)

However, improvement is small because:

### **Key Insight**

* **Month dominates prediction (80%)**
* Other features contribute very little

This means the dataset is structurally simple.

---

## **Interpretation**

* Temperature is mainly driven by **seasonality**
* Location has minor influence
* Day variation is minimal

The model cannot improve much because the dataset itself has limited complexity.

---

# **Conclusion**

Random Forest is the better model for temperature prediction, but only slightly.

* Both models perform similarly
* Random Forest is more stable
* Limited improvement due to dominant feature (Month)

---

# **Final Insight**

* **Task 1 (Spam Detection):** ANN clearly outperforms KNN due to complex text patterns
* **Task 2 (Temperature Prediction):** Random Forest only slightly outperforms Decision Tree due to simple seasonal structure

This shows that **model performance depends heavily on data complexity and feature structure**.
