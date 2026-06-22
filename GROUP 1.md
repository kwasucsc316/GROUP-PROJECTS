# **DEPARTMENT OF COMPUTER SCIENCE**

## **Kwara State University (KWASU)**

### **CSC 316 — MACHINE LEARNING**

---

# **PROJECT TITLE:**

## **Comparative Evaluation of Classification and Regression Architectures**

**ASSIGNED GROUP:** Group 1
**EXECUTION ENVIRONMENT:** Jupyter Notebook (Anaconda Distribution)

---

# **1. Introduction & Executive Summary**

This technical report outlines the methodology, preprocessing considerations, evaluation frameworks, and performance results for the machine learning tasks assigned to **Group 1**.

The implementation is split into two foundational problems:

* A **binary classification task** for spam email detection
* A **continuous regression task** for house price prediction

Both modules were engineered and structured within a synchronized **Jupyter Notebook environment** using the Anaconda package ecosystem.

---

# **2. Part 1: Binary Classification (Spam & Ham Detection)**

---

## **2.1 Preprocessing & Feature Engineering**

The objective of this phase was to accurately separate spam emails from legitimate messages (“ham”). Since machine learning models cannot directly process raw text, the data underwent transformation into numerical form.

### **Text Processing Steps**

* **Tokenization & Normalization**

  * Converted all text to lowercase
  * Removed punctuation and irrelevant symbols

* **Stop-word Removal**

  * Eliminated common English words such as *“the”*, *“is”*, *“and”*
  * Improved signal-to-noise ratio in textual data

* **TF-IDF Vectorization**

  * Applied **Term Frequency–Inverse Document Frequency (TF-IDF)**
  * Converted text into weighted numerical feature vectors
  * Captured importance of words relative to the dataset

* **Data Splitting**

  * Dataset divided into:

    * **Training Set: 80%**
    * **Validation Set: 20%**

---

## **2.2 Algorithmic Configurations & Empirical Output**

Two classification models were implemented:

* **Logistic Regression**
* **K-Nearest Neighbors (KNN), k = 5**

---

### **Model Performance Comparison**

| **Classification Model**      | **Accuracy**   | **Precision**  | **Recall**     | **F1-Score**   |
| ----------------------------- | -------------- | -------------- | -------------- | -------------- |
| **Logistic Regression**       | 0.9815 (98.1%) | 0.9892 (98.9%) | 0.9713 (97.1%) | 0.9801 (98.0%) |
| **K-Nearest Neighbors (KNN)** | 0.6877 (68.7%) | 0.9647 (96.5%) | 0.3489 (34.9%) | 0.5125 (51.3%) |

---

## **2.3 Discussion of Classification Findings**

The performance difference between the two models is rooted in their underlying architecture.

### **Logistic Regression Performance**

Logistic Regression achieved **98.15% accuracy**, demonstrating strong performance in high-dimensional text spaces.

* Works effectively with **sparse TF-IDF matrices**
* Learns linear decision boundaries
* Assigns optimized weights to important tokens (e.g., spam indicators)

---

### **KNN Performance Issues**

KNN achieved only **68.77% accuracy**, with a critically low **Recall of 34.89%**.

This failure is explained by the:

> **Curse of Dimensionality**

In high-dimensional TF-IDF space:

* Distance metrics become less meaningful
* All points appear nearly equidistant
* KNN struggles to form meaningful neighborhoods
* Leads to severe misclassification of spam emails

---

# **3. Part 2: Continuous Regression (House Price Prediction)**

---

## **3.1 Preprocessing & Feature Pipeline**

The regression task used the **Ames Housing Dataset**, which contains structured numerical and categorical features.

### **Data Cleaning Steps**

* **Missing Value Handling**

  * Numerical columns → filled using **median values**
  * Categorical columns → filled with `"None"`

* **Categorical Encoding**

  * Applied **One-Hot Encoding**
  * Converted categorical variables into binary feature columns

* **Feature Scaling**

  * Applied **StandardScaler**
  * Normalized numerical features to:

    * Mean = 0
    * Standard deviation = 1

---

## **3.2 Algorithmic Configurations & Evaluation**

Two regression models were evaluated:

* **Linear Regression**
* **Decision Tree Regressor (max depth = 5)**

---

### **Model Performance Comparison**

| **Regression Model**        | **MAE ($)** | **MSE ($²)** | **R² Score** | **Performance Tier** |
| --------------------------- | ----------- | ------------ | ------------ | -------------------- |
| **Linear Regression**       | 23,929.45   | 6.9037 × 10⁹ | 0.0999       | Deficient            |
| **Decision Tree Regressor** | 27,511.28   | 1.5516 × 10⁹ | 0.7977       | Robust               |

---

## **3.3 Discussion of Regression Findings**

The regression results reveal a clear contrast between model capabilities.

---

### **Decision Tree Regressor Performance**

The Decision Tree achieved an **R² score of 0.7977 (~80%)**, making it the best-performing model in this task.

* Captures **non-linear relationships**
* Splits data using hierarchical decision rules
* Handles mixed feature interactions effectively

---

### **Linear Regression Limitations**

Linear Regression performed poorly with an **R² score of 0.0999**.

Reasons include:

* Inability to model non-linear relationships
* High multicollinearity from One-Hot Encoding
* Over-simplification of complex housing patterns
* High-dimensional feature space distortion

---

# **4. Conclusion & Group Synthesis**

This project demonstrates that **model selection must align with data structure and complexity**.

### Key Findings:

* **Logistic Regression** performs best for:

  * High-dimensional sparse text data
  * Spam detection systems

* **KNN** performs poorly in:

  * High-dimensional feature spaces
  * Sparse vector environments

* **Decision Trees** are best suited for:

  * Mixed-type tabular datasets
  * Non-linear relationships

* **Linear Regression** is effective only when:

  * Relationships are approximately linear
  * Feature space is well-conditioned

---

### **Final Recommendation**

* Use **Logistic Regression** for text classification systems
* Use **Decision Tree-based models** for structured regression tasks

---

## **Overall Insight**

Machine learning performance is not universal—it is **data-dependent**. The correct pairing of algorithm and dataset structure is the key determinant of success.
