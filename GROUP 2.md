**ASSIGNED DATASETS AND ALGORITHMS**
•	Classification (Fake News): SVM, Random Forest
•	Regression (Bitcoin Price): Linear Regression, KNN

**PART A: FAKE NEWS DETECTION (CLASSIFICATION)**
Fake News Detection Using Support Vector Machine (SVM) and Random Forest Classification Algorithms

**Introduction**
Fake news has become a major challenge in the digital age, spreading misinformation and influencing public opinion. Machine learning techniques can be used to automatically classify news articles as either fake or true based on their textual content. This practical focuses on developing and evaluating machine learning models for fake news detection using Support Vector Machine (SVM) and Random Forest algorithms.

**Objectives**
The objectives of this practical are to:
1.	Load and preprocess the Fake News dataset.
2.	Convert textual data into numerical representations suitable for machine learning.
3.	Train classification models using SVM and Random Forest algorithms.
4.	Evaluate the performance of the models using standard classification metrics.
5.	Compare the effectiveness of both algorithms in detecting fake news.

**Dataset Description**
The dataset consists of two files:
•	Fake.csv: Contains fake news articles.
•	True.csv: Contains genuine news articles.
Each dataset contains the following attributes:
•	Title
•	Text
•	Subject
•	Date
A new column called "label" was created where:
•	0 represents Fake News
•	1 represents True News
The two datasets were merged into a single dataset for model training and evaluation.

**Methodology**
a) Data Preprocessing
The following preprocessing steps were performed:
1.	Loaded Fake.csv and True.csv using Pandas.
2.	Added labels to distinguish fake and true news.
3.	Combined both datasets into a single DataFrame.
4.	Shuffled the dataset to ensure random distribution.
5.	Checked and handled missing values.
6.	Selected the text column as the input feature.
7.	Applied TF-IDF Vectorization to convert text data into numerical features.

b) Dataset Splitting
The dataset was divided into:
•	Training Set: 70%
•	Testing Set: 30%
The split was performed using the train_test_split() function from Scikit-learn.

c) Model Training
Two classification algorithms were implemented:
a) Support Vector Machine (SVM): SVM attempts to find the optimal boundary that separates fake news from true news based on the extracted textual features.
b) Random Forest: Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification accuracy and reduce overfitting.

**Evaluation Metrics**
The following performance metrics were used:
a) Accuracy: Measures the percentage of correctly classified instances.
b) Precision: Measures how many articles predicted as true are actually true.
c) Recall: Measures the ability of the model to identify all relevant instances.
d) F1-Score: Provides a balance between precision and recall.


**Results**
a) SVM Results
•	Accuracy: 0.990943
•	Precision: 0.988949
•	Recall: 0.992037
•	F1-Score: 0.990491

b) Random Forest Results  
•	Accuracy: 0.989087
•	Precision: 0.986474
•	Recall: 0. 990632 
•	F1-Score: 0.988549

**Comparison of Models**
**Model	         Accuracy		Precision		Recall		  F1-Score**
  SVM	           0.990943		0.988949 		0.992037		0.990491
  Random Forest	 0.989087		0.986474		0.990632		0.988549


**Discussion**
The results indicate the effectiveness of machine learning techniques in detecting fake news. The SVM model demonstrated strong performance due to its ability to handle high-dimensional text data effectively. The Random Forest model also performed well by leveraging multiple decision trees for classification. However, the SVM model achieved slightly better results across all evaluation metrics.
The SVM model recorded an Accuracy of 99.09%, Precision of 98.89%, Recall of 99.20%, and an F1-Score of 99.05%. These results demonstrate the model's strong ability to correctly identify both fake and genuine news articles while maintaining a very low rate of misclassification.
The Random Forest model also achieved excellent performance, with an Accuracy of 98.91%, Precision of 98.65%, Recall of 99.06%, and an F1-Score of 98.85%. Although its performance was slightly lower than that of SVM, the differences were relatively small, indicating that Random Forest is also highly effective for fake news detection.
Comparing both models, SVM outperformed Random Forest in Accuracy, Precision, Recall, and F1-Score. This suggests that SVM was better able to learn the complex patterns present in the textual data generated through TF-IDF vectorization. The high Recall values achieved by both models indicate that very few fake or true news articles were incorrectly classified.
Overall, the results show that machine learning techniques can be highly effective in detecting fake news, with SVM providing the best overall performance on this dataset.

**CODE IMPLEMENTATION**
Step 1: Import Required Libraries
# Import pandas for data handling
import pandas as pd
________________________________________
Step 2: Load the Datasets
# Load fake news dataset
fake = pd.read_csv("Fake.csv")

# Load true news dataset
true = pd.read_csv("True.csv")
________________________________________
Step 3: Inspect the Data
# Display first 5 rows of fake news dataset
print(fake.head())

# Display first 5 rows of true news dataset
print(true.head())

# Check dataset sizes
print(fake.shape)
print(true.shape)

# Check column names
print(fake.columns)
________________________________________
Step 4: Add Labels
# Label fake news as 0
fake["label"] = 0

# Label true news as 1
true["label"] = 1
Why?
•	0 = Fake News
•	1 = True News
The model learns from these labels.
________________________________________
Step 5: Merge the Datasets
# Combine fake and true news into one dataset
df = pd.concat([fake, true], axis=0)
________________________________________
Step 6: Shuffle the Dataset
# Randomly mix fake and true articles
df = df.sample(frac=1, random_state=42)

# Reset row numbering
df = df.reset_index(drop=True)
________________________________________
Step 7: Check for Missing Values
# Count missing values in each column
print(df.isnull().sum()) 
#No missing values in the dataset so no further action needed
________________________________________
Step 8: Select Features and Target
# Input feature (article text)
X = df["text"]

# Output label (fake or true)
y = df["label"]
________________________________________
Step 9: Convert Text into Numbers Using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert words into numerical features
vectorizer = TfidfVectorizer( stop_words="english", max_df=0.7)

# Transform text data
X = vectorizer.fit_transform(X)
Why?
Machine Learning models cannot understand text directly.
________________________________________
Step 10: Split Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split

# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3,
    random_state=42)
________________________________________
Step 11: Train SVM Model
from sklearn.svm import SVC

# Create SVM model
svm_model = SVC()

# Train model
svm_model.fit(X_train, y_train)
________________________________________
Step 12: Make Predictions Using SVM
# Predict labels for test data
svm_pred = svm_model.predict(X_test)
________________________________________
Step 13: Evaluate SVM Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("SVM Results")

print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Precision:", precision_score(y_test, svm_pred))
print("Recall:", recall_score(y_test, svm_pred))
print("F1 Score:", f1_score(y_test, svm_pred))
________________________________________
Step 14: Train Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)
________________________________________
Step 15: Make Predictions Using Random Forest
# Predict labels
rf_pred = rf_model.predict(X_test)
________________________________________
Step 16: Evaluate Random Forest Model
print("Random Forest Results")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
________________________________________
Step 17: Compare Both Models
results = pd.DataFrame({
    "Model": ["SVM", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        precision_score(y_test, svm_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, svm_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1 Score": [
        f1_score(y_test, svm_pred),
        f1_score(y_test, rf_pred)
    ]
})

print(results)


**Conclusion**
This practical successfully applied machine learning techniques to classify news articles as fake or true. The dataset was preprocessed, transformed into numerical features using TF-IDF vectorization, and used to train SVM and Random Forest classification models.
Based on the evaluation metrics obtained, the SVM model achieved the best overall performance with an Accuracy of 99.09% and an F1-Score of 99.05%, outperforming the Random Forest model. Although both models demonstrated excellent classification capabilities, SVM proved to be the more effective algorithm for this dataset.
The results confirm that machine learning models can accurately distinguish between fake and true news articles and can serve as valuable tools in combating the spread of misinformation.


**PART B: BITCOIN PRICE PREDICTION (REGRESSION)**
Bitcoin Price Prediction Using Linear Regression and K-Nearest Neighbors (KNN) Regression

**Introduction**
Bitcoin is one of the most widely traded cryptocurrencies in the world. Due to its highly dynamic market value, predicting Bitcoin prices has become an important application of machine learning. Regression algorithms can be used to analyze historical Bitcoin data and predict future prices based on relevant features.
This practical focuses on developing and evaluating machine learning regression models for Bitcoin price prediction using Linear Regression and K-Nearest Neighbors (KNN) Regression.

**Objectives**
The objectives of this practical are to:
1.	Load and preprocess the Bitcoin historical dataset.
2.	Select relevant features for price prediction.
3.	Train regression models using Linear Regression and KNN Regression.
4.	Evaluate the models using standard regression metrics.
5.	Compare the performance of both models and determine the most suitable algorithm for Bitcoin price prediction.

**Dataset Description**
The dataset used for this practical is the Bitcoin Historical Data dataset obtained from Kaggle.
The dataset contains historical Bitcoin trading information such as:
•	Timestamp/Date
•	Open Price
•	High Price
•	Low Price
•	Close Price
•	Volume
The target variable selected for prediction is the Close Price of Bitcoin.

**Methodology**
a) Data Preprocessing
The following preprocessing steps were carried out:
1.	Loaded the Bitcoin dataset using Pandas.
2.	Examined the dataset structure and data types.
3.	Checked for missing values and removed incomplete records where necessary.
4.	Selected relevant features such as Open, High, Low, and Volume.
5.	Selected Close Price as the target variable.
6.	Applied feature scaling using StandardScaler to normalize the data for KNN Regression.

b) Feature Selection
Input Features:
•	Open Price
•	High Price
•	Low Price
•	Volume
Target Variable:
•	Close Price

c) Dataset Splitting
The dataset was divided into:
•	Training Set: 70%
•	Testing Set: 30%
The train_test_split() function from Scikit-learn was used for this purpose.

d) Model Training
(i) Linear Regression: Linear Regression is a statistical algorithm that models the relationship between dependent and independent variables using a linear equation.
The model attempts to predict Bitcoin closing prices based on the selected input features.
(ii) K-Nearest Neighbors (KNN) Regression: KNN Regression predicts the value of a data point by considering the values of its nearest neighbors.
The algorithm uses similarity between observations to estimate the target value.

**Evaluation Metrics**
The following regression metrics were used to evaluate model performance:
a) Mean Absolute Error (MAE): Measures the average magnitude of prediction errors.
b) Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
c) R-Squared (R² Score): Measures how well the model explains the variation in the target variable. An R² value closer to 1 indicates better model performance.


**Results**
a) Linear Regression Results
•	MAE: 4.444420
•	MSE: 143.215646
•	R² Score: 1.000000

b) KNN Regression Results
•	MAE: 8.595085
•	MSE: 1503.697967
•	R² Score: 0.999998

**Comparison of Models**
**Model	               MAE		      MSE		   R² Score**
Linear Regression	   4.444420		143.215646 		1.000000		
KNN	                 8.595085		1503.697967		0.999998		

**Discussion**
The results show how effectively each algorithm predicted Bitcoin closing prices from historical market data. 
Linear Regression assumes a linear relationship between features and the target variable. It is computationally efficient and easy to interpret. KNN Regression predicts values based on nearby observations in the dataset. While it can capture local patterns, its performance may depend on the choice of K value and the quality of feature scaling.
The comparison of MAE, MSE, and R² Score was used to determine the better-performing model. A model with lower MAE and MSE values and a higher R² Score is generally considered superior. According to the results obtained from the regression models show that both Linear Regression and K-Nearest Neighbors (KNN) Regression were highly effective in predicting Bitcoin closing prices. However, Linear Regression demonstrated superior performance across all evaluation metrics.
Linear Regression achieved a Mean Absolute Error (MAE) of 4.44, a Mean Squared Error (MSE) of 143.22, and an R² Score of 1.000000. These values indicate that the model's predictions were extremely close to the actual Bitcoin prices, with very minimal prediction error. The R² Score of 1.000000 shows that the model was able to explain virtually all the variation in the target variable.
KNN Regression also produced strong results, with an MAE of 8.60, an MSE of 1503.70, and an R² Score of 0.999998. Although the R² Score was still very close to 1, the MAE and MSE values were significantly higher than those of Linear Regression, indicating larger prediction errors.
The comparison suggests that the relationship between the selected features (Open, High, Low, and Volume) and the Bitcoin closing price is largely linear. As a result, Linear Regression was able to model this relationship more effectively than KNN Regression. The lower error values obtained by Linear Regression indicate better prediction accuracy and greater reliability for this dataset.
Overall, both models performed exceptionally well, but Linear Regression provided the most accurate and consistent predictions.

**CODE IMPLEMENTATION**
Step 1: Import Libraries
import pandas as pd
________________________________________
Step 2: Load Dataset
# Load Bitcoin dataset
btc = pd.read_csv("btcusd.csv")
________________________________________
Step 3: Explore Dataset
# Display first 5 rows
print(btc.head())

# Check dataset information
print(btc.info())

# Check missing values
print(btc.isnull().sum())#No missing values in the dataset
________________________________________
Step 4: Select Features and Target
# Input features
X = btc[["Open", "High", "Low", "Volume"]]

# Target variable
y = btc["Close"]
Goal:
Predict Bitcoin Closing Price.
________________________________________
Step 5: Scale the Data
from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Scale features
X = scaler.fit_transform(X)
Why?
KNN performs better when features are scaled. Without scaing, KNN often performs poorly because distance calculations becomes biased toward larger-valued featues.
________________________________________
Step 6: Split Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
________________________________________
Step 7: Train Linear Regression Model
from sklearn.linear_model import LinearRegression

# Create model
lr = LinearRegression()

# Train model
lr.fit(X_train, y_train)
________________________________________
Step 8: Predict Using Linear Regression
# Predict Bitcoin prices
lr_pred = lr.predict(X_test)
________________________________________
Step 9: Evaluate Linear Regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Linear Regression Results")

print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R²:", r2_score(y_test, lr_pred))
________________________________________
Step 10: Train KNN Regression Model
from sklearn.neighbors import KNeighborsRegressor

# Create KNN model
knn = KNeighborsRegressor(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)
________________________________________
Step 11: Predict Using KNN
# Predict Bitcoin prices
knn_pred = knn.predict(X_test)
________________________________________
Step 12: Evaluate KNN Regression
print("KNN Regression Results")

print("MAE:", mean_absolute_error(y_test, knn_pred))
print("MSE:", mean_squared_error(y_test, knn_pred))
print("R²:", r2_score(y_test, knn_pred))
________________________________________
Step 13: Compare Both Models
results = pd.DataFrame({
    "Model": ["Linear Regression", "KNN Regression"],
    "MAE": [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, knn_pred)
    ],
    "MSE": [
        mean_squared_error(y_test, lr_pred),
        mean_squared_error(y_test, knn_pred)
    ],
    "R² Score": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, knn_pred)
    ]
})

print(results)
________________________________________

**Conclusion**
This practical successfully implemented Linear Regression and K-Nearest Neighbors Regression for Bitcoin price prediction using historical Bitcoin market data.
The evaluation results showed that Linear Regression outperformed KNN Regression, achieving the lowest prediction errors with an MAE of 4.44 and an MSE of 143.22, while also obtaining a perfect R² Score of 1.000000. Although KNN Regression also achieved excellent performance, its prediction errors were higher than those of Linear Regression.
Based on the results obtained, Linear Regression is the preferred model for predicting Bitcoin closing prices in this dataset. The findings demonstrate that machine learning techniques can be effectively applied to financial datasets to generate highly accurate price predictions.
Future work may involve testing additional regression algorithms, performing hyperparameter tuning, and incorporating more market indicators to further improve prediction performance.

**Tools Used**
•	Python
•	Jupyter Notebook
•	Pandas
•	NumPy
•	Scikit-learn
•	TF-IDF Vectorizer
•	StandardScaler


**References**
1.	Scikit-learn Documentation.
2.	Kaggle Fake News Detection Dataset.
3.	Kaggle Bitcoin Historical Data Dataset.
4.	Python Documentation

