# Energy Consumption Classification

## 📌 Project Overview

This project uses Machine Learning techniques to classify household energy consumption based on environmental and temperature-related features.

The dataset used in this project is `energydata_complete.csv`. The original `Appliances` column contains numerical energy consumption values. For this classification project, the values are divided into two classes: **Low Energy Consumption** and **High Energy Consumption**.

This project includes data preprocessing, target creation, class balancing using SMOTE, feature selection, feature scaling, train-test splitting, and classification model training.

---

## 🎯 Objective

The main objective of this project is to classify energy consumption into two categories:

* **0 → Low Energy Consumption**
* **1 → High Energy Consumption**

This is a **binary classification problem**.

---

## 📂 Dataset

The project uses the **Energy Consumption dataset**.

### Important Columns

| Column              | Description                      |
| ------------------- | -------------------------------- |
| `date`              | Date and time of measurement     |
| `Appliances`        | Energy consumption of appliances |
| `lights`            | Energy used by lights            |
| `T1`, `T2`, ...     | Temperature measurements         |
| `RH_1`, `RH_2`, ... | Humidity measurements            |

The dataset contains different indoor and outdoor temperature and humidity measurements that can be used to analyze energy consumption.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Target Creation
   ↓
SMOTE Class Balancing
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Machine Learning Classification
   ↓
Model Prediction
   ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df1 = pd.read_csv("energydata_complete.csv")
```

### 2. Target Creation

Since `Appliances` is originally a continuous numerical variable, it is converted into two classes using the median split:

```python
df1['target'] = pd.qcut(df1['Appliances'], 2, labels=[0, 1])
```

The resulting classes are:

```text
0 → Low Energy Consumption
1 → High Energy Consumption
```

### 3. Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) is used to balance the classes.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
x_smote, y_smote = smote.fit_resample(X, y)
```

### 4. Feature Selection

`SelectKBest` can be used to select the most important features for classification.

### 5. Feature Scaling

`StandardScaler` is used to bring numerical features to a similar scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 6. Train-Test Split

The dataset is divided into training and testing data.

* **80% → Training**
* **20% → Testing**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=40
)
```

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is used as a classification model to predict whether the energy consumption belongs to the low or high category.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
```

Other classification algorithms can also be used for comparison:

* Decision Tree Classifier
* Random Forest Classifier
* AdaBoost Classifier
* Gradient Boosting Classifier
* K-Nearest Neighbors

---

## 📊 Model Evaluation

The classification model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

Example:

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classificatio
```

