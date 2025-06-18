# Machine-Learning-Predictions-Housing-Prices-Titanic-Survival-D-atasets

# COMP1816 Machine Learning Coursework

## Author
**Iustin-Andrei Moisa-Tudor**  
Student ID: 001228763

## Overview
This project covers two main machine learning tasks:  
- **Regression** using the California Housing dataset  
- **Classification** using the Titanic dataset

Both tasks involved data preprocessing, feature engineering, model training, hyperparameter tuning, and performance evaluation.

---

## 📊 Part 1: Regression - California Housing Dataset

### Objective
Predict median house values using:
- **Main Model**: Ridge Regression
- **Baselines**: Linear Regression, Lasso Regression

### Key Steps
- Preprocessed data: handled missing values, encoded categorical features, and created new features like `roomsHousehold`, `population_House`, etc.
- Normalized using `StandardScaler`
- Used `GridSearchCV` for hyperparameter tuning
- Evaluated using **MSE** and **R² score**

### Best Performance
- **Model**: Ridge Regression  
- **R² Score**: 0.6077  
- **MSE**: 5.04 × 10⁹

---

## 🤖 Part 2: Classification - Titanic Dataset

### Objective
Predict survival outcomes of passengers using:
- **Main Model**: Decision Tree Classifier
- **Baselines**: SVM, Neural Network

### Key Steps
- Imputed missing values, encoded categorical variables, and engineered features like `FamilySize`, `FarePerPerson`
- Normalized using `StandardScaler`
- Used `GridSearchCV` for model tuning
- Evaluated using **accuracy**, **precision**, **recall**, and **F1-score**

### Best Performance
- **Model**: Decision Tree Classifier  
- **Accuracy**: 87.14%  
- **Balanced precision/recall** for both survivor classes

---

## 📁 Project Structure
- `COMP1816_Housing_Dataset_Regression.csv` – Regression dataset
- `COMP1816_Titanic_Dataset_Classification.csv` – Classification dataset
- `regression_model.py` – Scripts for training and evaluating regression models
- `classification_model.py` – Scripts for classification models
- `report.pdf` – Full coursework report detailing methods, results, and analysis

---

## ⚙️ Technologies
- Python (NumPy, Pandas, Matplotlib)
- scikit-learn (GridSearchCV, regression/classification models)
- TensorFlow/Keras (Neural Network model)
- Jupyter Notebook (EDA and experimentation)

---

## 📌 Notes
- Final results suggest that **Ridge Regression** is best for regression tasks with multicollinearity.
- **Decision Tree Classifier** excelled at handling mixed data and capturing non-linear patterns in classification.
- Further improvement ideas: use ensemble methods (e.g. Random Forest, XGBoost), and explore deeper neural networks.

---
