# Heart-Disease-Prediction
This project predicts heart disease using machine learning based on medical features such as age, blood pressure, cholesterol, chest pain type, and heart rate. It includes data preprocessing, EDA, model training, and evaluation to identify the most accurate model for early disease detection.
❤️ Heart Disease Prediction using Machine Learning
📌 Project Overview

This project focuses on developing a machine learning model to predict the likelihood of heart disease in patients based on various clinical and demographic features. The goal is to assist healthcare professionals in early diagnosis and intervention.

🎯 Objectives

Perform thorough data preprocessing, including handling missing values and outliers.

Conduct Exploratory Data Analysis (EDA) to identify key features.

Apply feature engineering and scaling techniques.

Train multiple classification models (e.g., Logistic Regression, Random Forest, SVM, XGBoost).

Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

Deploy the best-performing model for practical use in medical settings.

📂 Project Structure
Heart-Disease-Prediction/
│
├── data/
│   └── heart_disease_data.csv
│
├── notebooks/
│   └── heart_disease_prediction.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   └── best_model.pkl
│
├── images/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
└── README.md

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-Learn, XGBoost

Matplotlib, Seaborn

Jupyter Notebook

📊 Dataset Description

The dataset includes features like:

Age

Sex

Chest pain type

Resting blood pressure

Serum cholesterol

Fasting blood sugar

Resting ECG

Maximum heart rate

Exercise-induced angina

ST depression

Slope of the peak exercise ST segment

Number of major vessels

Thalassemia

🔍 Data Preprocessing

Handling missing values and outliers

Encoding categorical variables

Feature scaling (StandardScaler or MinMaxScaler)

Splitting data into training and testing sets

📈 Exploratory Data Analysis (EDA)

Analyzing feature distributions

Correlation heatmaps

Identifying key risk factors for heart disease

Visualizing feature importance

🛠 Model Building and Evaluation

Multiple models were trained and evaluated:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

XGBoost

The best-performing model (typically Random Forest or XGBoost) was selected based on metrics such as accuracy, precision, recall, and ROC-AUC.

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/<your-username>/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction


Install required libraries:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook


Train the model:

python src/model_training.py


Evaluate and predict:

python src/evaluation.py

📝 Conclusion

This project demonstrates a comprehensive approach to predicting heart disease using machine learning. It highlights the importance of data preprocessing, feature engineering, and model evaluation in building accurate and reliable predictive models for healthcare applications.
