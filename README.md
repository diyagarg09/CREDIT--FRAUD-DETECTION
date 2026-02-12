#  Credit Card Fraud Detection System
A Machine Learning based web application that detects whether a credit card transaction is Legitimate or Fraudulent using Logistic Regression.

# Project Overview
Credit card fraud is a major financial issue worldwide.  
This project uses Machine Learning to identify fraudulent transactions based on transaction features.
The model is trained on the famous Kaggle Credit Card Fraud dataset.

# Dataset Information
- Source: Kaggle Credit Card Fraud Dataset
- Total Transactions: 284,807
- Fraud Cases: 492
- Highly Imbalanced Dataset

🔗 Dataset Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

⚠️ Note: Dataset is not included in this repository due to large file size.  
Please download it from Kaggle and place `creditcard.csv` in the project folder.

# Machine Learning Model Used
- Logistic Regression
- Class Weight Balancing
- StandardScaler for Feature Scaling
- ROC-AUC Evaluation
- 
# Model Performance
- Accuracy: ~93.8%
- Recall (Fraud Detection Rate): ~88%
- ROC-AUC Score: ~0.938 (approx)

Evaluation Metrics Used:
- Confusion Matrix
- ROC Curve

# Visualizations Included
- Confusion Matrix
- ROC Curve

 Credit-Fraud-Detection/
│
├── app.py                # Streamlit Web App
├── creditcard.ipynb      # Model training & graphs
├── test.py               # Model testing script (optional)
├── requirements.txt      # Dependencies
├── README.md             # Documentation

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

pip install -r requirements.txt

### 2️⃣ Run Jupyter Notebook

jupyter notebook

### 3️⃣ Run Streamlit App

streamlit run app.py

