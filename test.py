import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load dataset
credited_card_df = pd.read_csv('creditcard.csv')

X = credited_card_df.drop(columns='Class', axis=1)
Y = credited_card_df['Class']

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, Y_train)

# Evaluation
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))
roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:,1])

# Streamlit UI
st.title("💳 Credit Card Fraud Detection System")

st.write("### Model Performance")
st.write(f"Train Accuracy: {train_acc:.4f}")
st.write(f"Test Accuracy: {test_acc:.4f}")
st.write(f"ROC-AUC Score: {roc_auc:.4f}")

cm = confusion_matrix(Y_test, model.predict(X_test))
st.write("Confusion Matrix:")
st.write(cm)

st.write("---")
st.write("### Predict New Transaction")

threshold = st.slider("Select Fraud Detection Threshold", 0.1, 0.9, 0.5)

input_df = st.text_input("Enter all feature values separated by commas:")

if st.button("Predict"):
    try:
        input_values = [float(x.strip()) for x in input_df.split(',') if x.strip() != ""]

        if len(input_values) != X.shape[1]:
            st.error(f"⚠️ Expected {X.shape[1]} features, but got {len(input_values)}")
        else:
            features = np.array(input_values).reshape(1, -1)
            features = scaler.transform(features)

            prob = model.predict_proba(features)[0][1]

            if prob > threshold:
                st.error(f"⚠️ Fraudulent Transaction (Fraud Probability: {prob:.4f})")
            else:
                st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.4f})")

    except ValueError:
        st.error("⚠️ Please enter only numeric values separated by commas.")
