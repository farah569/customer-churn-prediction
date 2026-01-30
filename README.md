
# Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts whether a customer is likely to churn.

# Features
- Trained ML model (Logistic Regression)
- Handled class imbalance via class_weight
- Evaluated using ROC-AUC & Precision-Recall
- Achieved AUC = 0.84 and Recall (Churn) = 0.78
- Tuned decision threshold based on business objective
- FastAPI backend for predictions
- Streamlit interactive frontend
- Real-time churn probability visualization


# Tech Stack
- Python
- Scikit-learn
- FastAPI
- Streamlit
- Pandas
- Matplotlib

# Demo Screenshot

![APP UI ](images/screenshot.png)


# How to Run Locally:
 1- start API : 

```bash
python -m uvicorn app.main:app --reload


 2- start frontend:
 ```bash
 python -m streamlit run frontend/streamlit_app.py


 3-API endpoint:
 {
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 800,
  "Contract_One year": 0,
  "Contract_Two year": 1,
  "InternetService_Fiber optic": 1,
  "InternetService_No": 0,
  "OnlineSecurity_Yes": 1,
  "TechSupport_Yes": 1
}
 
Built by Farah Omar 💙




