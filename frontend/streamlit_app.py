import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Guardian",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #FF0000;
        border-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model_loaded = joblib.load("models/logistic_model.pkl")
        scaler_loaded = joblib.load("models/scaler.pkl")
        return model_loaded, scaler_loaded
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_assets()

FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

with st.sidebar:
    st.title("🔮 Churn Guardian")
    st.markdown("### AI-Powered Retention Tool")
    st.info("This tool predicts customer churn probability based on tenure, financials, and service contracts.")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("👤 Customer Profile")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=10.0)

with col2:
    st.subheader("📄 Services & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    st.write(" **Add-on Services:**")
    c1, c2 = st.columns(2)
    with c1:
        fiber = st.checkbox("⚡ Fiber Optic")
        online_security = st.checkbox("🛡️ Online Security")
    with c2:
        tech_support = st.checkbox("🔧 Tech Support")
        no_internet = st.checkbox("🚫 No Internet")

contract_one = 1 if contract == "One year" else 0
contract_two = 1 if contract == "Two year" else 0

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract_One year": contract_one,
    "Contract_Two year": contract_two,
    "InternetService_Fiber optic": int(fiber),
    "InternetService_No": int(no_internet),
    "OnlineSecurity_Yes": int(online_security),
    "TechSupport_Yes": int(tech_support)
}

st.markdown("---")

if st.button("🚀 Analyze Churn Risk"):
    if model is None or scaler is None:
        st.error("Model not loaded. Check model files.")
    else:
        with st.spinner("Calculating..."):
            input_df = pd.DataFrame(0, columns=FEATURE_COLUMNS, index=[0])
            
            for key, value in input_data.items():
                if key in input_df.columns:
                    input_df[key] = value
            
            input_scaled = scaler.transform(input_df)
            churn_prob = model.predict_proba(input_scaled)[0][1]
            
            risk_pct = churn_prob * 100
            safe_pct = 100 - risk_pct
            
            st.markdown("### 🎯 Analysis Results")
            r_col1, r_col2, r_col3 = st.columns([1, 2, 1])

            with r_col1:
                if churn_prob > 0.5:
                    st.metric(label="Churn Probability", value=f"{risk_pct:.1f}%", delta="-High Risk", delta_color="inverse")
                else:
                    st.metric(label="Safety Score", value=f"{safe_pct:.1f}%", delta="+Safe", delta_color="normal")

            with r_col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = ['#ff4b4b', '#2ecc71'] 
                wedges, texts, autotexts = ax.pie(
                    [risk_pct, safe_pct], labels=["Churn", "Safe"],
                    autopct="%1.1f%%", startangle=90, colors=colors,
                    wedgeprops=dict(width=0.4, edgecolor='w'), 
                    textprops={'fontsize': 10, 'color': '#333'}
                )
                plt.setp(autotexts, size=10, weight="bold", color="white")
                ax.axis("equal")  
                fig.patch.set_alpha(0) 
                st.pyplot(fig)

            with r_col3:
                if churn_prob > 0.5:
                    st.error("⚠️ **High Risk!**")
                    st.markdown("Customer is likely to leave.")
                else:
                    st.success("✅ **Safe**")
                    st.markdown("Customer is loyal.")

            st.write("Risk Meter:")
            if churn_prob > 0.5:
                st.progress(float(churn_prob), text="CRITICAL LEVEL")
            else:
                st.progress(float(churn_prob), text="STABLE LEVEL")
