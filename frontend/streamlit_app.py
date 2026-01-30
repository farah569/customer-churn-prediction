
# import streamlit as st
# import requests
# import matplotlib.pyplot as plt


# st.set_page_config(
#     page_title="Customer Churn Prediction",
#     layout="centered"
# )

# st.title(" Customer Churn Prediction System")
# st.markdown("Predict whether a customer is likely to leave the company.")

# st.divider()

# st.subheader(" Customer Information")

# tenure = st.number_input("Tenure (months)", min_value=0, value=12)
# monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
# total = st.number_input("Total Charges", min_value=0.0, value=800.0)

# st.subheader(" Contract & Services")

# contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
# fiber = st.checkbox("Fiber Optic Internet")
# no_internet = st.checkbox("No Internet Service")
# online_security = st.checkbox("Online Security")
# tech_support = st.checkbox("Tech Support")

# contract_one = 1 if contract == "One year" else 0
# contract_two = 1 if contract == "Two year" else 0

# data = {
#     "tenure": tenure,
#     "MonthlyCharges": monthly,
#     "TotalCharges": total,
#     "Contract_One year": contract_one,
#     "Contract_Two year": contract_two,
#     "InternetService_Fiber optic": int(fiber),
#     "InternetService_No": int(no_internet),
#     "OnlineSecurity_Yes": int(online_security),
#     "TechSupport_Yes": int(tech_support)
# }

# if st.button(" Predict Churn"):
#     try:
#         response = requests.post(
#             "http://127.0.0.1:8000/predict",
#             json=data
#         )

#         result = response.json()

#         prob = result["churn_probability"]
#         label = result["prediction_label"]

#         st.divider()
#         st.subheader(" Prediction Result")
#         st.markdown("Churn Risk Analysis")

#         risk = prob * 100
#         safe = 100 - risk

#         fig, ax = plt.subplots()
#         ax.pie(
#              [risk, safe],
#              labels=["Churn Risk", "Safe"],
#              autopct="%1.1f%%",
#              startangle=90
# )
#         ax.axis("equal")

#         st.pyplot(fig)
#         plt.close()


#         st.progress(prob)

#         if label == "Churn":

#           st.error(f" High risk of churn: {risk:.1f}%")
#           st.markdown("⚠️ Customer shows strong churn signals. Consider retention offers.")
#         else:
        
#           st.success(f" Low churn risk: {risk:.1f}%")
#           st.markdown("✅ Customer appears loyal and stable.")

        
#     except Exception as e:
#         st.error("API not running. Start FastAPI first!")

import streamlit as st
import requests
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

with st.sidebar:
    st.title("🔮 Churn Guardian")
    st.markdown("### AI-Powered Retention Tool")
    st.info(
        "This tool predicts customer churn probability based on tenure, "
        "financials, and service contracts."
    )
    st.image("https://cdn-icons-png.flaticon.com/512/4144/4144517.png", width=100)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("👤 Customer Profile")
    tenure = st.slider("Tenure (Months)", 0, 72, 12, help="How long they have been a customer")
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

data = {
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
    
    with st.spinner("Consulting the AI Model..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=data)
            result = response.json()

            prob = result["churn_probability"]
            prediction_label = result["prediction_label"] 
            risk_pct = prob * 100
            safe_pct = 100 - risk_pct

            st.markdown("### 🎯 Analysis Results")
            
            r_col1, r_col2, r_col3 = st.columns([1, 2, 1])

            with r_col1:
                if prob > 0.5:
                    st.metric(label="Churn Probability", value=f"{risk_pct:.1f}%", delta="-High Risk", delta_color="inverse")
                else:
                    st.metric(label="Safety Score", value=f"{safe_pct:.1f}%", delta="+Safe", delta_color="normal")

            with r_col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                
                colors = ['#ff4b4b', '#2ecc71'] 
                
                wedges, texts, autotexts = ax.pie(
                    [risk_pct, safe_pct],
                    labels=["Churn", "Safe"],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(width=0.4, edgecolor='w'), 
                    textprops={'fontsize': 10, 'color': '#333'}
                )
                
                plt.setp(autotexts, size=10, weight="bold", color="white")
                ax.axis("equal")  
                
                fig.patch.set_alpha(0) 
                
                st.pyplot(fig)

            with r_col3:
                if prob > 0.5:
                    st.error("⚠️ **High Risk!**")
                    st.markdown("Customer is likely to leave.")
                    st.markdown("**Recommendation:** Offer a discount or long-term contract.")
                else:
                    st.success("✅ **Safe**")
                    st.markdown("Customer is loyal.")
                    st.markdown("**Recommendation:** Upsell new features.")

            st.write("Risk Meter:")
            if prob > 0.5:
                st.progress(prob, text="CRITICAL LEVEL")
            else:
                st.progress(prob, text="STABLE LEVEL")

        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Is the FastAPI server running on port 8000?")
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")