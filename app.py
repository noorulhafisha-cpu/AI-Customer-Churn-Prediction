import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load the trained model and feature names
model = joblib.load("random_forest_churn_model.pkl")
features = joblib.load("churn_features.pkl")

st.title("AI Customer Churn Prediction")
st.write("Predict customer churn and identify customers who may need support.")

st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

phone = st.selectbox("Phone Service", ["Yes", "No"])

multiple = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

# Prediction button
if st.button("Predict Churn", type="primary"):

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    # Create input with exactly the training feature columns
    input_df = pd.DataFrame(
        [[0] * len(features)],
        columns=features
    )

    # Fill numeric and categorical features consistently
    for column, value in customer.items():

        if column in input_df.columns:
            input_df.loc[0, column] = value

        for feature in features:
            prefix = column + "_"

            if feature.startswith(prefix):
                category = feature[len(prefix):]
                input_df.loc[0, feature] = int(value == category)

    # Get prediction and probability
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Customer may leave.")
        st.write(
            "Recommendation: Consider a suitable plan, "
            "loyalty offer, or personalized customer support."
        )
    else:
        st.success("Prediction: Customer is likely to stay.")
        st.write(
            "Recommendation: Continue good service "
            "and regular customer engagement."
        )

    st.subheader("Churn Risk Analysis")

    st.metric(
        "Estimated Churn Probability",
        f"{probability * 100:.2f}%"
    )
    st.progress(probability)

    st.write(
        f"Estimated probability that the customer will leave: "
        f"{probability * 100:.2f}%"
    )


    if probability >= 0.60:
        st.error("Risk Level: High")
    elif probability >= 0.30:
        st.warning("Risk Level: Medium")
    else:
        st.success("Risk Level: Low")

    st.caption(
        "This is a machine-learning estimate, not a guarantee "
        "of future customer behavior."
    )