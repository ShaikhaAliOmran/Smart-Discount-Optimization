
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BrightMart Discount Optimization",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    path = Path(__file__).resolve().parent / "models" / "uplift_t_learner.joblib"
    return joblib.load(path)

st.title("BrightMart | Smart Discount Optimization")
st.write(
    "Find out whether offering a discount may increase "
    "a customer's chance of making a purchase."
)

try:
    bundle = load_model()
except Exception as error:
    st.error("Unable to load the saved model.")
    st.exception(error)
    st.stop()

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input(
        "Months since last purchase",
        min_value=1, max_value=12, value=3
    )
    history = st.number_input(
        "Historical spending ($)",
        min_value=0.0, value=150.0, step=10.0
    )
    used_discount = st.selectbox(
        "Previously used a discount?",
        ["No", "Yes"]
    )
    used_bogo = st.selectbox(
        "Previously used BOGO?",
        ["No", "Yes"]
    )

with col2:
    zip_code = st.selectbox(
        "Customer location",
        ["Urban", "Suburban", "Rural"]
    )
    is_referral = st.selectbox(
        "Referral customer?",
        ["No", "Yes"]
    )
    channel = st.selectbox(
        "Shopping channel",
        ["Web", "Phone", "Multichannel"]
    )

if st.button("Predict Discount Uplift", type="primary"):

    discount_binary = int(used_discount == "Yes")
    bogo_binary = int(used_bogo == "Yes")
    referral_binary = int(is_referral == "Yes")

    customer = {
        "recency": recency,
        "history": history,
        "used_discount": discount_binary,
        "used_bogo": bogo_binary,
        "zip_code": zip_code,
        "is_referral": referral_binary,
        "channel": channel,
        "recent_customer": int(recency <= 3),
        "history_log": np.log1p(history)
    }

    features = pd.DataFrame([customer])
    features = features[bundle["feature_columns"]]

    try:
        p_discount = bundle["discount_model"].predict_proba(
            features
        )[0, 1]

        p_control = bundle["control_model"].predict_proba(
            features
        )[0, 1]

        uplift = p_discount - p_control

        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "With Discount",
            f"{p_discount:.2%}"
        )
        c2.metric(
            "Without Offer",
            f"{p_control:.2%}"
        )
        c3.metric(
            "Estimated Uplift",
            f"{uplift:+.2%}"
        )

        if uplift > 0:
            st.success(
                "Recommendation: Offer a Discount"
            )
            st.write(
                "The model estimates that this customer "
                "is more likely to purchase with a discount."
            )
        else:
            st.info(
                "Recommendation: No Offer"
            )
            st.write(
                "The model does not estimate an increase "
                "in purchase probability from a discount."
            )

        st.caption(
            "Predictions are estimates based on historical "
            "data and do not guarantee customer behavior. "
            "Recommendations do not account for discount costs."
        )

    except Exception as error:
        st.error("Unable to generate a prediction.")
        st.exception(error)
