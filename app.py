
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="BrightMart | Smart Discount Optimization",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = Path(__file__).resolve().parent

BRIGHTMART_LOGO = BASE / "assets" / "BrightMart Shopping Analytics Logo.png"
HCT_LOGO = BASE / "assets" / "Higher Colleges of Technology Logo.png"

# --------------------------------------------------
# WEBSITE THEME
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F7F0E7;
        color: #493326;
    }

    [data-testid="stSidebar"] {
        background-color: #EDE0D2;
    }

    [data-testid="stSidebar"] * {
        color: #493326;
    }

    h1, h2, h3 {
        color: #513525 !important;
    }

    .stApp [data-testid="stMarkdownContainer"] p {
        color: #6C5545;
    }

    .stButton > button[kind="primary"],
    .stButton > button {
        background-color: #624330;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
    }

    .stButton > button:hover {
        background-color: #805A40;
        color: white !important;
        border: none;
    }

    .stButton button p {
        color: white !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6D8CA;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #513525;
    }

    [data-testid="stMetricLabel"] {
        color: #806858;
    }

    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="select"] > div {
        background-color: #FFFFFF;
        color: #493326;
        border-radius: 9px;
    }

    .hero {
        background-color: #E9D9C7;
        padding: 35px;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #493021;
        margin-bottom: 12px;
    }

    .hero p {
        color: #705544;
        font-size: 17px;
    }

    .info-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #E6D8CA;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .info-card h3 {
        margin-top: 0;
    }

    
/* Improve recommendation message readability */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #FFFFFF !important;
}

/* Improve download button readability */
[data-testid="stDownloadButton"] button {
    background-color: #624330 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

[data-testid="stDownloadButton"] button p {
    color: #FFFFFF !important;
}


/* Elegant top navigation */

[data-testid="stSegmentedControl"] {
    width: 100%;
    margin: 12px 0 20px;
}

[data-testid="stSegmentedControl"] [role="radiogroup"] {
    display: flex !important;
    width: 100% !important;
    gap: 10px !important;
    background: transparent !important;
}

[data-testid="stSegmentedControl"] button {
    flex: 1 !important;
    min-height: 58px !important;
    padding: 14px 20px !important;
    background-color: #E8DCCB !important;
    color: #513525 !important;
    border: 1px solid #DCCBBA !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

[data-testid="stSegmentedControl"] button * {
    color: inherit !important;
}

[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-selected="true"] {
    background-color: #654330 !important;
    color: #FFFFFF !important;
    border-color: #654330 !important;
}


/* BrightMart top navigation buttons */

.st-key-top_nav button {
    width: 100% !important;
    min-height: 64px !important;
    background-color: #E8DCCB !important;
    color: #513525 !important;
    border: 1px solid #D8C5B3 !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

.st-key-top_nav button p,
.st-key-top_nav button span {
    color: #513525 !important;
    font-size: 17px !important;
}

.st-key-top_nav button[kind="primary"] {
    background-color: #654330 !important;
    color: #FFFFFF !important;
    border-color: #654330 !important;
}

.st-key-top_nav button:hover {
    border-color: #654330 !important;
}


/* White text on selected navigation button */
.st-key-top_nav button[kind="primary"] p,
.st-key-top_nav button[kind="primary"] span {
    color: #FFFFFF !important;
}


</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD THE EXISTING TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    path = BASE / "models" / "uplift_t_learner.joblib"
    return joblib.load(path)


try:
    bundle = load_model()
except Exception as error:
    st.error("Unable to load the saved model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# CUSTOMER EXAMPLES
# --------------------------------------------------

EXAMPLES = {
    "Recent Customer": {
        "recency": 1,
        "history": 500.0,
        "used_discount": "Yes",
        "used_bogo": "Yes",
        "zip_code": "Urban",
        "is_referral": "No",
        "channel": "Web"
    },
    "Inactive Customer": {
        "recency": 12,
        "history": 40.0,
        "used_discount": "No",
        "used_bogo": "No",
        "zip_code": "Rural",
        "is_referral": "No",
        "channel": "Phone"
    },
    "High-Spending Customer": {
        "recency": 6,
        "history": 1200.0,
        "used_discount": "Yes",
        "used_bogo": "No",
        "zip_code": "Suburban",
        "is_referral": "Yes",
        "channel": "Multichannel"
    }
}

DEFAULT_CUSTOMER = {
    "recency": 3,
    "history": 150.0,
    "used_discount": "No",
    "used_bogo": "No",
    "zip_code": "Urban",
    "is_referral": "No",
    "channel": "Web"
}


def clear_prediction():
    st.session_state.pop("last_prediction", None)


def apply_example(name):
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value
    clear_prediction()


for key, value in DEFAULT_CUSTOMER.items():
    st.session_state.setdefault(key, value)


# --------------------------------------------------
# WEBSITE HEADER
# --------------------------------------------------

logo_col, hct_col = st.columns([3, 1.5])

with logo_col:
    if BRIGHTMART_LOGO.exists():
        st.image(str(BRIGHTMART_LOGO), width=310)
    else:
        st.title("BrightMart")

with hct_col:
    if HCT_LOGO.exists():
        st.image(str(HCT_LOGO), width=260)

st.divider()



# --------------------------------------------------
# TOP NAVIGATION
# --------------------------------------------------


if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

with st.container(key="top_nav"):
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        if st.button(
            "Home",
            use_container_width=True,
            type="primary" if st.session_state.active_page == "Home" else "secondary"
        ):
            st.session_state.active_page = "Home"
        

    with nav2:
        if st.button(
            "Customer Predictions",
            use_container_width=True,
            type="primary" if st.session_state.active_page == "Customer Predictions" else "secondary"
        ):
            st.session_state.active_page = "Customer Predictions"
           

    with nav3:
        if st.button(
            "Model Insights",
            use_container_width=True,
            type="primary" if st.session_state.active_page == "Model Insights" else "secondary"
        ):
            st.session_state.active_page = "Model Insights"
           

page = st.session_state.active_page

st.divider()


# --------------------------------------------------
# PAGE 1: HOME
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Smarter discounts. Better decisions.</h1>
        <p>
        Discover how machine learning can help BrightMart
        estimate customer responses to promotional offers
        and support smarter marketing decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Customer Records", "64,000")
    c2.metric("Models Compared", "4")
    c3.metric("Selected Approach", "T-Learner")

    st.markdown("""
    <div class="info-card">
        <h3>Our Business Problem</h3>
        <p>
        Offering discounts to every customer can lead to
        unnecessary promotional spending. Some customers
        may purchase products even without receiving an offer.
        </p>
        <p>
        BrightMart aims to identify customers who are more
        likely to respond positively to discounts, helping
        marketing managers make better targeting decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What Can You Do Here?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Customer Predictions</h3>
            <p>
            Enter customer information and estimate the
            difference in purchase probability between
            receiving a discount and receiving no offer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Model Insights</h3>
            <p>
            Learn how the Gradient Boosting T-Learner works,
            why uplift modeling is useful, and what the
            predictions mean for BrightMart.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Use the sidebar to explore Customer Predictions "
        "or Model Insights."
    )


# --------------------------------------------------
# PAGE 2: CUSTOMER PREDICTIONS
# --------------------------------------------------

elif page == "Customer Predictions":

    st.title("Customer Uplift Prediction")

    st.write(
        "Enter customer information to estimate whether "
        "a discount may increase purchase probability."
    )

    st.markdown("### Try an Example Customer")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.button(
            "Recent Customer",
            on_click=apply_example,
            args=("Recent Customer",),
            use_container_width=True
        )

    with e2:
        st.button(
            "Inactive Customer",
            on_click=apply_example,
            args=("Inactive Customer",),
            use_container_width=True
        )

    with e3:
        st.button(
            "High-Spending Customer",
            on_click=apply_example,
            args=("High-Spending Customer",),
            use_container_width=True
        )

    st.markdown("### Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Months since last purchase",
            min_value=1,
            max_value=12,
            step=1,
            key="recency",
            on_change=clear_prediction
        )

        history = st.number_input(
            "Historical spending ($)",
            min_value=0.0,
            step=10.0,
            key="history",
            on_change=clear_prediction
        )

        used_discount = st.selectbox(
            "Previously used a discount?",
            ["No", "Yes"],
            key="used_discount",
            on_change=clear_prediction
        )

        used_bogo = st.selectbox(
            "Previously used BOGO?",
            ["No", "Yes"],
            key="used_bogo",
            on_change=clear_prediction
        )

    with col2:
        zip_code = st.selectbox(
            "Customer location",
            ["Urban", "Suburban", "Rural"],
            key="zip_code",
            on_change=clear_prediction
        )

        is_referral = st.selectbox(
            "Referral customer?",
            ["No", "Yes"],
            key="is_referral",
            on_change=clear_prediction
        )

        channel = st.selectbox(
            "Shopping channel",
            ["Web", "Phone", "Multichannel"],
            key="channel",
            on_change=clear_prediction
        )

    if st.button(
        "Predict Discount Uplift",
        type="primary",
        use_container_width=True
    ):

        # Preserve the existing feature engineering
        customer = {
            "recency": recency,
            "history": history,
            "used_discount": int(used_discount == "Yes"),
            "used_bogo": int(used_bogo == "Yes"),
            "zip_code": zip_code,
            "is_referral": int(is_referral == "Yes"),
            "channel": channel,
            "recent_customer": int(recency <= 3),
            "history_log": np.log1p(history)
        }

        features = pd.DataFrame([customer])
        features = features[bundle["feature_columns"]]

        try:
            # Existing Gradient Boosting T-Learner
            p_discount = bundle[
                "discount_model"
            ].predict_proba(features)[0, 1]

            p_control = bundle[
                "control_model"
            ].predict_proba(features)[0, 1]

            uplift = p_discount - p_control

            st.session_state["last_prediction"] = {
                "discount": float(p_discount),
                "control": float(p_control),
                "uplift": float(uplift),
                "customer": customer
            }

        except Exception as error:
            st.error("Unable to generate a prediction.")
            st.exception(error)

    # Display prediction results
    if "last_prediction" in st.session_state:

        result = st.session_state["last_prediction"]

        p_discount = result["discount"]
        p_control = result["control"]
        uplift = result["uplift"]

        st.divider()
        st.header("Prediction Results")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "With Discount",
            f"{p_discount:.2%}"
        )

        r2.metric(
            "Without Offer",
            f"{p_control:.2%}"
        )

        r3.metric(
            "Estimated Uplift",
            f"{uplift:+.2%}"
        )

        st.markdown("### Purchase Probability Comparison")

       
        st.markdown(
            f"""
            <div style="margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between;
                            color:#513525; margin-bottom:10px;">
                    <span>With Discount</span>
                    <strong>{p_discount:.2%}</strong>
                </div>
                <div style="background:#E6D8CA; height:14px;
                            border-radius:20px; overflow:hidden;">
                    <div style="background:#654330;
                                width:{np.clip(p_discount, 0, 1)*100:.2f}%;
                                height:100%; border-radius:20px;">
                    </div>
                </div>
            </div>

            <div style="margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between;
                            color:#513525; margin-bottom:10px;">
                    <span>Without Offer</span>
                    <strong>{p_control:.2%}</strong>
                </div>
                <div style="background:#E6D8CA; height:14px;
                            border-radius:20px; overflow:hidden;">
                    <div style="background:#B69C84;
                                width:{np.clip(p_control, 0, 1)*100:.2f}%;
                                height:100%; border-radius:20px;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("### Recommendation")

     
        if uplift > 0:
            st.markdown(
                '<div style="background-color:#D2D9CD; '
                'color:#3E5945; padding:20px; '
                'border-radius:12px; font-weight:600;">'
                '✓ Recommendation: Offer a Discount</div>',
                unsafe_allow_html=True
            )
            st.write(
                "The model estimates that this customer "
                "is more likely to purchase with a discount."
            )
        else:
            st.markdown(
                '<div style="background-color:#536779; '
                'color:#FFFFFF; padding:20px; '
                'border-radius:12px; font-weight:600;">'
                '− Recommendation: No Offer</div>',
                unsafe_allow_html=True
            )
            st.write(
                "The model does not estimate an increase "
                "in purchase probability from a discount."
            )


        st.markdown("### Understanding Your Result")

        st.write(
            "Estimated uplift is the difference between "
            "the predicted purchase probability with a "
            "discount and without an offer."
        )

        st.write(
            f"For this customer, the estimated change "
            f"in purchase probability is "
            f"{uplift * 100:+.2f} percentage points."
        )

        st.caption(
            "These are model estimates, not guaranteed "
            "outcomes. The recommendation does not account "
            "for the financial cost of discounts."
        )

        # Download the current prediction
        export = result["customer"].copy()
        export.update({
            "discount_probability": p_discount,
            "no_offer_probability": p_control,
            "estimated_uplift": uplift,
            "recommendation": (
                "Offer Discount" if uplift > 0 else "No Offer"
            )
        })

        csv_data = pd.DataFrame([export]).to_csv(
            index=False
        )

        st.download_button(
            "Download Prediction Results",
            data=csv_data,
            file_name="brightmart_prediction.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# PAGE 3: MODEL INSIGHTS
# --------------------------------------------------

elif page == "Model Insights":

    st.title("Model Insights")

    st.write(
        "Explore the machine learning approach used "
        "to estimate customer discount uplift."
    )

    st.markdown("### Selected Model")

    st.markdown("""
    <div class="info-card">
        <h3>Gradient Boosting T-Learner</h3>
        <p>
        The T-Learner uses two separate machine learning
        models to estimate purchase probabilities under
        different promotional conditions.
        </p>
        <p>
        One model estimates conversion with a discount,
        while the other estimates conversion without
        an offer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Models Compared")

    comparison = pd.DataFrame({
        "Machine Learning Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Status": [
            "Compared",
            "Compared",
            "Compared",
            "Selected"
        ]
    })

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### How Uplift Is Calculated")

    st.latex(
        r"Uplift = P(Purchase|Discount)"
        r" - P(Purchase|No\ Offer)"
    )

    st.write(
        "Positive uplift suggests that a discount may "
        "increase the customer's purchase probability."
    )

    st.write(
        "Zero or negative uplift suggests that the model "
        "does not predict an additional purchase benefit "
        "from offering a discount."
    )

    st.markdown("### Business Value")

    st.markdown("""
    <div class="info-card">
        <h3>Supporting Marketing Decisions</h3>
        <p>
        BrightMart can use estimated uplift to help
        prioritize customers who may respond positively
        to discounts rather than sending the same
        promotion to everyone.
        </p>
        <p>
        This may support better promotional targeting
        and reduce unnecessary discount spending.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Limitations")

    st.warning(
        "Individual uplift predictions are uncertain. "
        "The model does not guarantee additional purchases "
        "or higher profits. Discount costs and business "
        "constraints should also be considered before "
        "using recommendations in real campaigns."
    )


# --------------------------------------------------
# WEBSITE FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "BrightMart | Smart Discount Optimization"
)

st.caption(
    "Academic Data Science Project — "
    "Higher Colleges of Technology"
)
