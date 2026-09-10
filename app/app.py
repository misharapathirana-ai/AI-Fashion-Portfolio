import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fashion AI",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "notebooks" / "fashion_category_classifier.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN TITLE
    -------------------------------------------------------- */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }


    /* --------------------------------------------------------
       SUBTITLE
    -------------------------------------------------------- */

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }


    /* --------------------------------------------------------
       PREDICTION CATEGORY
    -------------------------------------------------------- */

    .prediction-category {
        font-size: 32px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }


    /* --------------------------------------------------------
       PREDICTION CARD
    -------------------------------------------------------- */

    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }


    .prediction-label {
        font-size: 18px;
        font-weight: 600;
    }


    /* --------------------------------------------------------
       PRODUCT INFORMATION
    -------------------------------------------------------- */

    .product-info {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }


    .info-label {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 3px;
    }


    .info-value {
        font-size: 17px;
        margin-bottom: 15px;
    }


    /* --------------------------------------------------------
       PREDICT BUTTON
    -------------------------------------------------------- */

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 600;
        border: 1px solid rgba(128, 128, 128, 0.5);
        transition: all 0.2s ease;
    }


    /* Keep button text visible when hovering */

    div.stButton > button:hover {
        color: inherit !important;
        border: 1px solid currentColor;
        background-color: rgba(128, 128, 128, 0.15);
    }


    /* Keep button text visible when focused */

    div.stButton > button:focus {
        color: inherit !important;
        border: 1px solid currentColor;
        box-shadow: none !important;
    }


    /* --------------------------------------------------------
       SECTION SPACING
    -------------------------------------------------------- */

    .section-space {
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title"> Fashion AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Fashion Product Category Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PRODUCT INPUT SECTION
# ============================================================

st.subheader(" Product Information")

product_title = st.text_input(
    "Product Title",
    placeholder="Example: Nike Men's Air Max Running Shoes"
)

brand = st.text_input(
    "Brand",
    placeholder="Example: Nike"
)

price = st.number_input(
    "Price",
    min_value=0.0,
    value=100.0,
    step=1.0
)

st.write("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(" Predict Category"):

    # --------------------------------------------------------
    # CHECK PRODUCT TITLE
    # --------------------------------------------------------

    if product_title.strip() == "":
        st.warning("Please enter a product title.")

    # --------------------------------------------------------
    # CHECK BRAND
    # --------------------------------------------------------

    elif brand.strip() == "":
        st.warning("Please enter a brand.")

    else:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({
            "title": [product_title],
            "brand": [brand],
            "price": [price]
        })


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # GET PREDICTION PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(input_data)[0]

        confidence = max(probabilities) * 100


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-label">'
            ' AI Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-category">{prediction}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # DISPLAY CONFIDENCE
        # ====================================================

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # ====================================================
        # DISPLAY PRODUCT DETAILS
        # ====================================================

        st.markdown(
            '<div class="product-info">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-label">Product</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-value">{product_title}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-label">Brand</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-value">{brand}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-label">Price</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-value">{price:.2f}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.divider()

st.subheader(" About This AI Model")

st.write(
    """
    This application uses a machine learning model trained to classify
    fashion products into 11 different categories.

    The model uses product title, brand and price as input features.
    Product titles are transformed using TF-IDF, while brand information
    is encoded using One-Hot Encoding.
    """
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(" Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Accuracy",
        "78.83%"
    )


with col2:

    st.metric(
        "Macro F1 Score",
        "0.7882"
    )


st.caption(
    "Model evaluation was performed on a test set of 2,064 products."
)