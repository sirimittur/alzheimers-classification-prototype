
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Alzheimer's Classification Prototype",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "alzheimer_classifier.joblib"
FEATURES_PATH = BASE_DIR / "artifacts" / "model_features.json"

st.title("🧠 Alzheimer's Classification Prototype")
st.write(
    "Enter the requested information to see which of the two "
    "groups in the training dataset the model classifies the "
    "profile with."
)

st.warning(
    "Research/educational prototype only. This tool is not a "
    "medical test and cannot diagnose Alzheimer's disease or "
    "predict whether a healthy person will develop it in the "
    "future. Its output depends on the supplied datasets."
)

# ---------- Load saved model ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    saved_model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as file:
        saved_features = json.load(file)

    return saved_model, saved_features


try:
    model, features = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# ---------- User input form ----------
st.subheader("Enter profile details")

with st.form("profile_form"):
    age = st.slider(
        "Age (years)",
        min_value=60,
        max_value=90,
        value=70
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=10.0,
        max_value=75.0,
        value=25.0,
        step=0.1
    )

    smoking = st.selectbox(
        "Smoking status",
        ["No", "Yes"]
    )

    alcohol = st.selectbox(
        "Alcohol consumption category",
        ["Unknown", "Light", "Moderate", "Heavy"]
    )

    sleep = st.slider(
        "Sleep (hours per night)",
        min_value=4.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

    submitted = st.form_submit_button(
        "Classify profile",
        use_container_width=True
    )

# ---------- Classification ----------
if submitted:
    input_values = {
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Smoking": smoking,
        "AlcoholCategory": alcohol,
        "Sleep": sleep
    }

    input_df = pd.DataFrame([input_values])

    # Keep the same feature order used during training.
    input_df = input_df[features]

    try:
        prediction = int(model.predict(input_df)[0])

        st.divider()
        st.subheader("Model output")

        if prediction == 1:
            st.error(
                "The profile was classified closer to the "
                "Alzheimer-labelled group in the training data."
            )
        else:
            st.success(
                "The profile was classified closer to the "
                "control group in the training data."
            )

        st.caption(
            "This is a dataset-group classification, not a "
            "diagnosis, an individual's real-world disease "
            "probability, or a forecast of future Alzheimer's."
        )

        with st.expander("View submitted details"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Classification failed: {error}")

st.divider()
st.caption(
    "Features used: age, gender, BMI, smoking, alcohol category, "
    "and sleep. Physical activity is not included because it is "
    "absent from the two supplied datasets."
)
