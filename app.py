from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "churn_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_VERSION = "sklearn_churn_pipeline_v2"

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "Churn"


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in NUMERIC_COLS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned[TARGET_COL] = cleaned[TARGET_COL].astype(str).str.strip()
    cleaned = cleaned[cleaned[TARGET_COL].isin(["No", "Yes"])]
    return cleaned.dropna(subset=FEATURE_COLS + [TARGET_COL])


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", StandardScaler(), NUMERIC_COLS),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def train_and_save_model() -> dict:
    data = clean_churn_data(pd.read_csv(DATA_PATH))
    model = build_pipeline()
    model.fit(data[FEATURE_COLS], data[TARGET_COL])

    bundle = {
        "version": MODEL_VERSION,
        "model": model,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def is_valid_model_bundle(bundle: object) -> bool:
    if not isinstance(bundle, dict):
        return False

    model = bundle.get("model")
    return (
        bundle.get("version") == MODEL_VERSION
        and bundle.get("features") == FEATURE_COLS
        and hasattr(model, "predict")
        and hasattr(model, "predict_proba")
    )


@st.cache_resource(show_spinner="Loading churn model...")
def load_or_train_model() -> tuple[dict, str]:
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if is_valid_model_bundle(bundle):
                return bundle, "loaded"
        except Exception:
            pass

    return train_and_save_model(), "trained"


def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


def predict_customer(bundle: dict, customer: dict) -> tuple[str, float]:
    model = bundle["model"]
    input_df = pd.DataFrame([customer], columns=FEATURE_COLS)

    prediction = str(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.named_steps["model"].classes_)
    churn_index = classes.index("Yes") if "Yes" in classes else 1
    churn_probability = float(probabilities[churn_index])

    return prediction, churn_probability


def main() -> None:
    st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")

    st.title("Telecom Customer Churn Prediction")
    st.write("Enter a customer profile and predict whether the customer is likely to churn.")

    try:
        model_bundle, model_status = load_or_train_model()
    except Exception as exc:
        st.error("The churn model could not be loaded or trained.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.header("Model")
        st.write("Scikit-learn Random Forest")
        st.write(f"Data file: `{DATA_PATH.name}`")
        st.write(f"Model file: `{MODEL_PATH.name}`")
        if model_status == "trained":
            st.success("Model trained and saved.")
        else:
            st.success("Saved model loaded.")

    with st.form("customer_form"):
        st.subheader("Customer Details")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure in months", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col2:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0,
        )

        submitted = st.form_submit_button("Predict churn")

    if submitted:
        customer = {
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "SeniorCitizen": yes_no_to_int(senior_citizen),
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        prediction, probability = predict_customer(model_bundle, customer)

        st.subheader("Prediction")
        st.metric("Churn probability", f"{probability:.1%}")
        st.progress(int(probability * 100))

        if prediction == "Yes":
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is not likely to churn.")


if __name__ == "__main__":
    main()
