from pathlib import Path

import streamlit as st
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.sql.types import DoubleType, IntegerType


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "churn_data.csv"
MODEL_PATH = BASE_DIR / "churn_prediction_pipeline_model"

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


def spark_path(path: Path) -> str:
    return path.resolve().as_posix()


@st.cache_resource(show_spinner=False)
def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("TelecomCustomerChurnStreamlit")
        .master("local[2]")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def clean_churn_data(df):
    return (
        df.withColumn(
            "TotalCharges",
            when(trim(col("TotalCharges")) == "", "0.0").otherwise(col("TotalCharges")),
        )
        .withColumn("SeniorCitizen", col("SeniorCitizen").cast(IntegerType()))
        .withColumn("tenure", col("tenure").cast(IntegerType()))
        .withColumn("MonthlyCharges", col("MonthlyCharges").cast(DoubleType()))
        .withColumn("TotalCharges", col("TotalCharges").cast(DoubleType()))
        .dropna()
    )


def build_pipeline() -> Pipeline:
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep")
        for column in CATEGORICAL_COLS
    ]
    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
    assembler = VectorAssembler(
        inputCols=[f"{column}_index" for column in CATEGORICAL_COLS] + NUMERIC_COLS,
        outputCol="raw_features",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False,
    )
    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=25,
        maxDepth=5,
        seed=42,
    )
    return Pipeline(stages=indexers + [label_indexer, assembler, scaler, classifier])


@st.cache_resource(show_spinner="Loading PySpark model...")
def load_or_train_model():
    spark = get_spark_session()

    if MODEL_PATH.exists():
        return PipelineModel.load(spark_path(MODEL_PATH)), "loaded"

    df = spark.read.csv(spark_path(DATA_PATH), header=True, inferSchema=True)
    df = clean_churn_data(df)

    model = build_pipeline().fit(df)
    model.write().overwrite().save(spark_path(MODEL_PATH))
    return model, "trained"


def get_label_order(model: PipelineModel) -> list[str]:
    for stage in model.stages:
        if hasattr(stage, "labels") and stage.getOutputCol() == "label":
            return list(stage.labels)
    return ["No", "Yes"]


def predict_customer(model: PipelineModel, customer: dict):
    spark = get_spark_session()
    input_df = spark.createDataFrame([customer])
    row = model.transform(input_df).select("prediction", "probability").first()
    labels = get_label_order(model)

    predicted_label = labels[int(row["prediction"])]
    churn_index = labels.index("Yes") if "Yes" in labels else 1
    churn_probability = float(row["probability"][churn_index])
    return predicted_label, churn_probability


def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


def main() -> None:
    st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")

    st.title("Telecom Customer Churn Prediction")
    st.write("Enter a customer profile and predict whether the customer is likely to churn.")

    try:
        model, model_status = load_or_train_model()
    except Exception as exc:
        st.error("PySpark could not load or train the churn model.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.header("Model")
        st.write("PySpark Random Forest")
        st.write(f"Data file: `{DATA_PATH.name}`")
        if model_status == "trained":
            st.success("Model trained and saved.")
        else:
            st.success("Saved Spark model loaded.")

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
            "SeniorCitizen": yes_no_to_int(senior_citizen),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
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
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "Churn": "No",
        }

        prediction, probability = predict_customer(model, customer)

        st.subheader("Prediction")
        st.metric("Churn probability", f"{probability:.1%}")
        st.progress(int(probability * 100))

        if prediction == "Yes":
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is not likely to churn.")


if __name__ == "__main__":
    main()
