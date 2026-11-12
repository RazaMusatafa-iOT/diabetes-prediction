"""
Diabetes Progression Predictor - Streamlit UI

Serves the Ridge pipeline saved in best_diabetes_model.joblib.
Run with:  streamlit run main.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes

MODEL_PATH = Path(__file__).parent / "best_diabetes_model.joblib"
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# Friendly labels, units and step size for each feature (raw clinical values)
FEATURE_INFO = {
    "age": ("Age", "years", 1.0),
    "sex": ("Sex", "group 1 or 2", 1.0),
    "bmi": ("Body Mass Index", "kg/m²", 0.1),
    "bp":  ("Average Blood Pressure", "mm Hg", 1.0),
    "s1":  ("Total Serum Cholesterol (tc)", "mg/dL", 1.0),
    "s2":  ("LDL Cholesterol (ldl)", "mg/dL", 0.1),
    "s3":  ("HDL Cholesterol (hdl)", "mg/dL", 1.0),
    "s4":  ("Total Chol / HDL ratio (tch)", "ratio", 0.01),
    "s5":  ("Log of Serum Triglycerides (ltg)", "log", 0.01),
    "s6":  ("Blood Sugar Level (glu)", "mg/dL", 1.0),
}

st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺", layout="wide")


# ---------- Loading ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference():
    """Raw dataset stats, used to convert clinical values to the model's scaled format.

    sklearn's diabetes features are scaled as (x - mean) / sqrt(sum((x - mean)^2)),
    so we rebuild that exact transform from the unscaled copy of the dataset.
    """
    raw = load_diabetes(scaled=False, as_frame=True).frame
    X_raw = raw[FEATURES]
    mean = X_raw.mean()
    norm = np.sqrt(((X_raw - mean) ** 2).sum())
    return X_raw, raw["target"], mean, norm


def to_model_scale(df_raw: pd.DataFrame, mean: pd.Series, norm: pd.Series) -> pd.DataFrame:
    return (df_raw[FEATURES] - mean) / norm


def progression_band(value: float, target: pd.Series) -> tuple[str, str]:
    q1, q2, q3 = target.quantile([0.25, 0.5, 0.75])
    if value < q1:
        return "Low", "green"
    if value < q2:
        return "Below average", "blue"
    if value < q3:
        return "Above average", "orange"
    return "High", "red"


# ---------- App ----------
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put it next to main.py.")
    st.stop()

model = load_model()
X_raw, target, mean, norm = load_reference()

st.title("🩺 Diabetes Progression Predictor")
st.caption(
    "Predicts a quantitative measure of disease progression **one year after baseline** "
    "using a tuned Ridge Regression model trained on the scikit-learn Diabetes dataset."
)

tab_single, tab_batch, tab_about = st.tabs(["Single Prediction", "Batch Prediction (CSV)", "About the Model"])

# ---- Single prediction ----
with tab_single:
    st.subheader("Patient measurements")
    st.write("Defaults are the dataset medians. Ranges are limited to what the model saw in training.")

    values = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        label, unit, step = FEATURE_INFO[feat]
        lo, hi, med = float(X_raw[feat].min()), float(X_raw[feat].max()), float(X_raw[feat].median())
        with cols[i % 2]:
            if feat == "sex":
                values[feat] = float(st.radio(f"{label}", [1, 2], horizontal=True, key=feat))
            else:
                values[feat] = st.slider(f"{label} ({unit})", lo, hi, med, step=step, key=feat)

    if st.button("Predict", type="primary", use_container_width=True):
        patient_raw = pd.DataFrame([values])
        patient_scaled = to_model_scale(patient_raw, mean, norm)
        pred = float(model.predict(patient_scaled)[0])
        band, color = progression_band(pred, target)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted progression score", f"{pred:.1f}")
        c2.metric("Dataset average", f"{target.mean():.1f}", delta=f"{pred - target.mean():+.1f}", delta_color="inverse")
        c3.markdown(f"**Level**  \n:{color}[**{band}**]")

        pct = (target < pred).mean()
        st.progress(min(max(pct, 0.0), 1.0), text=f"Higher than {pct:.0%} of patients in the dataset "
                                                   f"(range {target.min():.0f} – {target.max():.0f})")

        with st.expander("Show the values sent to the model"):
            st.dataframe(pd.concat([patient_raw, patient_scaled], keys=["Raw input", "Scaled for model"]),
                         use_container_width=True)

# ---- Batch prediction ----
with tab_batch:
    st.subheader("Predict for many patients")
    st.write(f"Upload a CSV with these columns (raw clinical values): `{', '.join(FEATURES)}`")

    template = X_raw.head(5).to_csv(index=False).encode()
    st.download_button("Download sample CSV", template, "sample_patients.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif df_in[FEATURES].isnull().any().any():
            st.error("Some required values are empty. Please fill them and upload again.")
        else:
            df_out = df_in.copy()
            df_out["predicted_progression"] = model.predict(to_model_scale(df_in, mean, norm)).round(1)
            df_out["level"] = [progression_band(v, target)[0] for v in df_out["predicted_progression"]]
            st.success(f"Predicted {len(df_out)} patients.")
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download predictions", df_out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ---- About ----
with tab_about:
    st.subheader("Model details")
    st.markdown(
        """
- **Pipeline:** `StandardScaler` → `Ridge(alpha=1, solver='lsqr')`
- **Selected from:** 15+ regressors compared with 5-fold cross-validation, then tuned with GridSearchCV
- **Performance on the test set:** R² ≈ 0.45, RMSE ≈ 53.8, MAE ≈ 42.8
- **Data:** 442 patients, 10 baseline features (scikit-learn Diabetes dataset)
        """
    )
    st.markdown("**Ridge coefficients (on standardized features)**")
    coefs = pd.Series(model.named_steps["model"].coef_, index=FEATURES).sort_values()
    st.bar_chart(coefs, horizontal=True)
    st.info(
        "This is a learning project, not a medical tool. An R² of ~0.45 means the model explains "
        "under half of the variation in progression, so predictions should be read as rough estimates."
    )
