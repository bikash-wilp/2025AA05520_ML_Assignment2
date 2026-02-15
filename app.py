import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

st.write("Files root:", os.listdir())
if os.path.exists("model"):
    st.write("Model files:", os.listdir("model"))

encoders = joblib.load("model/encoders.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_cols = joblib.load("model/features.pkl")

# ------------------------------
# LOAD MODELS
# ------------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("ML Classification Assignment App")

# MODEL SELECTION
selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[selected_model_name]

# DATASET UPLOAD
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV Only)",
    type=["csv"]
)

# ------------------------------
# PROCESS DATA
# ------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(df.head())

    # CHANGE TARGET COLUMN NAME IF DIFFERENT
    TARGET_COL = "income"

    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found!")
    else:

        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]
        y = y.astype(str).str.strip()

        # remove trailing dot
        y = y.str.replace(".", "", regex=False)

        # convert to numeric labels
        y = y.apply(lambda x: 1 if ">50K" in x else 0)

        X.replace("?", None, inplace=True)
        X.dropna(inplace=True)
        y = y.loc[X.index]


        # clean spaces
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].astype(str).str.strip()

        # encode using saved encoders
        for col, le in encoders.items():
            if col in X.columns:
                X[col] = X[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X[col] = le.transform(X[col])

        # reorder columns
        X = X[feature_cols]

        # scale
        X = scaler.transform(X)

        # PREDICTION
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:,1]
        else:
            y_prob = None

        # ------------------------------
        # METRICS
        # ------------------------------
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        if y_prob is not None:
            auc = roc_auc_score(y, y_prob)
        else:
            auc = "N/A"

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy","AUC","Precision","Recall","F1","MCC"],
            "Value": [accuracy, auc, precision, recall, f1, mcc]
        })

        st.table(metrics_df)

        # ------------------------------
        # CONFUSION MATRIX
        # ------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)

        st.pyplot(fig)

        # ------------------------------
        # CLASSIFICATION REPORT
        # ------------------------------
        st.subheader("Classification Report")

        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())