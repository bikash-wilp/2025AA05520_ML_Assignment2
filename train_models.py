import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


cols = [
"age","workclass","fnlwgt","education","education_num",
"marital_status","occupation","relationship","race","sex",
"capital_gain","capital_loss","hours_per_week",
"native_country","income"
]

df_train = pd.read_csv(
    "data/adult.data",
    names=cols,
    sep=",",
    skipinitialspace=True
)

print(df_train.head())

df_test = pd.read_csv(
    "data/adult.test",
    names=cols,
    sep=",",
    skiprows=1,
    skipinitialspace=True
)

df = pd.concat([df_train, df_test], ignore_index=True)

# df_train.to_csv("data/adult_income.csv", index=False)
# df_test.to_csv("data/adult_income_test.csv", index=False)


df = pd.read_csv("data/adult_income.csv")

df["income"] = df["income"].str.replace(".", "", regex=False)
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)
df["income"] = df["income"].map({
"<=50K":0,
">50K":1
})

encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str).str.strip()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le



X = df.drop("income", axis=1)
y = df["income"]

feature_cols = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_cols, "model/features.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "logistic": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier()
}

def evaluate(model):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = evaluate(model)
    joblib.dump(model, f"model/{name}.pkl")

print(pd.DataFrame(results).T)