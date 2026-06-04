# ================================================
# TITANIC SURVIVAL PREDICTOR
# Author: Rahul Kumar
# GitHub: rahultwoapl8130
# Tech: Python | Scikit-learn | GCP | BigQuery
# ================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ──────────────────────────────────
titanic = sns.load_dataset("titanic")
print("Data loaded:", titanic.shape)

# ── CLEAN DATA ─────────────────────────────────
features = ["pclass","sex","age","sibsp","parch","fare","embarked"]
df = titanic[features + ["survived"]].copy()
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df.dropna(inplace=True)
df["sex"]      = df["sex"].map({"female":0,"male":1})
df["embarked"] = df["embarked"].map({"C":0,"Q":1,"S":2})
print("Cleaned:", df.shape)

# ── TRAIN TEST SPLIT ───────────────────────────
X = df[features]
y = df["survived"]
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train:{X_train.shape[0]} Test:{X_test.shape[0]}")

# ── TRAIN MODELS ───────────────────────────────
models = {
    "Random Forest":      RandomForestClassifier(n_estimators=100,random_state=42),
    "Decision Tree":      DecisionTreeClassifier(max_depth=6,random_state=42),
    "Logistic Regression":LogisticRegression(max_iter=1000),
}
for name,model in models.items():
    model.fit(X_train,y_train)
    acc = accuracy_score(y_test,model.predict(X_test))
    print(f"{name}: {acc*100:.2f}%")

# ── BEST MODEL: RANDOM FOREST ──────────────────
best_model = models["Random Forest"]
print("\nBest Model Accuracy:",
      accuracy_score(y_test,best_model.predict(X_test)))
print(classification_report(y_test,best_model.predict(X_test),
      target_names=["Died","Survived"]))

# ── CROSS VALIDATION ───────────────────────────
cv = cross_val_score(best_model, X, y, cv=5)
print(f"CV Mean: {cv.mean():.4f}  Std: {cv.std():.4f}")

# ── FEATURE IMPORTANCE ─────────────────────────
importances = pd.Series(
    best_model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\nFeature Importance:\n", importances)

# ── SAVE MODEL ─────────────────────────────────
joblib.dump(best_model, "model.joblib")
print("model.joblib saved!")

# ── PREDICT NEW PASSENGERS ─────────────────────
new_data = pd.DataFrame({
    "pclass":[3,1,2],
    "sex":[1,0,0],
    "age":[22,38,26],
    "sibsp":[1,1,0],
    "parch":[0,0,0],
    "fare":[7.25,71.28,13.0],
    "embarked":[2,0,0]
})
preds = best_model.predict(new_data)
probs = best_model.predict_proba(new_data)
for i,(p,prob) in enumerate(zip(preds,probs)):
    status = "Survived" if p==1 else "Did Not Survive"
    print(f"Passenger {i+1}: {status} ({max(prob)*100:.1f}%)")
