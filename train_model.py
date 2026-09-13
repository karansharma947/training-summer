import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import clone

DATA_PATH = "superstore_cleaned.csv"
MODEL_OUTPUT_PATH = "superstore_profit_loss_model.pkl"

FEATURES = [
    "Sales", "Quantity", "Discount", "Delivery Days", "Order Month Number",
    "Ship Mode", "Segment", "Region", "Category", "Sub-Category"
]

def load_data(filepath=DATA_PATH):
    """Loads and validates the dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def train_and_evaluate(df):
    """Trains multiple classifiers, evaluates them, and returns results + best model bundle."""
    X = df[FEATURES]
    y = df["Profit Status"]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=500), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=42, max_depth=5), False),
        "Random Forest": (RandomForestClassifier(random_state=42, n_estimators=60, n_jobs=-1), False),
    }

    results = {}
    fitted_models = {}

    print("\n--- Model Training & Evaluation ---")
    for name, (model, uses_scaling) in models.items():
        X_tr = X_train_scaled if uses_scaling else X_train
        X_te = X_test_scaled if uses_scaling else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="Loss", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="Loss", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="Loss", zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Loss Precision": prec,
            "Loss Recall": rec,
            "Loss F1-Score": f1,
        }
        fitted_models[name] = (model, uses_scaling)
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    comparison_df = pd.DataFrame(results).T.sort_values(
        by=["Loss F1-Score", "Loss Recall", "Accuracy"], ascending=False
    )
    print("\n--- Final Model Comparison ---")
    print(comparison_df.to_string())

    best_model_name = comparison_df.index[0]
    print(f"\n=> Best Model Selected: {best_model_name}")

    best_base_model, best_uses_scaling = fitted_models[best_model_name]
    final_model = clone(best_base_model)
    if hasattr(final_model, "n_jobs"):
        final_model.set_params(n_jobs=-1)

    if best_uses_scaling:
        final_scaler = StandardScaler()
        X_full = final_scaler.fit_transform(X_encoded)
    else:
        final_scaler = None
        X_full = X_encoded

    final_model.fit(X_full, y)

    bundle = {
        "comparison": comparison_df,
        "best_name": best_model_name,
        "final_model": final_model,
        "final_scaler": final_scaler,
        "encoded_columns": X_encoded.columns.tolist(),
        "features": FEATURES,
    }
    return bundle

def save_model(bundle, output_path=MODEL_OUTPUT_PATH):
    """Saves the trained model bundle to a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel bundle saved successfully to '{output_path}'")

def main():
    print("Starting Model Training Pipeline...")
    df = load_data()
    bundle = train_and_evaluate(df)
    save_model(bundle)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
