


"""Utilities to load data, train and save a RandomForest model.

This module is a lightweight training script used to build the
`parkinsons_model.pkl` consumed by the API. It provides helpers
to load data, prepare features, train and evaluate a model.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_data(path: str = "parkinsons.csv") -> pd.DataFrame:
    """Load dataset from CSV file located at `path`."""
    return pd.read_csv(path)


def prepare_features(data: pd.DataFrame):
    """Split DataFrame into features and target."""
    X = data.drop(columns=["name", "status"])
    y = data["status"]
    return X, y


def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """Train and return a RandomForest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> None:
    """Print classification report for model on test data."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


def save_model(model, path: str = "parkinsons_model.pkl") -> None:
    """Persist trained model to disk."""
    joblib.dump(model, path)


def main() -> None:
    data = load_data()
    print(data.head())

    X, y = prepare_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Train the classifier on the training split
    model = train_model(X_train, y_train)

    # Evaluate with the held-out test set
    evaluate_model(model, X_test, y_test)

    # Persist the trained model for inference
    save_model(model)

    # Fit a scaler on the training data and persist it. Note: the scaler
    # should normally be fit before any model pipeline that expects scaled
    # input; here we save it separately for future preprocessing at inference.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "scaler.pkl")

    # Print feature importances for debugging / analysis
    importances = model.feature_importances_
    print(importances)
    





if __name__ == "__main__":
    main()




