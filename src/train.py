import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("healthcare_classification")
mlflow.autolog()


def load_data():
    # Load the provided healthcare dataset
    df = pd.read_csv("data/healthcare_dataset.csv")
    return df


def preprocess_data(df):
    # Binary classification: Cancer vs. not Cancer
    df["target"] = (df["Medical Condition"].str.lower() == "cancer").astype(int)
    # Use more features for training
    numerical = ["Age", "Billing Amount"]
    categorical = [
        "Gender",
        "Blood Type",
        "Admission Type",
        "Medication",
        "Test Results",
    ]
    features = numerical + categorical
    X = df[features]
    # Encode categorical features
    X = pd.get_dummies(X, columns=categorical)
    y = df["target"]
    return X, y


def train():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # mlflow.log_metric("accuracy", acc)
    # mlflow.sklearn.log_model(clf, "model")
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    train()
