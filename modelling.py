import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Load dataset
train_df = pd.read_csv("train_processed.csv")
test_df = pd.read_csv("test_processed.csv")


# Split feature dan target
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]


# Start MLflow
mlflow.set_experiment("Heart Disease Experiment")


with mlflow.start_run():

    # Autolog
    mlflow.sklearn.autolog()

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    print(classification_report(y_test, y_pred))
