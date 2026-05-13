import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


train_df = pd.read_csv("train_processed.csv")
test_df = pd.read_csv("test_processed.csv")


X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]


mlflow.set_experiment("Heart Disease Experiment")


mlflow.sklearn.autolog()


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))
