import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Placeholder for dataset loading
# Expected columns: gender, bmi_category, goal, schedule, meal_plan

import os



def load_data(path: str | None = None):
    """Load training data from a CSV file.

    If no path is provided the function will prefer `GYM.csv` (the dataset
    the user supplied) and fall back to the placeholder `dataset.csv`.
    """

    if path is None:
        default1 = os.path.join(os.path.dirname(__file__), "..", "data", "GYM.csv")
        default2 = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
        if os.path.exists(default1):
            path = default1
        else:
            path = default2
    return pd.read_csv(path)


def preprocess(df):
    """Encode categorical columns and normalise names.

    The dataset coming from `GYM.csv` uses spaces and slightly different
    column names (`BMI Category`, `Exercise Schedule`), so we normalise the
    headers to lowercase underscore form.  Each column gets its own
    ``LabelEncoder`` stored in a dict so we can inverse-transform later.
    """

    # normalise column names for easier reference
    df = df.rename(columns=lambda c: c.strip().lower().replace(' ', '_'))

    encoders = {}
    for col in ['gender', 'bmi_category', 'goal', 'exercise_schedule', 'meal_plan']:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in dataset, got {list(df.columns)}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def main():
    df = load_data()
    df, encoders = preprocess(df)
    X = df[['gender', 'bmi_category', 'goal']]
    # outputs use the normalised names
    y = df[['exercise_schedule', 'meal_plan']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # simple decision tree for multi-output
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    # ``classification_report`` does not support multioutput directly; print
    # separate reports for each target column.
    for i, col in enumerate(y.columns):
        print(f"\nClassification report for output '{col}':")
        print(classification_report(y_test.iloc[:, i], preds[:, i]))

    # determine output paths relative to this script's location
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(base, exist_ok=True)
    joblib.dump(model, os.path.join(base, 'workout_diet_model.pkl'))
    joblib.dump(encoders, os.path.join(base, 'encoders.pkl'))

    print("Model trained and saved.")


if __name__ == '__main__':
    main()
