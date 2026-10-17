"""Compare full-feature and K-means-selected XGBoost HAR classifiers."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_DIR = Path("Processed Data")
REPRESENTATION = "features"  # "features" (561 inputs) or "signals" (1,152 inputs)
N_SELECTED_FEATURES = 50
MODEL_OUT = Path("har_model.joblib")
RANDOM_STATE = 42


class ClusterGuidedFeatureSelector:
    """Cluster standardized feature profiles and keep one representative per cluster."""

    def __init__(self, n_features=50, random_state=42):
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, x):
        if not 1 <= self.n_features <= x.shape[1]:
            raise ValueError(
                f"n_features must be between 1 and {x.shape[1]}, got {self.n_features}."
            )

        # Each column is treated as a feature profile across training windows.
        self.scaler_ = StandardScaler()
        profiles = self.scaler_.fit_transform(x).T
        self.kmeans_ = KMeans(
            n_clusters=self.n_features,
            n_init=10,
            random_state=self.random_state,
        ).fit(profiles)

        representatives = []
        for cluster_id in range(self.n_features):
            members = np.flatnonzero(self.kmeans_.labels_ == cluster_id)
            if members.size == 0:
                raise RuntimeError(f"K-means produced an empty cluster {cluster_id}.")
            distances = np.linalg.norm(
                profiles[members] - self.kmeans_.cluster_centers_[cluster_id], axis=1
            )
            representatives.append(int(members[np.argmin(distances)]))
        self.selected_indices_ = np.array(sorted(representatives), dtype=int)
        return self

    def transform(self, x):
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("Call fit before transform.")
        return x.iloc[:, self.selected_indices_]


def make_classifier():
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
    )


def evaluate(name, model, x_train, y_train, x_test, y_test, class_names):
    start = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    predictions = model.predict(x_test)
    labels = sorted(class_names)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name}")
    print(f"Feature count: {x_train.shape[1]:,}")
    print(f"Classifier fit time: {fit_seconds:.2f} seconds")
    print(f"Accuracy: {accuracy:.4%}")
    print("Per-activity metrics:")
    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=[class_names[label] for label in labels],
        zero_division=0,
    ))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        matrix,
        index=[class_names[label] for label in labels],
        columns=[class_names[label] for label in labels],
    ))
    return fit_seconds, accuracy


def main():
    train_path = DATA_DIR / f"train_{REPRESENTATION}.csv"
    test_path = DATA_DIR / f"test_{REPRESENTATION}.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Missing {train_path} or {test_path}; prepare the data first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feature_columns = [c for c in train.columns if c.startswith(("f", "sig_"))]
    test_feature_columns = [c for c in test.columns if c.startswith(("f", "sig_"))]
    if not feature_columns or feature_columns != test_feature_columns:
        raise ValueError("Train and test feature columns are missing or do not match.")
    if "activity_id" not in train or "activity_id" not in test:
        raise ValueError("Prepared CSVs must contain an activity_id column.")

    x_train, x_test = train[feature_columns], test[feature_columns]
    # UCI activity IDs are 1..6; XGBClassifier's sklearn wrapper expects 0..5.
    y_train = train["activity_id"].astype(int) - 1
    y_test = test["activity_id"].astype(int) - 1
    class_names = (
        train[["activity_id", "activity"]]
        .drop_duplicates()
        .assign(activity_id=lambda frame: frame["activity_id"].astype(int) - 1)
        .set_index("activity_id")["activity"]
        .to_dict()
    )

    baseline = make_classifier()
    full_seconds, full_accuracy = evaluate(
        "XGBoost using all input features",
        baseline,
        x_train,
        y_train,
        x_test,
        y_test,
        class_names,
    )

    selection_start = perf_counter()
    selector = ClusterGuidedFeatureSelector(N_SELECTED_FEATURES, RANDOM_STATE).fit(x_train)
    selection_seconds = perf_counter() - selection_start
    selected_columns = [feature_columns[i] for i in selector.selected_indices_]

    reduced_model = make_classifier()
    reduced_seconds, reduced_accuracy = evaluate(
        "XGBoost using K-means representative features",
        reduced_model,
        selector.transform(x_train),
        y_train,
        selector.transform(x_test),
        y_test,
        class_names,
    )
    fit_time_reduction = (
        100 * (full_seconds - reduced_seconds) / full_seconds if full_seconds else 0.0
    )

    print("\nComparison")
    print(f"Accuracy: {full_accuracy:.4%} -> {reduced_accuracy:.4%}")
    print(f"Selected features: {len(feature_columns)} -> {len(selected_columns)}")
    print(f"Classifier fit-time reduction: {fit_time_reduction:.2f}%")
    print(f"K-means selection preprocessing time: {selection_seconds:.2f} seconds")
    print("Fit-time reduction compares classifier fitting and excludes K-means setup.")

    joblib.dump(
        {
            "model": reduced_model,
            "selector": selector,
            "input_feature_columns": feature_columns,
            "selected_feature_columns": selected_columns,
            "class_names": class_names,
        },
        MODEL_OUT,
    )
    print(f"Saved reduced-feature model and selector: {MODEL_OUT}")


if __name__ == "__main__":
    main()
