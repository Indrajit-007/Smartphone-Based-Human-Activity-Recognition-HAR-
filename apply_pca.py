"""Standardize prepared HAR features and transform them with PCA."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT_DIR = Path("Processed Data")
OUTPUT_DIR = Path("PCA Transformed Data")
REPRESENTATIONS = ("features", "signals")  # Use one or both prepared representations.
N_COMPONENTS = 0.95  # Keep 95% of training variance; can also be an integer such as 50.
METADATA_COLUMNS = ["subject", "activity_id", "activity"]


def transform_representation(representation):
    """Fit scaling/PCA on the training split and reuse both on the test split."""
    train_path = INPUT_DIR / f"train_{representation}.csv"
    test_path = INPUT_DIR / f"test_{representation}.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Prepared CSVs for '{representation}' are missing. "
            "Run prepare_har.py first or adjust REPRESENTATIONS."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    missing_metadata = [
        column for column in METADATA_COLUMNS
        if column not in train.columns or column not in test.columns
    ]
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {missing_metadata}")

    feature_columns = [
        column for column in train.columns if column not in METADATA_COLUMNS
    ]
    test_feature_columns = [
        column for column in test.columns if column not in METADATA_COLUMNS
    ]
    if not feature_columns or feature_columns != test_feature_columns:
        raise ValueError(f"{representation}: train and test feature columns do not match.")

    # Fit both transforms using training data only to avoid test-set leakage.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[feature_columns])
    test_scaled = scaler.transform(test[feature_columns])

    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
    train_components = pca.fit_transform(train_scaled)
    test_components = pca.transform(test_scaled)
    component_columns = [
        f"PC{index:03d}" for index in range(1, pca.n_components_ + 1)
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, source, values in (
        ("train", train, train_components),
        ("test", test, test_components),
    ):
        transformed = pd.DataFrame(
            values,
            columns=component_columns,
            index=source.index,
        )
        result = pd.concat(
            [source[METADATA_COLUMNS].copy(), transformed],
            axis=1,
        )
        output_path = OUTPUT_DIR / f"{split}_{representation}_pca.csv"
        result.to_csv(output_path, index=False)
        print(
            f"Wrote {len(result):,} rows and {len(component_columns):,} "
            f"principal components: {output_path}"
        )

    model_path = OUTPUT_DIR / f"pca_{representation}.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "input_feature_columns": feature_columns,
            "metadata_columns": METADATA_COLUMNS,
            "component_columns": component_columns,
        },
        model_path,
    )
    print(
        f"Saved PCA transform to {model_path}; "
        f"explained training variance: {pca.explained_variance_ratio_.sum():.2%}"
    )


def main():
    if not REPRESENTATIONS:
        raise ValueError("Set REPRESENTATIONS to 'features', 'signals', or both.")
    for representation in REPRESENTATIONS:
        if representation not in {"features", "signals"}:
            raise ValueError(
                f"Unknown representation '{representation}'; use 'features' or 'signals'."
            )
        transform_representation(representation)


if __name__ == "__main__":
    main()
