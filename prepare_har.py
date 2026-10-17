"""Convert the raw HAR text files into labeled train/test CSV files."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("Raw Data")
OUTPUT_DIR = Path("Processed Data")
REPRESENTATION = "features"  # Choose "features" or "signals".

SIGNAL_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
METADATA_COLUMNS = ("subject", "activity_id", "activity")


def read_activity_names(data_dir):
    """Read the numeric activity ID to activity name mapping."""
    mapping = {}
    for line in (data_dir / "activity_labels.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            mapping[int(fields[0])] = fields[1]
    if not mapping:
        raise ValueError(f"No activity labels found in {data_dir / 'activity_labels.txt'}")
    return mapping


def read_feature_names(data_dir):
    """Read all 561 feature names, prefixing indices to keep headers unique."""
    names = []
    for line in (data_dir / "features.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            names.append(f"f{int(fields[0]):03d}_{fields[1]}")
    if not names:
        raise ValueError(f"No feature names found in {data_dir / 'features.txt'}")
    return names


def load_vector(path, dtype):
    """Load a one-dimensional label or subject file."""
    return np.loadtxt(path, dtype=dtype, ndmin=1).reshape(-1)


def load_matrix(path, expected_columns=None):
    """Load a numeric matrix and reject malformed or non-finite input."""
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ValueError(
            f"{path} has {values.shape[1]} columns; expected {expected_columns}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values.")
    return values


def load_split(data_dir, split, representation, activity_names, feature_names):
    """Load one official split and return a metadata-plus-features DataFrame."""
    split_dir = data_dir / split
    labels = load_vector(split_dir / f"y_{split}.txt", np.int64)
    subjects = load_vector(split_dir / f"subject_{split}.txt", np.int64)
    if labels.size != subjects.size:
        raise ValueError(f"{split}: label and subject row counts differ.")

    unknown_labels = set(np.unique(labels)) - set(activity_names)
    if unknown_labels:
        raise ValueError(f"{split}: unknown activity IDs: {sorted(unknown_labels)}")

    if representation == "features":
        values = load_matrix(
            split_dir / f"X_{split}.txt", expected_columns=len(feature_names)
        )
        feature_columns = feature_names
    elif representation == "signals":
        channels = []
        for channel_name in SIGNAL_CHANNELS:
            signal_path = (
                split_dir
                / "Inertial Signals"
                / f"{channel_name}_{split}.txt"
            )
            channel = load_matrix(signal_path, expected_columns=128)
            if channel.shape[0] != labels.size:
                raise ValueError(
                    f"{signal_path} has {channel.shape[0]} rows; expected {labels.size}."
                )
            channels.append(channel)
        values = np.concatenate(channels, axis=1)
        feature_columns = [
            f"sig_{channel}_t{step:03d}"
            for channel in SIGNAL_CHANNELS
            for step in range(1, 129)
        ]
    else:
        raise ValueError("REPRESENTATION must be either 'features' or 'signals'.")

    if values.shape[0] != labels.size:
        raise ValueError(f"{split}: feature and label row counts differ.")

    frame = pd.DataFrame(values, columns=feature_columns)
    frame.insert(0, "activity", [activity_names[label] for label in labels])
    frame.insert(0, "activity_id", labels)
    frame.insert(0, "subject", subjects)
    return frame


def main():
    if REPRESENTATION not in {"features", "signals"}:
        raise ValueError("REPRESENTATION must be either 'features' or 'signals'.")

    activity_names = read_activity_names(DATA_DIR)
    feature_names = read_feature_names(DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        frame = load_split(
            DATA_DIR,
            split,
            REPRESENTATION,
            activity_names,
            feature_names,
        )
        output_path = OUTPUT_DIR / f"{split}_{REPRESENTATION}.csv"
        frame.to_csv(output_path, index=False)
        print(
            f"Wrote {len(frame):,} rows and "
            f"{len(frame.columns) - len(METADATA_COLUMNS):,} features: {output_path}"
        )


if __name__ == "__main__":
    main()
