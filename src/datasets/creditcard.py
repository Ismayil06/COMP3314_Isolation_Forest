from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"


def load_creditcard(path=DEFAULT_PATH):
    """Load the Kaggle credit card fraud data.

    Returns X (n, 30) float array: Time, V1-V28, Amount
    and y (n,) int array: 1 = fraud, 0 = genuine.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found."
        )
    df = pd.read_csv(path)
    y = df["Class"].to_numpy(dtype=int)
    X = df.drop(columns="Class").to_numpy(dtype=float)
    return X, y
