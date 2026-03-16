"""
Data utilities — dataset loading and preprocessing helpers.
"""
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def load_dataset(name: str):
    """
    Load a sklearn built-in dataset by name.
    Returns: X (array), y (array), feature_names (list), target_names (list)
    """
    loaders = {
        "iris": datasets.load_iris,
        "breast_cancer": datasets.load_breast_cancer,
        "wine": datasets.load_wine,
        "diabetes": datasets.load_diabetes,
        "digits": datasets.load_digits,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}")

    data = loaders[name]()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else [f"f{i}" for i in range(X.shape[1])]
    target_names = list(data.target_names) if hasattr(data, "target_names") else ["target"]
    return X, y, feature_names, target_names


def load_synthetic(kind: str = "moons", n_samples: int = 300, noise: float = 0.1, random_state: int = 42):
    """Generate synthetic 2D datasets for visualization demos."""
    generators = {
        "moons": lambda: datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state),
        "circles": lambda: datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state),
        "blobs": lambda: datasets.make_blobs(n_samples=n_samples, centers=3, random_state=random_state),
        "classification": lambda: datasets.make_classification(n_samples=n_samples, n_features=2, n_redundant=0, random_state=random_state),
    }
    if kind not in generators:
        raise ValueError(f"Unknown synthetic dataset: {kind}")
    X, y = generators[kind]()
    return X, y


def split_and_scale(X, y, test_size: float = 0.2, scale: str = "standard", random_state: int = 42):
    """Split into train/test and optionally scale features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = None
    if scale == "standard":
        scaler = StandardScaler()
    elif scale == "minmax":
        scaler = MinMaxScaler()

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def dataset_info(name: str) -> str:
    """Return a markdown summary of dataset properties."""
    X, y, feature_names, target_names = load_dataset(name)
    n_classes = len(np.unique(y))
    info = f"""**Dataset:** `{name}`
**Samples:** {X.shape[0]} &nbsp;|&nbsp; **Features:** {X.shape[1]} &nbsp;|&nbsp; **Classes/Targets:** {n_classes}
**Features:** {", ".join(feature_names[:6])}{"..." if len(feature_names) > 6 else ""}
"""
    return info


def to_dataframe(X, y, feature_names=None, target_name="target") -> pd.DataFrame:
    """Convert numpy arrays to a pandas DataFrame."""
    cols = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df[target_name] = y
    return df
