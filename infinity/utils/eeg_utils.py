from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

def standard_scale_features(X, scaler=None, return_scaler=False):
    """Scale features with ``StandardScaler``."""
    orig_shape = X.shape[1:]
    X_2d = X.reshape(len(X), -1)

    if scaler is None:
        scaler = StandardScaler().fit(X_2d)

    X_scaled = scaler.transform(X_2d).reshape((len(X),) + orig_shape)

    if return_scaler:
        return X_scaled, scaler
    return X_scaled


def compute_raw_stats(X: np.ndarray):
    """Compute per-channel mean and std from training data."""
    # Assuming X shape (N, C, T)
    # Mean/Std over N and T (time), keeping Channels (C) distinct? 
    # Original code: mean = X.mean(axis=(0, 2))
    # If X is (N, C, T), axis (0, 2) is N and T. This computes stats per channel.
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2)) + 1e-6
    return mean, std


def normalize_raw(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize raw EEG with provided statistics."""
    # X: (N, C, T), mean: (C,), std: (C,)
    # Broadcasting: (N, C, T) - (1, C, 1) / (1, C, 1)
    return (X - mean[None, :, None]) / std[None, :, None]


def load_scaler(path: str) -> StandardScaler:
    """Load a ``StandardScaler`` object from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_raw_stats(path: str):
    """Load raw EEG normalization statistics from a ``.npz`` file."""
    data = np.load(path)
    return data["mean"], data["std"]
