from pathlib import Path

import numpy as np


def read_fvecs(path):
    """Read an fvecs file into a float32 matrix with shape [n, d]."""
    path = Path(path)
    raw = np.fromfile(path, dtype="float32")
    if raw.size == 0:
        raise RuntimeError(f"empty fvecs file: {path}")

    dim = raw.view("int32")[0]
    if dim <= 0:
        raise RuntimeError(f"invalid vector dimension in {path}: {dim}")
    if raw.size % (dim + 1) != 0:
        raise RuntimeError(f"invalid fvecs file format: {path}")

    raw = raw.reshape(-1, dim + 1)
    dims = raw[:, 0].view("int32")
    if not np.all(dims == dim):
        raise RuntimeError(f"inconsistent vector dimensions in {path}")

    return raw[:, 1:].astype("float32", copy=True)


def write_fvecs(path, vectors):
    """Write a float32 matrix with shape [n, d] to fvecs format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")

    n, dim = vectors.shape
    with path.open("wb") as f:
        for i in range(n):
            np.array([dim], dtype="int32").tofile(f)
            vectors[i].astype("float32", copy=False).tofile(f)


def read_ivecs(path):
    """Read an ivecs file into an int32 matrix with shape [n, k]."""
    path = Path(path)
    raw = np.fromfile(path, dtype="int32")
    if raw.size == 0:
        raise RuntimeError(f"empty ivecs file: {path}")

    k = int(raw[0])
    if k <= 0:
        raise RuntimeError(f"invalid ivecs row length in {path}: {k}")
    if raw.size % (k + 1) != 0:
        raise RuntimeError(f"invalid ivecs file format: {path}")

    raw = raw.reshape(-1, k + 1)
    row_lengths = raw[:, 0]
    if not np.all(row_lengths == k):
        raise RuntimeError(f"inconsistent ivecs row lengths in {path}")

    return raw[:, 1:].astype("int32", copy=True)


def write_ivecs(path, indices):
    """Write an int32 matrix with shape [n, k] to ivecs format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.asarray(indices, dtype="int32")
    if indices.ndim != 2:
        raise ValueError("indices must be a 2D array")

    n, k = indices.shape
    with path.open("wb") as f:
        for i in range(n):
            np.array([k], dtype="int32").tofile(f)
            indices[i].astype("int32", copy=False).tofile(f)
