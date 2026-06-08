import argparse
import csv
import random
import sys
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_io import read_fvecs, write_fvecs, write_ivecs  # noqa: E402


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rewrite_vector_ids(rows):
    out = []
    for i, row in enumerate(rows):
        next_row = dict(row)
        next_row["vector_id"] = i
        out.append(next_row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-fvecs", default="outputs/oxford_pet_all.fvecs")
    parser.add_argument("--all-paths", default="outputs/oxford_pet_all_path.csv")
    parser.add_argument("--base-fvecs", default="outputs/oxford_pet_base.fvecs")
    parser.add_argument("--query-fvecs", default="outputs/oxford_pet_query.fvecs")
    parser.add_argument("--base-paths", default="outputs/oxford_pet_base_path.csv")
    parser.add_argument("--query-paths", default="outputs/oxford_pet_query_path.csv")
    parser.add_argument("--groundtruth", default="outputs/oxford_pet_groundtruth.ivecs")
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    vectors = read_fvecs(args.all_fvecs).astype("float32")
    rows = read_rows(args.all_paths)
    if len(rows) != len(vectors):
        raise RuntimeError("path rows and vector count do not match")
    if args.query_size <= 0 or args.query_size >= len(vectors):
        raise ValueError("query-size must be between 1 and vector_count - 1")

    rng = random.Random(args.seed)
    query_ids = sorted(rng.sample(range(len(vectors)), args.query_size))
    query_id_set = set(query_ids)
    base_ids = [i for i in range(len(vectors)) if i not in query_id_set]

    query_vectors = vectors[query_ids]
    base_vectors = vectors[base_ids]
    query_rows = rewrite_vector_ids([rows[i] for i in query_ids])
    base_rows = rewrite_vector_ids([rows[i] for i in base_ids])

    write_fvecs(args.query_fvecs, query_vectors)
    write_fvecs(args.base_fvecs, base_vectors)
    write_rows(args.query_paths, query_rows)
    write_rows(args.base_paths, base_rows)

    if args.topk > len(base_vectors):
        raise ValueError("topk cannot be larger than base vector count")

    index = faiss.IndexFlatL2(base_vectors.shape[1])
    index.add(base_vectors)
    _, gt_indices = index.search(query_vectors, args.topk)
    write_ivecs(args.groundtruth, gt_indices.astype("int32"))

    print(f"base vectors: {base_vectors.shape}")
    print(f"query vectors: {query_vectors.shape}")
    print(f"ground truth saved: {args.groundtruth}")


if __name__ == "__main__":
    main()
