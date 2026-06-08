import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_io import read_fvecs, read_ivecs, write_ivecs  # noqa: E402


def mean_recall_at_k(results, groundtruth, k=10):
    values = []
    for result_ids, gt_ids in zip(results, groundtruth):
        result_set = set(int(x) for x in result_ids[:k])
        gt_set = set(int(x) for x in gt_ids[:k])
        values.append(len(result_set & gt_set) / float(k))
    return float(np.mean(values))


def index_size_mb(index, index_path):
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index_path.stat().st_size / 1024 / 1024


def search_and_measure(index, query_vectors, topk):
    t0 = time.perf_counter()
    distances, indices = index.search(query_vectors, topk)
    elapsed = time.perf_counter() - t0
    qps = len(query_vectors) / elapsed if elapsed > 0 else float("inf")
    return distances, indices, elapsed, qps


def run_ivf(base_vectors, query_vectors, groundtruth, args):
    dim = base_vectors.shape[1]
    nlist = min(args.ivf_nlist, max(1, len(base_vectors) // 20))
    nprobe = min(args.ivf_nprobe, nlist)

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    t0 = time.perf_counter()
    index.train(base_vectors)
    index.add(base_vectors)
    build_ms = (time.perf_counter() - t0) * 1000
    index.nprobe = nprobe

    out_index = Path(args.index_dir) / "ivfflat.index"
    size_mb = index_size_mb(index, out_index)
    _, indices, search_sec, qps = search_and_measure(index, query_vectors, args.topk)
    out_results = Path(args.output_dir) / "ivfflat_results.ivecs"
    write_ivecs(out_results, indices.astype("int32"))

    return {
        "index_method": "IndexIVFFlat",
        "index_params": f"nlist={nlist}, nprobe={nprobe}",
        "build_time_ms": build_ms,
        "index_size_mb": size_mb,
        "search_time_sec": search_sec,
        "qps": qps,
        "recall_at_10": mean_recall_at_k(indices, groundtruth, args.topk),
        "results_ivecs": str(out_results),
        "index_path": str(out_index),
    }


def run_hnsw(base_vectors, query_vectors, groundtruth, args):
    dim = base_vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_L2)
    index.hnsw.efConstruction = args.hnsw_ef_construction

    t0 = time.perf_counter()
    index.add(base_vectors)
    build_ms = (time.perf_counter() - t0) * 1000
    index.hnsw.efSearch = args.hnsw_ef_search

    out_index = Path(args.index_dir) / "hnsw.index"
    size_mb = index_size_mb(index, out_index)
    _, indices, search_sec, qps = search_and_measure(index, query_vectors, args.topk)
    out_results = Path(args.output_dir) / "hnsw_results.ivecs"
    write_ivecs(out_results, indices.astype("int32"))

    return {
        "index_method": "IndexHNSWFlat",
        "index_params": (
            f"M={args.hnsw_m}, "
            f"efConstruction={args.hnsw_ef_construction}, "
            f"efSearch={args.hnsw_ef_search}"
        ),
        "build_time_ms": build_ms,
        "index_size_mb": size_mb,
        "search_time_sec": search_sec,
        "qps": qps,
        "recall_at_10": mean_recall_at_k(indices, groundtruth, args.topk),
        "results_ivecs": str(out_results),
        "index_path": str(out_index),
    }


def write_metrics_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "embedding_model",
        "vector_dim",
        "dtype",
        "base_count",
        "base_vector_size_mb",
        "query_count",
        "index_method",
        "index_params",
        "build_time_ms",
        "index_size_mb",
        "qps",
        "recall_at_10",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fvecs", default="outputs/oxford_pet_base.fvecs")
    parser.add_argument("--query-fvecs", default="outputs/oxford_pet_query.fvecs")
    parser.add_argument("--groundtruth", default="outputs/oxford_pet_groundtruth.ivecs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--index-dir", default="indices")
    parser.add_argument("--metrics-csv", default="outputs/metrics.csv")
    parser.add_argument("--metrics-json", default="outputs/metrics.json")
    parser.add_argument("--dataset", default="Oxford-IIIT Pet")
    parser.add_argument("--embedding-model", default="CLIP ViT-B/32")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ivf-nlist", type=int, default=100)
    parser.add_argument("--ivf-nprobe", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    base_vectors = read_fvecs(args.base_fvecs).astype("float32")
    query_vectors = read_fvecs(args.query_fvecs).astype("float32")
    groundtruth = read_ivecs(args.groundtruth).astype("int32")

    if base_vectors.shape[1] != query_vectors.shape[1]:
        raise RuntimeError("base and query vector dimensions do not match")
    if len(query_vectors) != len(groundtruth):
        raise RuntimeError("query count and ground truth row count do not match")

    rows = []
    for run_fn in (run_ivf, run_hnsw):
        result = run_fn(base_vectors, query_vectors, groundtruth, args)
        result.update(
            {
                "dataset": args.dataset,
                "embedding_model": args.embedding_model,
                "vector_dim": base_vectors.shape[1],
                "dtype": str(base_vectors.dtype),
                "base_count": len(base_vectors),
                "base_vector_size_mb": base_vectors.nbytes / 1024 / 1024,
                "query_count": len(query_vectors),
            }
        )
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    write_metrics_csv(args.metrics_csv, rows)
    Path(args.metrics_json).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"metrics saved: {args.metrics_csv}")


if __name__ == "__main__":
    main()
