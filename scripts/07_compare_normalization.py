import argparse
import csv
import random
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_io import read_fvecs  # noqa: E402


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_ids(count, query_size, seed):
    rng = random.Random(seed)
    query_ids = sorted(rng.sample(range(count), query_size))
    query_id_set = set(query_ids)
    base_ids = [i for i in range(count) if i not in query_id_set]
    return query_ids, base_ids


def search_topk(all_vectors, query_ids, base_ids, topk):
    query_vectors = all_vectors[query_ids].astype("float32")
    base_vectors = all_vectors[base_ids].astype("float32")
    index = faiss.IndexFlatL2(base_vectors.shape[1])
    index.add(base_vectors)
    distances, indices = index.search(query_vectors, topk)
    return distances, indices


def overlap_at_k(left, right, k):
    return len(set(int(x) for x in left[:k]) & set(int(x) for x in right[:k])) / float(k)


def write_summary(path, normalized_indices, raw_indices, topk):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "overlap_at_10", "normalized_top10", "no_normalize_top10"],
        )
        writer.writeheader()
        for qid, (norm_ids, raw_ids) in enumerate(zip(normalized_indices, raw_indices)):
            writer.writerow(
                {
                    "query_id": qid,
                    "overlap_at_10": overlap_at_k(norm_ids, raw_ids, topk),
                    "normalized_top10": " ".join(str(int(x)) for x in norm_ids[:topk]),
                    "no_normalize_top10": " ".join(str(int(x)) for x in raw_ids[:topk]),
                }
            )


def load_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((224, 224))
    return image


def visualize(path, rows, query_ids, base_ids, query_id, normalized_indices, raw_indices, topk):
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Query", "Normalized L2 Top-10", "No-normalize L2 Top-10"]
    image_rows = [
        [rows[query_ids[query_id]]["path"]],
        [rows[base_ids[int(i)]]["path"] for i in normalized_indices[query_id][:topk]],
        [rows[base_ids[int(i)]]["path"] for i in raw_indices[query_id][:topk]],
    ]

    fig, axes = plt.subplots(nrows=3, ncols=topk, figsize=(topk * 1.35, 4.65))
    for row_axes, label, image_paths in zip(axes, labels, image_rows):
        row_axes[0].set_ylabel(label, rotation=0, labelpad=58, fontsize=9, va="center")
        for i, ax in enumerate(row_axes):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i < len(image_paths):
                ax.imshow(load_image(image_paths[i]))
            else:
                ax.axis("off")

    overlap = overlap_at_k(normalized_indices[query_id], raw_indices[query_id], topk)
    fig.suptitle(f"Normalization comparison | query {query_id} | overlap@10={overlap:.2f}", fontsize=12)
    plt.tight_layout(rect=(0.04, 0, 1, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"visualization saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-fvecs", default="outputs/oxford_pet_all.fvecs")
    parser.add_argument("--raw-fvecs", default="outputs/bonus3_oxford_pet_all_no_norm.fvecs")
    parser.add_argument("--paths", default="outputs/oxford_pet_all_path.csv")
    parser.add_argument("--output-csv", default="outputs/bonus3_normalization_compare.csv")
    parser.add_argument("--visualization", default="outputs/bonus3_query0_compare.png")
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    normalized = read_fvecs(args.normalized_fvecs).astype("float32")
    raw = read_fvecs(args.raw_fvecs).astype("float32")
    rows = read_rows(args.paths)

    if normalized.shape != raw.shape:
        raise RuntimeError("normalized and no-normalize vector shapes do not match")
    if len(rows) != len(normalized):
        raise RuntimeError("path rows and vector count do not match")
    if args.query_size <= 0 or args.query_size >= len(normalized):
        raise ValueError("query-size must be between 1 and vector_count - 1")
    if args.query_id < 0 or args.query_id >= args.query_size:
        raise ValueError("query-id is out of range")

    query_ids, base_ids = split_ids(len(normalized), args.query_size, args.seed)
    _, normalized_indices = search_topk(normalized, query_ids, base_ids, args.topk)
    _, raw_indices = search_topk(raw, query_ids, base_ids, args.topk)

    overlaps = [
        overlap_at_k(norm_ids, raw_ids, args.topk)
        for norm_ids, raw_ids in zip(normalized_indices, raw_indices)
    ]
    print(f"mean overlap@{args.topk}: {float(np.mean(overlaps)):.4f}")

    write_summary(args.output_csv, normalized_indices, raw_indices, args.topk)
    visualize(
        args.visualization,
        rows,
        query_ids,
        base_ids,
        args.query_id,
        normalized_indices,
        raw_indices,
        args.topk,
    )
    print(f"summary saved: {args.output_csv}")


if __name__ == "__main__":
    main()
