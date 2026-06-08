import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_io import read_ivecs  # noqa: E402


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_recall(result_ids, gt_ids, k=10):
    return len(set(result_ids[:k]) & set(gt_ids[:k])) / float(k)


def load_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((224, 224))
    return image


def plot_image_row(axes, title, image_paths):
    axes[0].set_ylabel(title, rotation=0, labelpad=52, fontsize=9, va="center")
    for i, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i < len(image_paths):
            ax.imshow(load_image(image_paths[i]))
        else:
            ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--query-paths", default="outputs/oxford_pet_query_path.csv")
    parser.add_argument("--base-paths", default="outputs/oxford_pet_base_path.csv")
    parser.add_argument("--groundtruth", default="outputs/oxford_pet_groundtruth.ivecs")
    parser.add_argument("--ivf-results", default="outputs/ivfflat_results.ivecs")
    parser.add_argument("--hnsw-results", default="outputs/hnsw_results.ivecs")
    parser.add_argument("--output", default="outputs/query0_visualization.png")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    query_rows = read_rows(args.query_paths)
    base_rows = read_rows(args.base_paths)
    gt = read_ivecs(args.groundtruth)
    ivf = read_ivecs(args.ivf_results)
    hnsw = read_ivecs(args.hnsw_results)

    qid = args.query_id
    if qid < 0 or qid >= len(query_rows):
        raise ValueError("query-id is out of range")

    gt_ids = [int(x) for x in gt[qid][: args.topk]]
    ivf_ids = [int(x) for x in ivf[qid][: args.topk]]
    hnsw_ids = [int(x) for x in hnsw[qid][: args.topk]]

    query_path = query_rows[qid]["path"]
    rows = [
        ("Query", [query_path]),
        ("Ground Truth Top-10", [base_rows[i]["path"] for i in gt_ids]),
        (
            f"IVFFlat Top-10 | R@10={row_recall(ivf_ids, gt_ids):.2f}",
            [base_rows[i]["path"] for i in ivf_ids],
        ),
        (
            f"HNSW Top-10 | R@10={row_recall(hnsw_ids, gt_ids):.2f}",
            [base_rows[i]["path"] for i in hnsw_ids],
        ),
    ]

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=args.topk,
        figsize=(args.topk * 1.35, len(rows) * 1.55),
    )
    if len(rows) == 1:
        axes = [axes]

    for row_axes, (title, paths) in zip(axes, rows):
        plot_image_row(row_axes, title, paths)

    fig.suptitle(f"Query {qid}: {Path(query_path).name}", fontsize=12)
    plt.tight_layout(rect=(0.04, 0, 1, 0.94))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"visualization saved: {output}")


if __name__ == "__main__":
    main()
