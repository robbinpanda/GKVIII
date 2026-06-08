import argparse
import csv
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_io import read_fvecs, write_fvecs  # noqa: E402


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_image(path):
    return Image.open(path).convert("RGB")


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def embed_images(args, rows, processor, model, device):
    cache_path = Path(args.image_fvecs)
    if cache_path.exists() and not args.rebuild_image_vectors:
        vectors = read_fvecs(cache_path).astype("float32")
        if len(vectors) != len(rows):
            raise RuntimeError("cached image vector count does not match base paths")
        print(f"reuse image vectors: {cache_path}")
        return vectors

    all_vectors = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="image features"):
        batch_rows = rows[start : start + args.batch_size]
        images = [load_image(row["path"]) for row in batch_rows]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            features = model.get_image_features(pixel_values=pixel_values)
            if args.normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
        all_vectors.append(features.cpu().numpy().astype("float32"))

    vectors = np.concatenate(all_vectors, axis=0)
    write_fvecs(cache_path, vectors)
    print(f"image vectors saved: {cache_path}")
    return vectors


def embed_texts(args, processor, model, device):
    inputs = processor(text=args.queries, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        if args.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().astype("float32")


def write_results_csv(path, query_texts, indices, distances, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "rank",
        "base_vector_id",
        "distance",
        "path",
        "class_name",
        "species",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid, query in enumerate(query_texts):
            for rank, base_id in enumerate(indices[qid], start=1):
                row = rows[int(base_id)]
                writer.writerow(
                    {
                        "query_id": qid,
                        "query_text": query,
                        "rank": rank,
                        "base_vector_id": int(base_id),
                        "distance": float(distances[qid][rank - 1]),
                        "path": row["path"],
                        "class_name": row.get("class_name", ""),
                        "species": row.get("species", ""),
                    }
                )


def visualize(path, query_texts, indices, rows, topk):
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(query_texts),
        ncols=topk,
        figsize=(topk * 1.35, max(1, len(query_texts)) * 1.55),
    )
    if len(query_texts) == 1:
        axes = np.asarray([axes])

    for qid, row_axes in enumerate(axes):
        row_axes[0].set_ylabel(f"Q{qid}", rotation=0, labelpad=24, fontsize=9, va="center")
        for rank, ax in enumerate(row_axes):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            base_id = int(indices[qid][rank])
            image = load_image(rows[base_id]["path"])
            image.thumbnail((224, 224))
            ax.imshow(image)

    fig.suptitle("CLIP text-to-image search", fontsize=12)
    plt.tight_layout(rect=(0.02, 0, 1, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"visualization saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-paths", default="outputs/oxford_pet_base_path.csv")
    parser.add_argument("--image-fvecs", default="outputs/bonus2_clip_base_text_image.fvecs")
    parser.add_argument("--results-csv", default="outputs/bonus2_text_search_results.csv")
    parser.add_argument("--visualization", default="outputs/bonus2_text_search.png")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--queries", nargs="+", default=["a black dog", "a cat lying on grass"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-image-vectors", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"model: {args.model_name}")

    rows = read_rows(args.base_paths)
    if not rows:
        raise RuntimeError("base paths file has no images")

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()

    image_vectors = embed_images(args, rows, processor, model, device)
    text_vectors = embed_texts(args, processor, model, device)

    if image_vectors.shape[1] != text_vectors.shape[1]:
        raise RuntimeError("image and text vector dimensions do not match")

    index = faiss.IndexFlatL2(image_vectors.shape[1])
    index.add(image_vectors)
    distances, indices = index.search(text_vectors, args.topk)

    write_results_csv(args.results_csv, args.queries, indices, distances, rows)
    visualize(args.visualization, args.queries, indices, rows, args.topk)
    print(f"results saved: {args.results_csv}")


if __name__ == "__main__":
    main()
