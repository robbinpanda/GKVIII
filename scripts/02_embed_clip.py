import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_io import write_fvecs  # noqa: E402


def read_metadata(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_image(path):
    return Image.open(path).convert("RGB")


def write_path_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vector_id",
                "image_id",
                "path",
                "class_id",
                "class_name",
                "species",
                "breed_id",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out["vector_id"] = i
            writer.writerow(out)


def embed_images(args):
    rows = read_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("metadata has no images")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model: {args.model_name}")
    print(f"image count: {len(rows)}")

    processor = CLIPImageProcessor.from_pretrained(args.model_name)
    model = CLIPVisionModel.from_pretrained(args.model_name).to(device)
    model.eval()

    all_vectors = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="embedding"):
        batch_rows = rows[start : start + args.batch_size]
        images = [load_image(row["path"]) for row in batch_rows]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            emb = outputs.pooler_output
            if args.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_vectors.append(emb.cpu().numpy().astype("float32"))

    vectors = np.concatenate(all_vectors, axis=0)
    write_fvecs(args.output_fvecs, vectors)
    write_path_csv(args.output_paths, rows)
    print(f"vectors saved: {args.output_fvecs}")
    print(f"path mapping saved: {args.output_paths}")
    print(f"shape: {vectors.shape}, dtype: {vectors.dtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/oxford_pet_metadata.csv")
    parser.add_argument("--output-fvecs", default="outputs/oxford_pet_all.fvecs")
    parser.add_argument("--output-paths", default="outputs/oxford_pet_all_path.csv")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    embed_images(args)


if __name__ == "__main__":
    main()
