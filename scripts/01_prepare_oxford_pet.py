import argparse
import csv
import random
import tarfile
import urllib.request
from pathlib import Path


IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def download(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"skip existing file: {output_path}")
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, output_path)


def extract(archive_path, output_dir):
    marker = output_dir / f".extracted_{archive_path.stem}"
    if marker.exists():
        print(f"skip extracted archive: {archive_path.name}")
        return
    print(f"extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir)
    marker.write_text("ok", encoding="utf-8")


def read_labels(list_path):
    labels = {}
    class_names = {}
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, class_id, species, breed_id = line.split()
            class_id = int(class_id)
            species = int(species)
            breed_id = int(breed_id)
            class_name = image_id.rsplit("_", 1)[0]
            labels[image_id] = {
                "class_id": class_id,
                "class_name": class_name,
                "species": species,
                "breed_id": breed_id,
            }
            class_names[class_id] = class_name
    return labels, class_names


def build_metadata(data_dir, output_csv, limit=None, seed=42):
    image_dir = data_dir / "images"
    labels, _ = read_labels(data_dir / "annotations" / "list.txt")

    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = image_path.stem
        label = labels.get(image_id)
        if label is None:
            continue
        rows.append(
            {
                "image_id": image_id,
                "path": str(image_path),
                "class_id": label["class_id"],
                "class_name": label["class_name"],
                "species": label["species"],
                "breed_id": label["breed_id"],
            }
        )

    if limit is not None and limit < len(rows):
        rng = random.Random(seed)
        rows = sorted(rng.sample(rows, limit), key=lambda x: x["image_id"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "path",
                "class_id",
                "class_name",
                "species",
                "breed_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"metadata saved: {output_csv}")
    print(f"image count: {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/oxford_pet")
    parser.add_argument("--metadata", default="outputs/oxford_pet_metadata.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    archives_dir = data_dir / "archives"
    images_tar = archives_dir / "images.tar.gz"
    annotations_tar = archives_dir / "annotations.tar.gz"

    download(IMAGES_URL, images_tar)
    download(ANNOTATIONS_URL, annotations_tar)
    extract(images_tar, data_dir)
    extract(annotations_tar, data_dir)
    build_metadata(data_dir, Path(args.metadata), limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
