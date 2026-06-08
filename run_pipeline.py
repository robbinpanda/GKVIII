import argparse
import subprocess
import sys


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    py = sys.executable

    prepare_cmd = [py, "scripts/01_prepare_oxford_pet.py"]
    if args.limit is not None:
        prepare_cmd.extend(["--limit", str(args.limit)])
    run(prepare_cmd)

    embed_cmd = [
        py,
        "scripts/02_embed_clip.py",
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    run(embed_cmd)

    run(
        [
            py,
            "scripts/03_split_and_groundtruth.py",
            "--query-size",
            str(args.query_size),
        ]
    )

    run([py, "scripts/04_run_faiss_experiments.py"])

    run(
        [
            py,
            "scripts/05_visualize_results.py",
            "--query-id",
            str(args.query_id),
            "--output",
            f"outputs/query{args.query_id}_visualization.png",
        ]
    )


if __name__ == "__main__":
    main()
