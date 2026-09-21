from __future__ import annotations

import argparse
import json
from pathlib import Path

from uiregress.data.pairs import PairedScreenshotDataset, build_label_mapping
from uiregress.training import _classical_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Milestone 3.2 classical baselines using validation-only threshold "
            "calibration, evaluate on the frozen test split, and update benchmark.json."
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/generated/synthetic-v0.2",
        help="Dataset root containing dataset.json and manifest.jsonl.",
    )
    parser.add_argument(
        "--benchmark",
        default="artifacts/checkpoints/milestone3-v0.2-best.benchmark.json",
        help="Existing Milestone 3.2 benchmark JSON to update.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("auto", "cpu", "cuda"),
        help="Device used by the classical image-comparison implementation.",
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.90,
        help="Minimum validation recall used when calibrating baseline thresholds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.minimum_recall <= 1.0:
        raise ValueError("--minimum-recall must be between 0 and 1.")

    dataset_root = Path(args.dataset)
    benchmark_path = Path(args.benchmark)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Benchmark JSON not found: {benchmark_path}\n"
            "Run the Milestone 3.2 training command first."
        )

    label_mapping = build_label_mapping(dataset_root)

    validation_dataset = PairedScreenshotDataset(
        dataset_root,
        split="validation",
        label_to_index=label_mapping,
    )
    test_dataset = PairedScreenshotDataset(
        dataset_root,
        split="test",
        label_to_index=label_mapping,
    )

    print(
        f"Running classical benchmark: "
        f"validation={len(validation_dataset)}, test={len(test_dataset)}, "
        f"device={args.device}, minimum_recall={args.minimum_recall:.2f}"
    )

    classical_results = _classical_benchmark(
        validation_dataset,
        test_dataset,
        device=args.device,
        minimum_recall=args.minimum_recall,
    )

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["classical_baselines"] = classical_results

    benchmark_path.write_text(
        json.dumps(benchmark, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nClassical baseline results:")
    print(json.dumps(classical_results, indent=2))
    print(f"\nUpdated benchmark: {benchmark_path.resolve()}")


if __name__ == "__main__":
    main()
