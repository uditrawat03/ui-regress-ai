from __future__ import annotations

import json
import random
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from uiregress.data.pairs import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NO_REGRESSION_LABEL,
    PREPROCESSING_VERSION,
    PairedScreenshotDataset,
    build_label_mapping,
    load_dataset_metadata,
)
from uiregress.device import resolve_device
from uiregress.inference import compare_images
from uiregress.models import SiameseRegressionClassifier

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_root: str
    checkpoint_path: str
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    input_size: int = 224
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    pretrained: bool = True
    minimum_binary_recall: float = 0.90
    classical_benchmark: bool = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _binary_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions, strict=True))
    total = max(1, len(targets))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(1, fp + tn),
    }


def _multiclass_metrics(
    targets: list[int],
    predictions: list[int],
    *,
    index_to_label: dict[int, str],
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    ordered_classes = sorted(index_to_label)
    confusion_matrix = [
        [0 for _ in ordered_classes]
        for _ in ordered_classes
    ]

    for target, prediction in zip(targets, predictions, strict=True):
        if target in index_to_label and prediction in index_to_label:
            confusion_matrix[target][prediction] += 1

    for class_index, label in sorted(index_to_label.items()):
        tp = sum(
            t == class_index and p == class_index
            for t, p in zip(targets, predictions, strict=True)
        )
        fp = sum(
            t != class_index and p == class_index
            for t, p in zip(targets, predictions, strict=True)
        )
        fn = sum(
            t == class_index and p != class_index
            for t, p in zip(targets, predictions, strict=True)
        )
        support = sum(t == class_index for t in targets)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = sum(t == p for t, p in zip(targets, predictions, strict=True)) / max(
        1, len(targets)
    )
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / max(1, len(f1_values)),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [index_to_label[index] for index in ordered_classes],
            "matrix": confusion_matrix,
        },
    }


def _threshold_predictions(scores: list[float], threshold: float) -> list[int]:
    return [int(score >= threshold) for score in scores]


def _calibrate_threshold(
    targets: list[int],
    scores: list[float],
    *,
    minimum_recall: float,
) -> dict[str, Any]:
    if len(targets) != len(scores):
        raise ValueError("targets and scores must contain the same number of values")
    if not targets:
        raise ValueError("threshold calibration requires at least one sample")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")

    unique_scores = sorted({float(score) for score in scores})
    spread = max(unique_scores) - min(unique_scores)
    epsilon = max(1e-12, spread * 1e-9)
    thresholds = [unique_scores[0] - epsilon, *unique_scores, unique_scores[-1] + epsilon]

    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _binary_metrics(targets, _threshold_predictions(scores, threshold))
        candidates.append(
            {
                "threshold": float(threshold),
                "metrics": metrics,
                "constraint_met": metrics["recall"] >= minimum_recall,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["constraint_met"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                candidate["metrics"]["false_positive_rate"],
                -candidate["metrics"]["f1"],
                -candidate["metrics"]["precision"],
                -candidate["threshold"],
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                -candidate["metrics"]["recall"],
                candidate["metrics"]["false_positive_rate"],
                -candidate["metrics"]["f1"],
            ),
        )

    return {
        "threshold": selected["threshold"],
        "minimum_recall": minimum_recall,
        "constraint_met": selected["constraint_met"],
        "metrics": selected["metrics"],
    }


def _operational_selection_key(
    calibration: dict[str, Any],
    *,
    macro_f1: float,
) -> tuple[float, ...]:
    metrics = calibration["metrics"]
    if calibration["constraint_met"]:
        return (
            1.0,
            -float(metrics["false_positive_rate"]),
            float(macro_f1),
            float(metrics["f1"]),
        )
    return (
        0.0,
        float(metrics["recall"]),
        -float(metrics["false_positive_rate"]),
        float(macro_f1),
    )


def _run_epoch(
    model: SiameseRegressionClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    index_to_label: dict[int, str],
    binary_threshold: float = 0.5,
    return_outputs: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    binary_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    binary_targets: list[int] = []
    binary_probabilities: list[float] = []
    class_targets: list[int] = []
    class_predictions: list[int] = []

    for batch in loader:
        baseline = batch["baseline"].to(device, non_blocking=True)
        current = batch["current"].to(device, non_blocking=True)
        binary_target = batch["binary_target"].to(device, non_blocking=True)
        class_target = batch["class_target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(baseline, current)
                binary_loss = binary_loss_fn(output.binary_logits, binary_target)
                class_loss = class_loss_fn(output.class_logits, class_target)
                loss = binary_loss + class_loss

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = baseline.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size

        probabilities = torch.sigmoid(output.binary_logits.detach())
        binary_targets.extend(binary_target.detach().round().int().cpu().tolist())
        binary_probabilities.extend(probabilities.cpu().tolist())
        class_targets.extend(class_target.detach().cpu().tolist())
        class_predictions.extend(output.class_logits.detach().argmax(dim=1).cpu().tolist())

    binary_predictions = _threshold_predictions(binary_probabilities, binary_threshold)
    result: dict[str, Any] = {
        "loss": total_loss / max(1, total_examples),
        "binary": _binary_metrics(binary_targets, binary_predictions),
        "binary_threshold": binary_threshold,
        "multiclass": _multiclass_metrics(
            class_targets,
            class_predictions,
            index_to_label=index_to_label,
        ),
    }
    if return_outputs:
        result["_outputs"] = {
            "binary_targets": binary_targets,
            "binary_probabilities": binary_probabilities,
            "class_targets": class_targets,
            "class_predictions": class_predictions,
        }
    return result


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _checkpoint_payload(
    *,
    model: SiameseRegressionClassifier,
    optimizer: AdamW,
    epoch: int,
    config: TrainingConfig,
    label_to_index: dict[str, int],
    dataset_metadata: dict[str, Any],
    validation_metrics: dict[str, Any],
    checkpoint_role: str,
    binary_threshold: float,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "architecture": model.architecture_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model": {
            "num_classes": model.num_classes,
        },
        "class_mapping": label_to_index,
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "input_size": config.input_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "training": asdict(config),
        "dataset": {
            "version": dataset_metadata.get("version"),
            "seed": dataset_metadata.get("seed"),
            "manifest": dataset_metadata.get("manifest"),
        },
        "decision": {
            "checkpoint_role": checkpoint_role,
            "binary_threshold": binary_threshold,
            "minimum_binary_recall": config.minimum_binary_recall,
        },
        "environment": {
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "validation_metrics": validation_metrics,
    }


def _multiclass_checkpoint_path(checkpoint_path: Path) -> Path:
    suffix = checkpoint_path.suffix or ".pt"
    stem = checkpoint_path.stem if checkpoint_path.suffix else checkpoint_path.name
    return checkpoint_path.with_name(f"{stem}.multiclass{suffix}")


def _classical_scores(
    dataset: PairedScreenshotDataset,
    *,
    device: str,
) -> tuple[list[int], dict[str, list[float]]]:
    targets: list[int] = []
    scores = {
        "mean_absolute_error": [],
        "changed_area_ratio": [],
        "ssim_distance": [],
    }

    for record in dataset.records:
        comparison = compare_images(
            dataset.root / str(record["baseline"]),
            dataset.root / str(record["current"]),
            device=device,
        )
        targets.append(int(str(record["label"]) != NO_REGRESSION_LABEL))
        scores["mean_absolute_error"].append(comparison.mean_absolute_error)
        scores["changed_area_ratio"].append(comparison.changed_area_ratio)
        scores["ssim_distance"].append(1.0 - comparison.ssim)

    return targets, scores


def _classical_benchmark(
    validation_dataset: PairedScreenshotDataset,
    test_dataset: PairedScreenshotDataset,
    *,
    device: str,
    minimum_recall: float,
) -> dict[str, Any]:
    validation_targets, validation_scores = _classical_scores(
        validation_dataset,
        device=device,
    )
    test_targets, test_scores = _classical_scores(test_dataset, device=device)

    results: dict[str, Any] = {}
    for name in validation_scores:
        calibration = _calibrate_threshold(
            validation_targets,
            validation_scores[name],
            minimum_recall=minimum_recall,
        )
        threshold = float(calibration["threshold"])
        test_predictions = _threshold_predictions(test_scores[name], threshold)
        results[name] = {
            "validation": calibration,
            "test": {
                "threshold": threshold,
                "binary": _binary_metrics(test_targets, test_predictions),
            },
        }
    return results


def _test_split_distribution(dataset: PairedScreenshotDataset) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for record in dataset.records:
        label = str(record["label"])
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items()))


def train_model(
    config: TrainingConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if config.epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if config.num_workers < 0:
        raise ValueError("num_workers must be zero or greater")
    if not 0.0 <= config.minimum_binary_recall <= 1.0:
        raise ValueError("minimum_binary_recall must be between 0 and 1")

    _set_seed(config.seed)
    dataset_root = Path(config.dataset_root)
    dataset_metadata = load_dataset_metadata(dataset_root)
    label_to_index = build_label_mapping(dataset_root)
    if len(label_to_index) < 2:
        raise ValueError("Training requires at least two classes in the dataset.")

    train_dataset = PairedScreenshotDataset(
        dataset_root,
        split="train",
        label_to_index=label_to_index,
        input_size=config.input_size,
    )
    validation_dataset = PairedScreenshotDataset(
        dataset_root,
        split="validation",
        label_to_index=label_to_index,
        input_size=config.input_size,
    )
    test_dataset = PairedScreenshotDataset(
        dataset_root,
        split="test",
        label_to_index=label_to_index,
        input_size=config.input_size,
    )

    device_info = resolve_device(config.device)
    device = torch.device(device_info.selected)
    pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = SiameseRegressionClassifier(
        num_classes=len(label_to_index),
        pretrained=config.pretrained,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    index_to_label = {index: label for label, index in label_to_index.items()}

    operational_checkpoint_path = Path(config.checkpoint_path)
    operational_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    multiclass_checkpoint_path = _multiclass_checkpoint_path(operational_checkpoint_path)

    best_macro_f1 = -1.0
    best_multiclass_epoch = 0
    best_operational_epoch = 0
    best_operational_key: tuple[float, ...] | None = None
    best_operational_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            index_to_label=index_to_label,
        )
        with torch.inference_mode():
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                index_to_label=index_to_label,
                return_outputs=True,
            )

        outputs = validation_metrics.pop("_outputs")
        calibration = _calibrate_threshold(
            outputs["binary_targets"],
            outputs["binary_probabilities"],
            minimum_recall=config.minimum_binary_recall,
        )
        validation_metrics["binary"] = calibration["metrics"]
        validation_metrics["binary_threshold"] = calibration["threshold"]
        validation_metrics["binary_calibration"] = {
            "minimum_recall": calibration["minimum_recall"],
            "constraint_met": calibration["constraint_met"],
        }

        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_summary)
        if progress is not None:
            progress(epoch_summary)

        macro_f1 = float(validation_metrics["multiclass"]["macro_f1"])
        threshold = float(validation_metrics["binary_threshold"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_multiclass_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    label_to_index=label_to_index,
                    dataset_metadata=dataset_metadata,
                    validation_metrics=validation_metrics,
                    checkpoint_role="multiclass",
                    binary_threshold=threshold,
                ),
                multiclass_checkpoint_path,
            )

        operational_key = _operational_selection_key(calibration, macro_f1=macro_f1)
        if best_operational_key is None or operational_key > best_operational_key:
            best_operational_key = operational_key
            best_operational_epoch = epoch
            best_operational_validation = validation_metrics
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    label_to_index=label_to_index,
                    dataset_metadata=dataset_metadata,
                    validation_metrics=validation_metrics,
                    checkpoint_role="operational",
                    binary_threshold=threshold,
                ),
                operational_checkpoint_path,
            )

    checkpoint = torch.load(
        operational_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    operational_threshold = float(checkpoint["decision"]["binary_threshold"])

    with torch.inference_mode():
        test_metrics = _run_epoch(
            model,
            test_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            index_to_label=index_to_label,
            binary_threshold=operational_threshold,
        )

    classical_results: dict[str, Any] | None = None
    if config.classical_benchmark:
        classical_results = _classical_benchmark(
            validation_dataset,
            test_dataset,
            device=device.type,
            minimum_recall=config.minimum_binary_recall,
        )

    benchmark = {
        "format_version": 1,
        "dataset": {
            "version": dataset_metadata.get("version"),
            "split": "test",
            "examples": len(test_dataset),
            "class_distribution": _test_split_distribution(test_dataset),
        },
        "operational_checkpoint": {
            "path": str(operational_checkpoint_path.resolve()),
            "epoch": int(checkpoint["epoch"]),
            "binary_threshold": operational_threshold,
            "minimum_binary_recall": config.minimum_binary_recall,
            "validation": best_operational_validation,
            "test": test_metrics,
        },
        "multiclass_checkpoint": {
            "path": str(multiclass_checkpoint_path.resolve()),
            "epoch": best_multiclass_epoch,
            "best_validation_macro_f1": best_macro_f1,
        },
        "classical_baselines": classical_results,
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "input_size": config.input_size,
        },
    }
    benchmark_path = operational_checkpoint_path.with_suffix(".benchmark.json")
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")

    summary = {
        "checkpoint": str(operational_checkpoint_path.resolve()),
        "operational_checkpoint": str(operational_checkpoint_path.resolve()),
        "multiclass_checkpoint": str(multiclass_checkpoint_path.resolve()),
        "device": device.type,
        "amp_enabled": amp_enabled,
        "classes": label_to_index,
        "best_epoch": best_multiclass_epoch,
        "best_multiclass_epoch": best_multiclass_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "best_operational_epoch": best_operational_epoch,
        "operational_binary_threshold": operational_threshold,
        "operational_validation": best_operational_validation,
        "test": test_metrics,
        "benchmark_file": str(benchmark_path.resolve()),
        "history": history,
    }
    history_path = operational_checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["history_file"] = str(history_path.resolve())
    return summary
