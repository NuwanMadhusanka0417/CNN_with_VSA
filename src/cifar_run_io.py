"""JSON, CSV, and figure export for cifar_cnn_rgb.py / cifar_cnn_hv.py runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def resolve_output_path(
    out: str | Path | None,
    results_dir: str | Path,
    run_subdir: str,
    default_filename: str,
) -> Path:
    """Default layout: ``<results_dir>/<run_subdir>/<default_filename>`` (e.g. results/rgb/...)."""
    if out is not None:
        return Path(out)
    return Path(results_dir) / run_subdir / default_filename


def save_json(res: Mapping[str, Any], json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return json_path


def save_history_csv(history: Sequence[Mapping[str, Any]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("epoch", "train_loss", "train_acc", "test_acc", "sec")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k) for k in fields})
    return csv_path


def save_robustness_csv(rob: Mapping[str, Mapping[str, float]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("sweep", "level", "test_acc"))
        for sweep, levels in rob.items():
            for level, acc in sorted(levels.items(), key=lambda kv: float(kv[0])):
                w.writerow((sweep, level, acc))
    return csv_path


def _plot_training(history: Sequence[Mapping[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(h["epoch"]) for h in history]
    train_acc = [float(h["train_acc"]) for h in history]
    test_acc = [float(h["test_acc"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(title)

    ax_acc.plot(epochs, train_acc, marker="o", markersize=3, label="train acc (%)")
    ax_acc.plot(epochs, test_acc, marker="o", markersize=3, label="test acc (%)")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    ax_loss.plot(epochs, train_loss, marker="o", markersize=3, color="C3", label="train loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_robustness(rob: Mapping[str, Mapping[str, float]], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "pixel_noise": "pixel noise σ (norm. units)",
        "comp_dropout": "channel dropout p",
        "comp_noise": "channel noise σ (× RMS)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)

    for ax, (sweep, levels) in zip(axes, rob.items()):
        pairs = sorted(((float(k), float(v)) for k, v in levels.items()), key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", markersize=4)
        ax.set_xlabel(labels.get(sweep, sweep))
        ax.set_ylabel("test acc (%)")
        ax.set_title(sweep.replace("_", " "))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_run_artifacts(
    res: Mapping[str, Any],
    json_path: str | Path,
    *,
    save_plots: bool = True,
    save_csv: bool = True,
) -> dict[str, str]:
    """Write JSON plus optional CSV/plots next to the JSON stem. Returns paths written."""
    json_path = Path(json_path)
    stem = json_path.stem
    out_dir = json_path.parent
    written: dict[str, str] = {"json": str(json_path.resolve())}

    save_json(res, json_path)

    history = res.get("history") or []
    rob = res.get("robustness")

    if save_csv and history:
        p = save_history_csv(history, out_dir / f"{stem}_history.csv")
        written["history_csv"] = str(p.resolve())
    if save_csv and rob:
        p = save_robustness_csv(rob, out_dir / f"{stem}_robustness.csv")
        written["robustness_csv"] = str(p.resolve())

    if save_plots and history:
        tag = res.get("script", "run")
        enc = res.get("encoder", "")
        title = f"{tag}/{enc} training" if enc else f"{tag} training"
        p = out_dir / f"{stem}_training.png"
        _plot_training(history, p, title.strip("/"))
        written["training_plot"] = str(p.resolve())

    if save_plots and rob:
        tag = res.get("script", "run")
        enc = res.get("encoder", "")
        title = f"{tag}/{enc} robustness" if enc else f"{tag} robustness"
        p = out_dir / f"{stem}_robustness.png"
        _plot_robustness(rob, p, title.strip("/"))
        written["robustness_plot"] = str(p.resolve())

    return written
