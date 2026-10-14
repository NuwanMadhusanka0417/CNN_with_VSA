"""Feature-map PNG export for a trained Net (RGB or HV) on one test image."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


def _hook_targets(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    if not isinstance(model.encoder, nn.Identity):
        yield "encoder", model.encoder
    yield "cnn_stem", model.cnn.stem
    for i, block in enumerate(model.cnn.features):
        yield f"cnn_features_{i}", block


def _spatial_feature_map(t: torch.Tensor) -> bool:
    return t.dim() == 4 and t.shape[2] >= 2 and t.shape[3] >= 2


def _denormalize_rgb(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """x: (3, H, W) normalised -> (H, W, 3) float in [0, 1]."""
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    img = (x[:3] * s + m).clamp(0.0, 1.0)
    return img.permute(1, 2, 0).cpu().numpy()


def _save_input_png(x: torch.Tensor, path: Path, mean: Sequence[float], std: Sequence[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(_denormalize_rgb(x, mean, std))
    plt.axis("off")
    plt.title("input (RGB)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_channel_grid(fm: torch.Tensor, path: Path, max_channels: int, title: str) -> None:
    """fm: (C, H, W) activations after ReLU (or any signed maps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = min(max_channels, fm.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, c, figsize=(1.6 * c, 2))
    if c == 1:
        axes = [axes]
    for i in range(c):
        ch = fm[i].float().numpy()
        axes[i].imshow(ch, cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"ch {i}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _feature_map_to_pca_rgb(fm: torch.Tensor) -> tuple[np.ndarray, list[float]]:
    """Per-pixel C-dim vectors (C, H, W) -> (H, W, 3) RGB image via PCA (NumPy SVD)."""
    c, h, w = fm.shape
    n_comp = min(3, c)
    flat = fm.float().permute(1, 2, 0).reshape(-1, c).numpy()
    flat = flat - flat.mean(axis=0, keepdims=True)
    if flat.shape[0] < 2:
        return np.zeros((h, w, 3), dtype=np.float32), [0.0] * n_comp

    _, s, vt = np.linalg.svd(flat, full_matrices=False)
    coeff = flat @ vt[:n_comp].T
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(n_comp):
        ch = coeff[:, i].reshape(h, w)
        lo, hi = ch.min(), ch.max()
        rgb[..., i] = (ch - lo) / (hi - lo + 1e-8)

    var = (s**2) / max(1, flat.shape[0] - 1)
    total = float(var.sum()) + 1e-12
    explained = [float(var[i] / total) for i in range(n_comp)]
    return rgb, explained


def _save_pca_rgb_png(fm: torch.Tensor, path: Path, title: str) -> list[float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb, explained = _feature_map_to_pca_rgb(fm)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return explained


def _should_pca_layer(layer_name: str, num_channels: int, pca_min_channels: int) -> bool:
    if layer_name == "encoder":
        return True
    return num_channels >= pca_min_channels


@torch.no_grad()
def export_feature_maps(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    out_dir: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    max_channels: int = 8,
    dataset: str = "cifar10",
    use_pca: bool = True,
    pca_min_channels: int = 16,
) -> Mapping[str, Any]:
    """
    Run one forward pass with hooks on encoder (if any), stem, and each features block.
    Writes input.png, channel-grid PNGs, PCA→RGB PNGs for HV encoder (and wide layers),
    and manifest.json under out_dir.
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in _hook_targets(model):
        handles.append(
            module.register_forward_hook(
                lambda _m, _inp, out, layer=name: activations.__setitem__(layer, out.detach().cpu())
            )
        )

    logits = model(x)
    pred = int(logits.argmax(1).item())
    label = int(y_true.item())
    for h in handles:
        h.remove()

    _save_input_png(x[0].cpu(), out_dir / "input.png", mean, std)

    layers_meta: list[dict[str, Any]] = []
    for name, act in activations.items():
        if not _spatial_feature_map(act):
            continue
        shape = list(act.shape)
        png = out_dir / f"{name}.png"
        _plot_channel_grid(
            act[0],
            png,
            max_channels,
            title=f"{name}  {shape[1]}×{shape[2]}×{shape[3]}",
        )
        entry: dict[str, Any] = {"layer": name, "shape": shape, "file": png.name}
        if use_pca and _should_pca_layer(name, shape[1], pca_min_channels):
            pca_path = out_dir / f"{name}_pca_rgb.png"
            pca_title = f"{name} PCA→RGB  (C={shape[1]})"
            if name == "encoder":
                pca_title = f"HV hypervector field  D={shape[1]}  PCA→RGB"
            explained = _save_pca_rgb_png(act[0], pca_path, pca_title)
            entry["pca_rgb_file"] = pca_path.name
            entry["pca_variance_ratio"] = explained
        layers_meta.append(entry)

    n_cls = 10 if dataset == "cifar10" else 100
    meta: dict[str, Any] = {
        "label": label,
        "prediction": pred,
        "correct": pred == label,
        "num_classes": n_cls,
        "max_channels_plotted": max_channels,
        "pca_enabled": use_pca,
        "pca_min_channels": pca_min_channels,
        "layers": layers_meta,
    }
    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"dir": str(out_dir.resolve()), "manifest": str(manifest.resolve()), **meta}
