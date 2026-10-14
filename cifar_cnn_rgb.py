#!/usr/bin/env python3
"""
cifar_cnn_rgb.py  --  BASELINE: CNN on raw RGB pixels (CIFAR-10 / CIFAR-100).

Pairs with cifar_cnn_hv.py. Both scripts use the SAME CNN, optimiser, schedule,
augmentation and evaluation; the only difference is the per-pixel input
representation the CNN sees:

    baseline : x(p) in R^3              (normalised R, G, B)
    HV       : h(p) = enc(x(p)) in R^D  (fixed random projection, D >> 3)

Besides clean test accuracy, both scripts run the same robustness sweeps:

  * pixel_noise       Gaussian noise added to the normalised image BEFORE any
                      encoding (same corrupted image for both scripts: fixed seed).
  * comp_dropout      a random fraction p of the representation channels the CNN
                      sees is zeroed, independently per image and channel.
  * comp_noise        Gaussian noise added to every representation channel,
                      sigma relative to that channel's RMS.

For this baseline the "representation" is the 3 colour channels, so the comp_*
sweeps corrupt R/G/B directly. In the HV script they corrupt the D hypervector
components -- that is the test of distributed (holographic) coding.

Usage
  python cifar_cnn_rgb.py --data ./data --epochs 30
  python cifar_cnn_rgb.py --fake-data --epochs 1        # smoke test, no download

After each run (needs matplotlib for plots), artifacts go under results/rgb/ by default:
JSON + *_history.csv + *_training.png (+ *_robustness.* if sweeps ran). Run from the
classification/ directory so src/ imports resolve.

Dataset: Krizhevsky 2009, "Learning Multiple Layers of Features from Tiny Images".
Downloaded by torchvision on first run (no internet on HPC compute nodes: run
once with --download-only on a login node).
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from src.cifar_feature_maps import export_feature_maps
from src.cifar_run_io import resolve_output_path, save_run_artifacts

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

PIXEL_NOISE = [0.0, 0.1, 0.25, 0.5, 1.0]          # std in normalised (z) units
COMP_DROPOUT = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]   # fraction of channels zeroed
COMP_NOISE = [0.0, 0.25, 0.5, 1.0, 2.0]           # std relative to channel RMS


# ----------------------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="./data")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05, help="peak SGD learning rate")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--width", type=int, default=64, help="channels of the first CNN stage")
    p.add_argument("--stem-ch", type=int, default=128, help="output size of the 1x1 readout")
    p.add_argument("--no-augment", action="store_true", help="disable random crop + flip")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-robustness", action="store_true", help="skip corruption sweeps")
    p.add_argument("--results-dir", default="./results", help="root folder for run outputs (rgb/ and hv/ subdirs)")
    p.add_argument("--out", default=None, help="override JSON path (default <results-dir>/rgb/results_rgb_<dataset>_s<seed>.json)")
    p.add_argument("--no-plots", action="store_true", help="skip PNG training/robustness curves")
    p.add_argument("--no-csv", action="store_true", help="skip per-epoch and robustness CSV export")
    p.add_argument("--no-feature-maps", action="store_true", help="skip feature-map PNG export after training")
    p.add_argument("--feature-map-channels", type=int, default=8, help="channels per layer grid PNG")
    p.add_argument("--no-feature-map-pca", action="store_true", help="skip PCA→RGB summary PNGs")
    p.add_argument("--feature-map-pca-min-channels", type=int, default=16,
                   help="PCA→RGB for CNN layers with at least this many channels (encoder always)")
    p.add_argument("--save-model", default=None, help="optional path to save the trained weights")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--fake-data", action="store_true", help="random images, no download (smoke test)")
    p.add_argument("--fake-n", type=int, default=512)
    return p.parse_args()


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(args):
    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    aug = [] if args.no_augment else [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    tf_train = T.Compose(aug + [T.ToTensor(), norm])
    tf_test = T.Compose([T.ToTensor(), norm])
    n_cls = 10 if args.dataset == "cifar10" else 100

    if args.fake_data:
        train = torchvision.datasets.FakeData(args.fake_n, (3, 32, 32), n_cls, transform=tf_train, random_offset=0)
        test = torchvision.datasets.FakeData(args.fake_n // 4, (3, 32, 32), n_cls, transform=tf_test, random_offset=1)
    else:
        DS = torchvision.datasets.CIFAR10 if args.dataset == "cifar10" else torchvision.datasets.CIFAR100
        train = DS(args.data, train=True, download=True, transform=tf_train)
        test = DS(args.data, train=False, download=True, transform=tf_test)

    pin = torch.cuda.is_available()
    train_dl = torch.utils.data.DataLoader(train, args.batch, shuffle=True, num_workers=args.workers,
                                           pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    test_dl = torch.utils.data.DataLoader(test, 500, shuffle=False, num_workers=args.workers, pin_memory=pin)
    return train_dl, test_dl, n_cls


# ----------------------------------------------------------------------------- model
def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CNN(nn.Module):
    """1x1 readout stem + VGG-style CNN. Identical in both scripts except `in_ch`
    of the 1x1 readout (3 for RGB, D for hypervectors).

    Readout: a 1x1 conv sees the WHOLE per-pixel vector at once. Output channel k
    at pixel p is the inner product <q_k, h(p)> of a query vector q_k with the full
    hypervector -- a similarity readout, never a per-component operation. Only
    after this (stem_ch values per pixel) does spatial 3x3 processing start.
    Nothing touches individual hypervector components on its own: no BatchNorm,
    dropout or depthwise conv on the D input channels."""

    def __init__(self, in_ch, n_cls, w=64, stem_ch=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, stem_ch, 1, bias=False),
                                  nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            conv_bn(stem_ch, w), conv_bn(w, w), nn.MaxPool2d(2),          # 32 -> 16
            conv_bn(w, 2 * w), conv_bn(2 * w, 2 * w), nn.MaxPool2d(2),    # 16 -> 8
            conv_bn(2 * w, 4 * w), conv_bn(4 * w, 4 * w), nn.MaxPool2d(2),  # 8 -> 4
            conv_bn(4 * w, 8 * w), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8 * w, n_cls)

    def forward(self, h):
        return self.fc(torch.flatten(self.features(self.stem(h)), 1))


class Net(nn.Module):
    """encoder (fixed, no parameters) -> optional representation corruption -> CNN."""

    def __init__(self, encoder, cnn):
        super().__init__()
        self.encoder, self.cnn = encoder, cnn

    def forward(self, x, rep_corrupt=None):
        h = self.encoder(x)
        if rep_corrupt is not None:
            h = rep_corrupt(h)
        return self.cnn(h)


# ----------------------------------------------------------------------------- corruption
def pixel_noise(x, sigma, gen):
    if sigma == 0:
        return x
    return x + sigma * torch.randn(x.shape, generator=gen, device="cpu").to(x.device)


def make_comp_dropout(p, gen):
    def f(h):
        if p == 0:
            return h
        keep = (torch.rand(h.shape[0], h.shape[1], 1, 1, generator=gen, device="cpu") >= p).to(h)
        return h * keep.to(h.device)          # no rescaling: the CNN was never trained with dropout
    return f


def make_comp_noise(sigma, gen):
    def f(h):
        if sigma == 0:
            return h
        rms = h.float().pow(2).mean(dim=(0, 2, 3), keepdim=True).sqrt()
        n = torch.randn(h.shape, generator=gen, device="cpu").to(h.device)
        return h + sigma * rms.to(h.dtype) * n.to(h.dtype)
    return f


# ----------------------------------------------------------------------------- train / eval
def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(model, dl, device, pix_sigma=0.0, rep_corrupt_factory=None, seed=1234):
    """Accuracy (%). Corruption RNG is seeded per batch so every run (and both
    scripts, for pixel noise) sees exactly the same corruption."""
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(dl):
        gen = torch.Generator().manual_seed(seed + i)
        x = pixel_noise(x, pix_sigma, gen).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        rc = rep_corrupt_factory(gen) if rep_corrupt_factory is not None else None
        with autocast(device):
            logits = model(x, rep_corrupt=rc)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train(model, train_dl, test_dl, args, device):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)                               # 1 epoch linear warm-up, then cosine
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history = []
    for ep in range(args.epochs):
        model.train()
        t0, loss_sum, n, correct = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.numel()
        train_acc = 100.0 * correct / n
        test_acc = evaluate(model, test_dl, device)
        history.append({"epoch": ep + 1, "train_loss": loss_sum / n, "train_acc": train_acc,
                        "test_acc": test_acc, "sec": time.time() - t0})
        print(f"epoch {ep + 1:3d}/{args.epochs}  loss {loss_sum / n:.4f}  train {train_acc:6.2f}%  "
              f"test {test_acc:6.2f}%  ({time.time() - t0:.1f}s)", flush=True)
    return history


def robustness(model, test_dl, device):
    res = {"pixel_noise": {}, "comp_dropout": {}, "comp_noise": {}}
    for s in PIXEL_NOISE:
        res["pixel_noise"][str(s)] = evaluate(model, test_dl, device, pix_sigma=s)
    for p in COMP_DROPOUT:
        res["comp_dropout"][str(p)] = evaluate(model, test_dl, device, rep_corrupt_factory=lambda g, p=p: make_comp_dropout(p, g))
    for s in COMP_NOISE:
        res["comp_noise"][str(s)] = evaluate(model, test_dl, device, rep_corrupt_factory=lambda g, s=s: make_comp_noise(s, g))
    for k, v in res.items():
        print(f"  {k:13s} " + "  ".join(f"{a}:{b:5.1f}" for a, b in v.items()))
    return res


# ----------------------------------------------------------------------------- main
def main():
    args = get_args()
    if args.download_only:
        DS = torchvision.datasets.CIFAR10 if args.dataset == "cifar10" else torchvision.datasets.CIFAR100
        DS(args.data, train=True, download=True)
        DS(args.data, train=False, download=True)
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    train_dl, test_dl, n_cls = get_loaders(args)

    encoder = nn.Identity()                      # the representation IS the normalised RGB pixel
    in_ch = 3
    model = Net(encoder, CNN(in_ch, n_cls, args.width, args.stem_ch)).to(device).to(memory_format=torch.channels_last)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_stem = model.cnn.stem[0].weight.numel()
    print(f"[rgb] device {device}  input channels {in_ch}  trainable params {n_par:,} (1x1 readout {n_stem:,})")

    t0 = time.time()
    history = train(model, train_dl, test_dl, args, device)
    train_sec = time.time() - t0
    rob = None if args.no_robustness else (print("robustness:") or robustness(model, test_dl, device))

    out = resolve_output_path(
        args.out, args.results_dir, "rgb", f"results_rgb_{args.dataset}_s{args.seed}.json")
    res = {"script": "rgb", "encoder": "rgb", "D": in_ch, "args": vars(args), "params": n_par,
           "readout_params": n_stem, "train_sec": train_sec,
           "final_train_acc": history[-1]["train_acc"], "final_test_acc": history[-1]["test_acc"],
           "history": history, "robustness": rob}
    if not args.no_feature_maps:
        x0, y0 = next(iter(test_dl))
        fm_dir = out.parent / f"{out.stem}_feature_maps"
        res["feature_maps"] = export_feature_maps(
            model, x0[:1].to(device), y0[:1], fm_dir, CIFAR_MEAN, CIFAR_STD,
            max_channels=args.feature_map_channels, dataset=args.dataset,
            use_pca=not args.no_feature_map_pca,
            pca_min_channels=args.feature_map_pca_min_channels,
        )
        print(f"feature maps -> {res['feature_maps']['dir']}")
    paths = save_run_artifacts(res, out, save_plots=not args.no_plots, save_csv=not args.no_csv)
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
    print(f"final train {history[-1]['train_acc']:.2f}%  test {history[-1]['test_acc']:.2f}%  -> {out}")
    extras = [f"{k}: {v}" for k, v in paths.items() if k != "json"]
    if extras:
        print("artifacts: " + "  ".join(extras))


if __name__ == "__main__":
    main()
