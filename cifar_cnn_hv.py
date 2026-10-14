#!/usr/bin/env python3
"""
cifar_cnn_hv.py  --  VSA: per-pixel hypervector encoding by random projection,
then the SAME CNN as cifar_cnn_rgb.py (CIFAR-10 / CIFAR-100).

Each RGB pixel x(p) in R^3 (normalised) is mapped by a FIXED, untrained random
projection W in R^{D x 3} to a hypervector h(p) in R^D. The image becomes a
D-channel tensor; spatial layout is kept, so the CNN still sees a 32x32 grid.

Encoders (--encoder):
  linear  h = W x / sqrt(3)            pure random projection (rank 3). CONTROL:
                                       the first conv can absorb W exactly, so the
                                       information is unchanged; only width/params change.
  sign    h = sign(W x + b)            bipolar MAP hypervector (SimHash with offsets).
                                       b = -W u, u ~ N(0, I): every hyperplane passes through
                                       a random colour, so brightness is kept (sign(Wx)
                                       alone would only keep the colour direction).
  rff     h = sqrt(2) cos(beta W x + b) real FPE / random Fourier features:
                                       Re[ exp(i(w_r r + w_g g + w_b b)) ] = binding of
                                       three per-channel FPE codes; kernel
                                       <h(x), h(x')>/D ~ exp(-beta^2 |x - x'|^2 / 2).
                                       --beta sets the colour bandwidth (z units).

The SAME W (and b) is used for every pixel of every image: one fixed codebook,
drawn once from --enc-seed. The CNN then starts with a 1x1 readout that maps the
whole D-dim hypervector of each pixel to stem_ch (=128) values (see CNN), and
only then applies 3x3 convolutions. --readout fixed makes that 1x1 a frozen
similarity query against hypervectors of random colours (no learned weights
touch the hypervector at all).

Pairs with cifar_cnn_rgb.py. Both scripts use the SAME CNN, optimiser, schedule,
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

Here the comp_* sweeps corrupt the D hypervector components the CNN sees -- the
test of distributed (holographic) coding. In the baseline they corrupt R/G/B.

Usage
  python cifar_cnn_hv.py --data ./data --encoder sign --D 1024 --epochs 30

After each run (needs matplotlib), artifacts go under results/hv/ by default (JSON, CSV,
PNG; robustness files if sweeps ran). Run from classification/ so src/ imports resolve.
  python cifar_cnn_hv.py --encoder rff --beta 1.0 --D 1024
  python cifar_cnn_hv.py --encoder linear --D 1024      # control
  python cifar_cnn_hv.py --fake-data --epochs 1 --D 64  # smoke test, no download

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
    p.add_argument("--encoder", choices=["sign", "rff", "linear"], default="sign")
    p.add_argument("--D", type=int, default=1024, help="hypervector dimension per pixel")
    p.add_argument("--beta", type=float, default=1.0, help="rff colour bandwidth (1/length in z units)")
    p.add_argument("--readout", choices=["learned", "fixed"], default="learned",
                   help="learned: trainable 1x1 queries; fixed: queries = hypervectors of stem_ch random "
                        "colours (VSA cleanup-memory readout, frozen)")
    p.add_argument("--enc-seed", type=int, default=0, help="seed of the random projection (fixed codebook)")
    p.add_argument("--no-robustness", action="store_true", help="skip corruption sweeps")
    p.add_argument("--results-dir", default="./results", help="root folder for run outputs (rgb/ and hv/ subdirs)")
    p.add_argument("--out", default=None,
                   help="override JSON path (default <results-dir>/hv/results_hv_<enc>_<readout>_D<D>_<dataset>_s<seed>.json)")
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


class HVEncoder(nn.Module):
    """Fixed per-pixel random projection RGB -> R^D. No trainable parameters
    (W and b are buffers, saved with the model so the codebook is reproducible)."""

    def __init__(self, D, kind="sign", beta=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(D, 3, generator=g)                      # Gaussian random projection
        if kind == "linear":
            b = torch.zeros(D)
        elif kind == "sign":
            u = torch.randn(D, 3, generator=g)                  # one random colour per hyperplane ...
            b = -(W * u).sum(1)                                 # ... hyperplane passes through it
        elif kind == "rff":
            W = beta * W
            b = 2 * math.pi * torch.rand(D, generator=g)        # random phase
        else:
            raise ValueError(kind)
        self.kind, self.D = kind, D
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def encode(self, x):
        """x (..., 3) normalised colours -> (..., D)."""
        z = x.float() @ self.W.T + self.b
        if self.kind == "linear":
            return z / math.sqrt(3.0)
        if self.kind == "sign":
            return torch.where(z >= 0, 1.0, -1.0)
        return math.sqrt(2.0) * torch.cos(z)

    def forward(self, x):                                       # (B, 3, H, W) -> (B, D, H, W)
        h = self.encode(x.permute(0, 2, 3, 1))                  # per-pixel, (B, H, W, D)
        return h.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def describe_encoder(enc, n=4000, seed=0):
    """Kernel check: mean cosine similarity of h(x) and h(x + d*e) for random
    colours x and random unit directions e (d in z units), and of random pairs."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    e = torch.randn(n, 3, generator=g)
    e = e / e.norm(dim=1, keepdim=True)
    hx = F.normalize(enc.encode(x.to(enc.W.device)), dim=1)
    out = {}
    for d in [0.1, 0.25, 0.5, 1.0, 2.0]:
        hy = F.normalize(enc.encode((x + d * e).to(enc.W.device)), dim=1)
        out[str(d)] = (hx * hy).sum(1).mean().item()
    perm = torch.randperm(n, generator=g)
    out["random_pair"] = (hx * hx[perm.to(hx.device)]).sum(1).mean().item()
    print("  kernel  cos(h(x), h(x+d))  " + "  ".join(f"d={k}:{v:.3f}" for k, v in out.items()))
    return out


@torch.no_grad()
def set_fixed_readout(model, seed=0):
    """Freeze the 1x1 readout to q_k = h(c_k)/D for random colours c_k, so output
    k at pixel p is <h(c_k), h(x(p))>/D ~ kernel(c_k, x(p)): the similarity of the
    pixel's whole hypervector to codebook item k (VSA cleanup memory)."""
    conv = model.cnn.stem[0]
    g = torch.Generator().manual_seed(seed + 1)
    c = torch.randn(conv.out_channels, 3, generator=g).to(conv.weight.device)
    q = model.encoder.encode(c) / model.encoder.D                  # (stem_ch, D)
    conv.weight.copy_(q[:, :, None, None])
    conv.weight.requires_grad_(False)


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

    encoder = HVEncoder(args.D, args.encoder, args.beta, args.enc_seed)
    in_ch = args.D
    model = Net(encoder, CNN(in_ch, n_cls, args.width, args.stem_ch)).to(device).to(memory_format=torch.channels_last)

    if args.readout == "fixed":
        set_fixed_readout(model, args.enc_seed)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_stem = model.cnn.stem[0].weight.numel()
    print(f"[hv/{args.encoder}, {args.readout} readout] device {device}  D {in_ch}  trainable params {n_par:,} (1x1 readout {n_stem:,})"
          + (f"  beta {args.beta}" if args.encoder == "rff" else ""))
    kernel = describe_encoder(model.encoder)

    t0 = time.time()
    history = train(model, train_dl, test_dl, args, device)
    train_sec = time.time() - t0
    rob = None if args.no_robustness else (print("robustness:") or robustness(model, test_dl, device))

    out = resolve_output_path(
        args.out,
        args.results_dir,
        "hv",
        f"results_hv_{args.encoder}_{args.readout}_D{args.D}_{args.dataset}_s{args.seed}.json",
    )
    res = {"script": "hv", "encoder": args.encoder, "D": in_ch, "kernel": kernel, "args": vars(args), "params": n_par,
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
