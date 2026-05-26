from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

from catgen import artifacts
from catgen.data import RAW, read_split


_LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
_RGB_TO_GRAY = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)


def load_dir(d: Path, size: int = 64) -> torch.Tensor:
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    files = sorted(d.glob("*.png"))
    if not files:
        raise SystemExit(f"no .png in {d}")
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in files])


def laplacian_variance(imgs: torch.Tensor) -> torch.Tensor:
    gray = (imgs * _RGB_TO_GRAY).sum(dim=1, keepdim=True)
    edges = F.conv2d(gray, _LAPLACIAN, padding=1)
    return edges.flatten(1).var(dim=1)


def build_inception(device: torch.device) -> nn.Module:
    m = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False, aux_logits=True)
    m.fc = nn.Identity()
    m.aux_logits = False
    m.AuxLogits = None
    return m.to(device).eval()


@torch.no_grad()
def inception_features(imgs: torch.Tensor, model: nn.Module, device: torch.device, batch: int = 32) -> torch.Tensor:
    resized = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
    normed = (resized - 0.5) / 0.5
    feats = []
    for i in range(0, len(normed), batch):
        feats.append(model(normed[i:i + batch].to(device)).cpu())
    return torch.cat(feats)


def mean_pairwise_distance(feats: torch.Tensor) -> tuple[float, float]:
    d = torch.cdist(feats, feats)
    n = feats.size(0)
    triu = d[torch.triu_indices(n, n, offset=1).unbind()]
    return float(triu.mean()), float(triu.std())


def nn_distances(feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(feats_a, feats_b)
    return d.min(dim=1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-split", default=None, help="for NN-to-train memorization; defaults to manifest's data.split")
    ap.add_argument("--n-train-ref", type=int, default=1000, help="cap training images for NN feature cache")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = artifacts.RUNS / args.run_id
    gen_dir = d / "eval" / "samples"
    if not any(gen_dir.glob("*.png")):
        raise SystemExit(f"no generated samples in {gen_dir}; run catgen.sample first")

    manifest = json.loads((d / "manifest.json").read_text())
    train_split = args.train_split or manifest["config"]["data"]["split"]

    device = torch.device(args.device)
    gen = load_dir(gen_dir, size=manifest["config"]["data"]["image_size"])

    sharp = laplacian_variance(gen).numpy()
    sharp_summary = {
        "mean": float(sharp.mean()),
        "std": float(sharp.std()),
        "min": float(sharp.min()),
        "max": float(sharp.max()),
        "n": int(sharp.size),
    }

    incept = build_inception(device)
    gen_feats = inception_features(gen, incept, device)
    div_mean, div_std = mean_pairwise_distance(gen_feats)

    train_paths = read_split(train_split)[: args.n_train_ref]
    tf = transforms.Compose([
        transforms.Resize((manifest["config"]["data"]["image_size"],) * 2),
        transforms.ToTensor(),
    ])
    train_imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in train_paths])
    train_feats = inception_features(train_imgs, incept, device)
    nn_dists = nn_distances(gen_feats, train_feats).numpy()
    nn_summary = {
        "mean": float(nn_dists.mean()),
        "min": float(nn_dists.min()),
        "p05": float(np.quantile(nn_dists, 0.05)),
        "n_train_ref": len(train_paths),
        "train_split": train_split,
    }

    result = {
        "sharpness_laplacian_var": sharp_summary,
        "diversity_pairwise_inception": {"mean": div_mean, "std": div_std, "n_gen": int(len(gen_feats))},
        "nn_to_train_inception": nn_summary,
    }
    (d / "eval" / "quality.json").write_text(json.dumps(result, indent=2))
    np.save(d / "eval" / "sharpness.npy", sharp)
    np.save(d / "eval" / "nn_distances.npy", nn_dists)
    artifacts.log_event(d, "quality_computed", **{k: v.get("mean") if isinstance(v, dict) else v for k, v in result.items()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
