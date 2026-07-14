"""Compute reference input statistics on the LEVIR-CD+ training set.

The output JSON is committed to the repo and used by the serving layer
to detect input drift at inference time.
"""
import argparse
import json

import numpy as np
import torch
from torchgeo.datasets import LEVIRCDPlus


def main(root: str, out: str, max_images: int = 200):
    ds = LEVIRCDPlus(root=root, split="train")
    n = min(len(ds), max_images)

    per_image_means = []  # (n, 3)
    per_image_stds = []

    for i in range(n):
        sample = ds[i]
        img = sample["image1"].float()  # (3, H, W), raw [0, 255] scale
        per_image_means.append([img[c].mean().item() for c in range(3)])
        per_image_stds.append([img[c].std().item() for c in range(3)])

    means = np.array(per_image_means)
    stds = np.array(per_image_stds)

    reference = {
        "n_images": n,
        # Distribution of per-image channel means over the training set
        "channel_mean": means.mean(axis=0).tolist(),
        "channel_mean_std": means.std(axis=0).tolist(),
        # Distribution of per-image channel stds (contrast)
        "channel_std": stds.mean(axis=0).tolist(),
        "channel_std_std": stds.std(axis=0).tolist(),
    }

    with open(out, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference stats over {n} training images written to {out}")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/levir")
    parser.add_argument("--out", default="src/serving/reference_stats.json")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    main(args.root, args.out, args.max_images)
