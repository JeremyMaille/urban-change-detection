"""Tests for the LEVIR patch dataset logic (no data download required)."""
import torch

from src.datasets.dataset import LEVIRPatchDataset, IMAGENET_MEAN, IMAGENET_STD


class FakeBase:
    """Minimal stand-in for the LEVIR-CD+ base dataset (3 images of 1024x1024)."""

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {
            "image1": torch.rand(3, 1024, 1024),
            "image2": torch.rand(3, 1024, 1024),
            "mask": (torch.rand(1024, 1024) > 0.95).long(),
        }


def test_patch_index_count():
    """3 images of 1024x1024 in 256x256 patches -> 3 * 16 = 48 patches."""
    ds = LEVIRPatchDataset(FakeBase(), split="val", patch_size=256)
    assert len(ds) == 48


def test_patch_index_covers_image_without_overlap():
    ds = LEVIRPatchDataset(FakeBase(), split="val", patch_size=256)
    coords = [(r, c) for (i, r, c) in ds.index if i == 0]
    assert len(coords) == 16
    assert len(set(coords)) == 16  # no duplicates
    assert all(r % 256 == 0 and c % 256 == 0 for r, c in coords)
    assert max(r for r, _ in coords) == 1024 - 256


def test_normalization_constants_are_imagenet():
    """The pretrained ResNet34 encoder requires ImageNet statistics."""
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]
