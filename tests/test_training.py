"""Tests for the loss functions and evaluation metrics."""
import pytest
import torch

from src.training.trainer import DiceLoss, CombinedLoss, compute_metrics


# ---------- DiceLoss ----------

def test_dice_loss_near_zero_for_perfect_prediction():
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., :16, :] = 1.0
    # Large logits -> sigmoid saturates to the target
    logits = torch.where(targets == 1.0, torch.tensor(50.0), torch.tensor(-50.0))
    loss = DiceLoss()(logits, targets)
    assert loss.item() < 0.01


def test_dice_loss_near_one_for_inverted_prediction():
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., :16, :] = 1.0
    logits = torch.where(targets == 1.0, torch.tensor(-50.0), torch.tensor(50.0))
    loss = DiceLoss()(logits, targets)
    assert loss.item() > 0.95


def test_dice_loss_is_bounded():
    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = DiceLoss()(logits, targets)
    assert 0.0 <= loss.item() <= 1.0


# ---------- CombinedLoss ----------

def test_combined_loss_is_positive_scalar():
    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.9).float()  # ~10% positives
    loss = CombinedLoss()(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_combined_loss_decreases_with_better_prediction():
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., :8, :] = 1.0
    good = torch.where(targets == 1.0, torch.tensor(5.0), torch.tensor(-5.0))
    bad = torch.where(targets == 1.0, torch.tensor(-5.0), torch.tensor(5.0))
    criterion = CombinedLoss()
    assert criterion(good, targets).item() < criterion(bad, targets).item()


def test_combined_loss_is_differentiable():
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    targets = (torch.rand(1, 1, 32, 32) > 0.5).float()
    loss = CombinedLoss()(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------- compute_metrics ----------

def test_metrics_perfect_prediction():
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., :16, :] = 1.0
    logits = torch.where(targets == 1.0, torch.tensor(50.0), torch.tensor(-50.0))
    m = compute_metrics(logits, targets)
    assert m["f1"] == pytest.approx(1.0, abs=1e-4)
    assert m["iou"] == pytest.approx(1.0, abs=1e-4)
    assert m["precision"] == pytest.approx(1.0, abs=1e-4)
    assert m["recall"] == pytest.approx(1.0, abs=1e-4)


def test_metrics_all_negative_prediction_gives_zero_recall():
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., :16, :] = 1.0
    logits = torch.full_like(targets, -50.0)  # predicts 'no change' everywhere
    m = compute_metrics(logits, targets)
    assert m["recall"] == pytest.approx(0.0, abs=1e-4)
    assert m["f1"] == pytest.approx(0.0, abs=1e-4)


def test_metrics_known_half_overlap():
    """Prediction covers the target half plus an equal-sized false positive band.

    TP = FN = 0 (prediction includes all positives)... construct precisely:
    target = top quarter; prediction = top half.
    TP = 256, FP = 256, FN = 0 -> precision 0.5, recall 1.0, f1 2/3, iou 0.5.
    """
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., :8, :] = 1.0  # 8*32 = 256 positive pixels
    logits = torch.full_like(targets, -50.0)
    logits[..., :16, :] = 50.0  # predicts 16*32 = 512 pixels
    m = compute_metrics(logits, targets)
    assert m["precision"] == pytest.approx(0.5, abs=1e-3)
    assert m["recall"] == pytest.approx(1.0, abs=1e-3)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["iou"] == pytest.approx(0.5, abs=1e-3)
