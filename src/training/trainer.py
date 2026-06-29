import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss mesure le chevauchement entre prédiction et ground truth.
    Naturellement robuste au déséquilibre de classe car elle ne calcule
    que sur les pixels positifs (changés).
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    BCE pondérée + Dice Loss.
    Le pos_weight compense le déséquilibre ~95/5 en pénalisant davantage
    les faux négatifs (pixels changés manqués).
    """

    def __init__(self, pos_weight=10.0, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.dice      = DiceLoss()
        self.dice_w    = dice_weight
        self.bce_w     = bce_weight

    def forward(self, logits, targets):
        # Déplace pos_weight sur le bon device dynamiquement
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_w * bce_loss + self.dice_w * dice_loss


def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return {"f1": f1.item(), "iou": iou.item(),
            "precision": precision.item(), "recall": recall.item()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    metrics    = {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    for batch in loader:
        t1   = batch["t1"].to(device)
        t2   = batch["t2"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(t1, t2)
        loss   = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        m = compute_metrics(logits, mask)
        for k in metrics:
            metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics    = {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    for batch in loader:
        t1   = batch["t1"].to(device)
        t2   = batch["t2"].to(device)
        mask = batch["mask"].to(device)

        logits = model(t1, t2)
        loss   = criterion(logits, mask)

        total_loss += loss.item()
        m = compute_metrics(logits, mask)
        for k in metrics:
            metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}