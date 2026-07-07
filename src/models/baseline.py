import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def change_vector_analysis(t1, t2):
    """
    Change Vector Analysis (CVA), a classic non-ML remote sensing baseline.

    For each pixel, computes the magnitude of the spectral difference between
    T1 and T2. A pixel with a large difference is likely a change.

    Args:
        t1   : tensor (B, C, H, W) before image
        t2   : tensor (B, C, H, W) after image
    Returns:
        magnitude : tensor (B, 1, H, W), per-pixel change score
    """
    diff = t2 - t1                                    # pixel-by-pixel difference
    magnitude = torch.norm(diff, dim=1, keepdim=True) # L2 norm over channels
    return magnitude


def otsu_threshold(magnitude):
    """
    Otsu thresholding automatically finds the optimal threshold that separates
    'changed' pixels from 'stable' pixels by maximizing inter-class variance.

    Args:
        magnitude : tensor (B, 1, H, W)
    Returns:
        binary_mask : tensor (B, 1, H, W), 0 or 1
    """
    mag_np = magnitude.squeeze(1).cpu().numpy()  # (B, H, W)
    masks  = []

    for b in range(mag_np.shape[0]):
        flat = mag_np[b].flatten()

        # Manually compute the Otsu threshold from the histogram
        counts, bin_edges = np.histogram(flat, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = counts.sum()
        best_thresh, best_var = 0, 0

        w0, sum0 = 0, 0
        total_sum = (counts * bin_centers).sum()

        for i in range(len(counts)):
            w0    += counts[i]
            w1     = total - w0
            if w0 == 0 or w1 == 0:
                continue
            sum0  += counts[i] * bin_centers[i]
            mu0    = sum0 / w0
            mu1    = (total_sum - sum0) / w1
            var    = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
            if var > best_var:
                best_var    = var
                best_thresh = bin_centers[i]

        masks.append((mag_np[b] > best_thresh).astype(np.float32))

    return torch.tensor(np.stack(masks, axis=0)).unsqueeze(1)


def compute_metrics(pred, target):
    """
    Computes F1, IoU, precision and recall on the 'changed' class.

    Args:
        pred   : tensor (B, 1, H, W), binary predictions
        target : tensor (B, 1, H, W), ground truth masks
    Returns:
        dict of metrics
    """
    p = pred.cpu().numpy().flatten().astype(int)
    t = target.cpu().numpy().flatten().astype(int)

    return {
        "f1"        : f1_score(t, p, zero_division=0),
        "iou"       : jaccard_score(t, p, zero_division=0),
        "precision" : precision_score(t, p, zero_division=0),
        "recall"    : recall_score(t, p, zero_division=0),
    }