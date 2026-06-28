import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def change_vector_analysis(t1, t2):
    """
    Change Vector Analysis (CVA) baseline non-ML classique en télédétection.

    Pour chaque pixel, calcule la magnitude de la différence spectrale entre
    T1 et T2. Un pixel avec une grande différence est probablement un changement.

    Args:
        t1   : tensor (B, C, H, W) image avant
        t2   : tensor (B, C, H, W) image après
    Returns:
        magnitude : tensor (B, 1, H, W) — score de changement par pixel
    """
    diff = t2 - t1                                    # différence pixel par pixel
    magnitude = torch.norm(diff, dim=1, keepdim=True) # norme L2 sur les canaux
    return magnitude


def otsu_threshold(magnitude):
    """
    Seuillage d'Otsu trouve automatiquement le seuil optimal qui sépare
    les pixels 'changé' des pixels 'stable' en maximisant la variance inter-classe.

    Args:
        magnitude : tensor (B, 1, H, W)
    Returns:
        binary_mask : tensor (B, 1, H, W) — 0 ou 1
    """
    mag_np = magnitude.squeeze(1).cpu().numpy()  # (B, H, W)
    masks  = []

    for b in range(mag_np.shape[0]):
        flat = mag_np[b].flatten()

        # Calcul du seuil d'Otsu manuellement sur l'histogramme
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
    Calcule F1, IoU, précision et rappel sur la classe 'changé'.

    Args:
        pred   : tensor (B, 1, H, W) — prédictions binaires
        target : tensor (B, 1, H, W) — masques ground truth
    Returns:
        dict de métriques
    """
    p = pred.cpu().numpy().flatten().astype(int)
    t = target.cpu().numpy().flatten().astype(int)

    return {
        "f1"        : f1_score(t, p, zero_division=0),
        "iou"       : jaccard_score(t, p, zero_division=0),
        "precision" : precision_score(t, p, zero_division=0),
        "recall"    : recall_score(t, p, zero_division=0),
    }