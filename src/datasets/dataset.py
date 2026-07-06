import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchgeo.datasets import LEVIRCDPlus
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


# Normalisation ImageNet requise par l'encodeur ResNet34 pré-entraîné
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class LEVIRPatchDataset(Dataset):
    """
    Découpe les images LEVIR-CD (1024×1024) en patches 256×256.

    La séparation train/val se fait au niveau des images (pas des patches)
    pour éviter le data leakage  deux patches d'une même image
    ne peuvent pas se retrouver dans train et val simultanément.
    """

    def __init__(self, base_dataset, split="train", patch_size=256):
        self.base       = base_dataset
        self.split      = split
        self.patch_size = patch_size
        self.index      = self._build_index()

    def _build_index(self):
        p = self.patch_size
        n_per_image = (1024 // p) ** 2
        index = []
        for img_idx in range(len(self.base)):
            for patch_idx in range(n_per_image):
                row = (patch_idx // (1024 // p)) * p
                col = (patch_idx %  (1024 // p)) * p
                index.append((img_idx, row, col))
        return index

    def _augment(self, t1, t2, mask):
        # Augmentations géométriques uniquement — pas de color jitter
        # qui casserait la cohérence spectrale entre T1 et T2.
        if random.random() > 0.5:
            t1 = TF.hflip(t1); t2 = TF.hflip(t2); mask = TF.hflip(mask)
        if random.random() > 0.5:
            t1 = TF.vflip(t1); t2 = TF.vflip(t2); mask = TF.vflip(mask)
        k = random.randint(0, 3)
        if k > 0:
            t1   = torch.rot90(t1,   k, dims=[1, 2])
            t2   = torch.rot90(t2,   k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
        return t1, t2, mask

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_idx, row, col = self.index[idx]
        p = self.patch_size

        sample = self.base[img_idx]
        t1   = sample["image"][0]
        t2   = sample["image"][1]
        mask = sample["mask"].float()

        if self.split == "train":
            row = random.randint(0, 1024 - p)
            col = random.randint(0, 1024 - p)

        t1   = t1  [:, row:row+p, col:col+p]
        t2   = t2  [:, row:row+p, col:col+p]
        mask = mask[:, row:row+p, col:col+p]

        if self.split == "train":
            t1, t2, mask = self._augment(t1, t2, mask)

        # Normalisation ImageNet pour l'encodeur ResNet34 pré-entraîné
        t1 = normalize(t1)
        t2 = normalize(t2)

        return {"t1": t1, "t2": t2, "mask": mask}


def make_dataloaders(root, patch_size=256, batch_size=8, val_ratio=0.1, seed=42):
    """
    Construit les trois DataLoaders train/val/test.

    La séparation train/val se fait au niveau des images source
    pour éviter le data leakage entre patches d'une même image.
    """
    full_base  = LEVIRCDPlus(root=root, split="train", download=False)
    test_base  = LEVIRCDPlus(root=root, split="test",  download=False)

    # Split train/val au niveau des images
    n_images = len(full_base)
    n_val    = max(1, int(n_images * val_ratio))
    n_train  = n_images - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_images), [n_train, n_val], generator=generator
    )

    # Sous-datasets indexés par image
    class SubsetBase:
        def __init__(self, base, indices):
            self.base    = base
            self.indices = list(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.base[self.indices[idx]]

    train_base = SubsetBase(full_base, train_indices)
    val_base   = SubsetBase(full_base, val_indices)

    train_ds = LEVIRPatchDataset(train_base, split="train", patch_size=patch_size)
    val_ds   = LEVIRPatchDataset(val_base,   split="val",   patch_size=patch_size)
    test_ds  = LEVIRPatchDataset(test_base,  split="test",  patch_size=patch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader