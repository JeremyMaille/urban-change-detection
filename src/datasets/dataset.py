import torch
from torch.utils.data import Dataset, Subset
from torchgeo.datasets import LEVIRCDPlus
import torchvision.transforms.functional as TF
import random


class LEVIRPatchDataset(Dataset):
    """
    Découpe les images LEVIR-CD (1024×1024) en patches 256×256.

    Deux modes :
    - train : crop aléatoire + augmentations géométriques
    - test  : grille fixe non-overlapping, reproductible
    """

    def __init__(self, root, split="train", patch_size=256):
        self.base = LEVIRCDPlus(root=root, split=split, download=False)
        self.split = split
        self.patch_size = patch_size
        self.index = self._build_index()

    def _build_index(self):
        p = self.patch_size
        n_per_image = (1024 // p) ** 2  # 16 patches par image 1024×1024
        index = []
        for img_idx in range(len(self.base)):
            for patch_idx in range(n_per_image):
                row = (patch_idx // (1024 // p)) * p
                col = (patch_idx %  (1024 // p)) * p
                index.append((img_idx, row, col))
        return index

    def _random_augment(self, t1, t2, mask):
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
            t1, t2, mask = self._random_augment(t1, t2, mask)

        return {"t1": t1, "t2": t2, "mask": mask}


def make_dataloaders(root, patch_size=256, batch_size=8, val_ratio=0.1, seed=42):
    """
    Construit les trois DataLoaders train/val/test.
    Le val est extrait du train (10%) avec un seed fixe pour la reproductibilité.
    """
    from torch.utils.data import DataLoader, random_split

    full_train = LEVIRPatchDataset(root=root, split="train", patch_size=patch_size)
    test_ds    = LEVIRPatchDataset(root=root, split="test",  patch_size=patch_size)

    n_val   = int(len(full_train) * val_ratio)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader