import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution 3×3 → BN → ReLU, bloc de base du U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool — descend d'un niveau dans l'encodeur."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)   # features avant pooling → skip connection
        pooled   = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Upsampling + ConvBlock.

    Reçoit la différence de features Siamese (skip) et les features
    du niveau précédent du décodeur, les concatène, puis applique ConvBlock.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concatène avec la skip connection
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    FC-Siam-diff : Siamese U-Net avec différence de features.

    L'encodeur est partagé entre T1 et T2 (mêmes poids).
    Les skip connections transmettent la différence |features_T1 - features_T2|
    au décodeur plutôt que les features brutes — c'est ce qui rend
    l'architecture sensible au changement à chaque échelle spatiale.
    """

    def __init__(self, in_channels=3, base_filters=32):
        super().__init__()

        # Encodeur partagé — 4 niveaux
        self.enc1 = EncoderBlock(in_channels,      base_filters)       # 256 → 128
        self.enc2 = EncoderBlock(base_filters,     base_filters * 2)   # 128 → 64
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)   # 64  → 32
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)   # 32  → 16

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Décodeur — reçoit les différences de features en skip
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8,  base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4,  base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2,  base_filters)

        # Tête de classification binaire
        self.head = nn.Conv2d(base_filters, 1, kernel_size=1)

    def encode(self, x):
        """Passe une image dans l'encodeur, retourne features + sortie poolée."""
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        return (s1, s2, s3, s4), x

    def forward(self, t1, t2):
        # Encodage de T1 et T2 avec les mêmes poids
        (s1_t1, s2_t1, s3_t1, s4_t1), x_t1 = self.encode(t1)
        (s1_t2, s2_t2, s3_t2, s4_t2), x_t2 = self.encode(t2)

        # Bottleneck sur la différence
        x = self.bottleneck(torch.abs(x_t1 - x_t2))

        # Décodeur avec différences de features en skip connections
        x = self.dec4(x, torch.abs(s4_t1 - s4_t2))
        x = self.dec3(x, torch.abs(s3_t1 - s3_t2))
        x = self.dec2(x, torch.abs(s2_t1 - s2_t2))
        x = self.dec1(x, torch.abs(s1_t1 - s1_t2))

        return self.head(x)  # logits (B, 1, H, W) — pas de sigmoid ici