import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Bilinear upsampling + ConvBlock.
    Receives the features from the previous level and the skip connection difference.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net with an ImageNet-pretrained ResNet34 encoder.

    The ResNet34 encoder is shared between T1 and T2 (same weights).
    Skip connections carry the absolute feature difference at each level,
    giving change sensitivity at every spatial scale.
    Dropout2d in the decoder regularizes against overfitting.
    """

    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()

        # Pretrained ResNet34 encoder
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Extract the encoder blocks to access the skip connections
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64ch, /2
        self.pool = backbone.maxpool                                              # /4
        self.enc1 = backbone.layer1   # 64ch,  /4
        self.enc2 = backbone.layer2   # 128ch, /8
        self.enc3 = backbone.layer3   # 256ch, /16
        self.enc4 = backbone.layer4   # 512ch, /32

        # Decoder, channels matched to the ResNet34 outputs
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)

        # Upsample back to the original resolution (final x2)
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Binary classification head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def encode(self, x):
        s0 = self.enc0(x)       # (B, 64,  H/2,  W/2)
        x  = self.pool(s0)      # (B, 64,  H/4,  W/4)
        s1 = self.enc1(x)       # (B, 64,  H/4,  W/4)
        s2 = self.enc2(s1)      # (B, 128, H/8,  W/8)
        s3 = self.enc3(s2)      # (B, 256, H/16, W/16)
        s4 = self.enc4(s3)      # (B, 512, H/32, W/32)
        return s0, s1, s2, s3, s4

    def forward(self, t1, t2):
        s0_t1, s1_t1, s2_t1, s3_t1, s4_t1 = self.encode(t1)
        s0_t2, s1_t2, s2_t2, s3_t2, s4_t2 = self.encode(t2)

        # Bottleneck: difference of the deepest features
        x = torch.abs(s4_t1 - s4_t2)

        # Decoder with feature differences as skip connections
        x = self.dec4(x, torch.abs(s3_t1 - s3_t2))
        x = self.dec3(x, torch.abs(s2_t1 - s2_t2))
        x = self.dec2(x, torch.abs(s1_t1 - s1_t2))
        x = self.dec1(x, torch.abs(s0_t1 - s0_t2))

        x = self.up_final(x)

        return self.head(x)  # logits (B, 1, H, W)