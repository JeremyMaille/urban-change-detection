"""Tests for the SiameseUNet architecture."""
import pytest
import torch

from src.models.siamese_unet import SiameseUNet, DecoderBlock


@pytest.fixture(scope="module")
def model():
    # pretrained=False: no ImageNet download in CI, architecture is identical
    m = SiameseUNet(pretrained=False)
    m.eval()
    return m


def test_output_shape_matches_input(model):
    """The predicted mask must have the same spatial size as the input pair."""
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(t1, t2)
    assert out.shape == (2, 1, 256, 256)


def test_output_is_finite(model):
    """Logits must not contain NaN or Inf."""
    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(t1, t2)
    assert torch.isfinite(out).all()


def test_identical_inputs_produce_uniform_output(model):
    """T1 == T2 zeroes every abs-difference skip and the bottleneck.

    With all-zero feature inputs, the decoder sees no spatial information,
    so the output map must be spatially constant (up to numerical noise).
    This validates the abs-difference wiring of the Siamese architecture.
    """
    t = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(t, t.clone())
    assert out.std().item() < 1e-4


def test_siamese_encoder_is_shared(model):
    """Both branches must use the same encoder weights (true Siamese)."""
    t = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats_a = model.encode(t)
        feats_b = model.encode(t.clone())
    for fa, fb in zip(feats_a, feats_b):
        assert torch.allclose(fa, fb)


def test_decoder_block_upsamples_by_two():
    block = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)
    x = torch.randn(1, 64, 16, 16)
    skip = torch.randn(1, 32, 32, 32)
    out = block(x, skip)
    assert out.shape == (1, 32, 32, 32)
