import torch
import torch.nn as nn
from torch.testing import assert_allclose
from models_conditional import ConditionalEmbeddings  # Replace with appropriate module import if needed


def test_conditional_embeddings_init():
    """Test the initialization of the ConditionalEmbeddings class."""
    input_channels = 3
    num_channels = 64
    emb_dim = 16
    model = ConditionalEmbeddings(input_channels, num_channels, emb_dim)

    assert isinstance(model.relu, nn.ReLU), "ReLU layer is not initialized correctly."
    assert isinstance(model.conv1, nn.Conv2d), "conv1 is not a Conv2d layer."
    assert model.conv1.in_channels == input_channels, "conv1 in_channels mismatch."
    assert model.conv1.out_channels == num_channels, "conv1 out_channels mismatch."

    assert isinstance(model.conv2, nn.Conv2d), "conv2 is not a Conv2d layer."
    assert model.conv2.in_channels == num_channels, "conv2 in_channels mismatch."
    assert model.conv2.out_channels == num_channels, "conv2 out_channels mismatch."

    assert isinstance(model.down, nn.AdaptiveAvgPool2d), "down is not an AdaptiveAvgPool2d layer."
    assert model.down.output_size == (emb_dim, emb_dim), "down output size mismatch."


def test_conditional_embeddings_forward():
    """Test the forward method of the ConditionalEmbeddings class."""
    input_channels = 3
    num_channels = 64
    emb_dim = 16
    batch_size = 2
    low_res_height, low_res_width = 32, 32

    # Initialize model
    model = ConditionalEmbeddings(input_channels, num_channels, emb_dim)

    # Create dummy inputs
    x = torch.randn(batch_size, input_channels, 64, 64)  # Random high-res input (not used in current forward)
    t = torch.randn(batch_size)  # Random time step (not used in current forward)
    y = torch.randn(batch_size, input_channels, low_res_height, low_res_width)  # Low-res image

    # Forward pass
    c_emb = model(x, t, y)

    # Assertions
    assert c_emb.shape == (batch_size, num_channels, emb_dim, emb_dim), \
        f"Expected output shape {(batch_size, num_channels, emb_dim, emb_dim)}, got {c_emb.shape}"

    assert c_emb.isfinite().all(), "Output contains non-finite values (NaN/Inf)."

    # Check intermediate values (optional)
    # For instance, check relu activation is applied correctly
    # In some cases, compare numerical results as well


if __name__ == '__main__':
    import pytest

    pytest.main([__file__])
