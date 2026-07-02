import torch
from instinct_rl.modules.foothold_imagination import FootholdImaginationModel


def test_imagination_shapes_and_sigma_clamp():
    model = FootholdImaginationModel(input_size=32, hidden_sizes=(64, 32))
    x = torch.randn(8, 32)
    mu, log_sigma = model(x)
    assert mu.shape == (8, 2, 2)
    assert log_sigma.shape == (8, 2)
    assert torch.all(log_sigma <= 1.0 + 1e-6) and torch.all(log_sigma >= -4.0 - 1e-6)
