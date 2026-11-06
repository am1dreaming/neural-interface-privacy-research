import pytest
import torch

from neurotext.models import ENCODERS, EEGEncoder, build_encoder, model_options

N_CH, N_T = 8, 128


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_every_encoder_returns_unit_vectors_of_the_requested_size(name):
    enc = build_encoder(name, N_CH, N_T, out_dim=12, d_model=32, n_heads=2, patch=16)
    x = torch.randn(5, N_CH, N_T)
    enc.train()
    z = enc(x)
    assert z.shape == (5, 12) and torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    enc.eval()
    assert enc(x[:1]).shape == (1, 12)           # batch of one in eval mode
    assert enc.n_channels == N_CH and enc.out_dim == 12
    assert callable(enc.prepare)


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_gradients_reach_every_parameter(name):
    enc = build_encoder(name, N_CH, N_T, out_dim=8, d_model=32, n_heads=2, patch=16)
    enc.train()
    enc(torch.randn(6, N_CH, N_T)).sum().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert not missing


def test_options_an_encoder_does_not_know_are_ignored_and_unknown_names_fail():
    build_encoder("linear", N_CH, N_T, out_dim=4, d_model=64, patch=16, n_layers=2)
    with pytest.raises(ValueError, match="unknown encoder"):
        build_encoder("resnet", N_CH, N_T)


def test_model_options_defaults_to_the_transformer_so_old_configs_still_work():
    assert model_options({"d_model": 64, "out_dim": 32}) == ("transformer", {"d_model": 64, "out_dim": 32})
    assert model_options({"name": "cnn", "out_dim": 8}) == ("cnn", {"out_dim": 8})
    assert isinstance(build_encoder(*[model_options({"out_dim": 8, "patch": 16})[0], N_CH, N_T],
                                    out_dim=8, patch=16), EEGEncoder)


def test_eegnet_rejects_windows_that_are_too_short():
    with pytest.raises(ValueError, match="too short"):
        build_encoder("eegnet", N_CH, 16)


def test_the_alignment_loop_trains_every_encoder_without_changes():
    import numpy as np

    from neurotext.models import TextProjector
    from neurotext.training import embed, fit_alignment

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, N_CH, N_T)).astype("float32")
    sent = np.repeat(np.arange(10), 4)
    text = rng.standard_normal((10, 6)).astype("float32")
    for name in ENCODERS:
        enc = build_encoder(name, N_CH, N_T, out_dim=8, d_model=32, n_heads=2, patch=16)
        hist = fit_alignment(enc, TextProjector(6, 8), x, sent, text, epochs=2, batch_size=16)
        assert len(hist) == 2 and embed(enc, x).shape == (40, 8)
