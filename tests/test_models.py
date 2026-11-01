import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neurotext.data import make_synthetic
from neurotext.efficiency import (
    count_parameters,
    estimate_int8_linear_bytes,
    quantized_size_bytes,
    serialized_size_bytes,
)
from neurotext.models import AdaptedEncoder, EEGEncoder, SubjectAdapter, TextProjector, freeze, info_nce
from neurotext.training import embed, fit_adapter, fit_alignment


def test_encoder_output_is_unit_norm_and_right_shape():
    enc = EEGEncoder(8, 64, d_model=32, out_dim=16, patch=16)
    z = enc(torch.randn(5, 8, 64))
    assert z.shape == (5, 16)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_encoder_rejects_bad_patch():
    with pytest.raises(ValueError):
        EEGEncoder(8, 60, patch=16)


def test_info_nce_does_not_punish_same_sentence_pairs():
    """Two recordings of one sentence in a batch are not negatives of each other."""
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]]), dim=-1)
    ids_same = torch.tensor([0, 0, 1])
    ids_diff = torch.tensor([0, 1, 2])
    # identical embeddings for ids 0/1: as 'different sentences' they are impossible to
    # separate (high loss); as the same sentence the loss must be lower
    assert info_nce(z, z, ids_same) < info_nce(z, z, ids_diff)


def test_adapter_is_identity_at_init():
    enc = freeze(EEGEncoder(8, 64, d_model=32, out_dim=16))
    adapted = AdaptedEncoder(enc, SubjectAdapter(8, 16, rank=2)).eval()
    x = torch.randn(4, 8, 64)
    assert torch.allclose(enc(x), adapted(x), atol=1e-6)


def test_adapter_is_tiny_compared_with_base():
    enc = EEGEncoder(16, 128)
    ad = SubjectAdapter(16, 32, rank=4)
    assert ad.n_parameters() < 0.05 * count_parameters(enc)


def test_freeze_stops_gradients():
    enc = freeze(EEGEncoder(8, 64, d_model=32, out_dim=16))
    assert all(not p.requires_grad for p in enc.parameters())
    assert not enc.training


def test_fit_adapter_leaves_base_model_untouched():
    d = make_synthetic(n_subjects=3, n_sentences=12, n_channels=8, n_times=64, text_dim=8, seed=0)
    enc = EEGEncoder(8, 64, d_model=32, out_dim=16)
    proj = TextProjector(8, 16)
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    fit_adapter(enc, proj, d.x[:20], d.sentence_ids[:20], d.text_emb, epochs=2)
    after = enc.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def _tiny_setup():
    d = make_synthetic(n_subjects=2, n_sentences=8, n_channels=8, n_times=64, text_dim=8, seed=0)
    enc = EEGEncoder(8, 64, d_model=32, out_dim=16).eval()
    proj = TextProjector(8, 16)
    return d, enc, proj


def test_channel_only_adapter_leaves_lowrank_untouched():
    d, enc, proj = _tiny_setup()
    ad = fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, epochs=3, components=("channel",))
    assert torch.count_nonzero(ad.adapter.up.weight) == 0
    assert not torch.equal(ad.adapter.scale, torch.ones_like(ad.adapter.scale))


def test_lowrank_only_adapter_leaves_channels_untouched():
    d, enc, proj = _tiny_setup()
    ad = fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, epochs=3, components=("lowrank",))
    assert torch.equal(ad.adapter.scale, torch.ones_like(ad.adapter.scale))
    assert torch.count_nonzero(ad.adapter.bias) == 0
    assert torch.count_nonzero(ad.adapter.up.weight) > 0


def test_identity_regularisation_keeps_adapter_closer_to_identity():
    d, enc, proj = _tiny_setup()
    kw = dict(epochs=20, lr=1e-2)

    def distance(reg):
        a = fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, identity_reg=reg, **kw).adapter
        with torch.no_grad():
            return float(((a.scale - 1) ** 2).sum() + (a.bias ** 2).sum() + (a.up.weight ** 2).sum())

    assert distance(10.0) < distance(0.0)


def test_mixing_only_adapter_trains_only_mixing_parameters():
    d, enc, proj = _tiny_setup()
    ad = fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, epochs=3, components=("mixing",)).adapter
    assert torch.count_nonzero(ad.mix_up.weight) > 0
    assert torch.equal(ad.scale, torch.ones_like(ad.scale)) and torch.count_nonzero(ad.up.weight) == 0


def test_mixing_adapter_is_identity_at_init_and_can_recombine_channels():
    ad = SubjectAdapter(4, 8, rank=2, mix_rank=2)
    x = torch.randn(3, 4, 16)
    assert torch.allclose(ad.transform_input(x), x)
    with torch.no_grad():  # make the mixing non-trivial
        ad.mix_down.weight.copy_(torch.eye(2, 4))
        ad.mix_up.weight.copy_(torch.eye(4, 2))
    y = ad.transform_input(x)
    assert torch.allclose(y[:, 0], 2 * x[:, 0]) and torch.allclose(y[:, 3], x[:, 3])


def test_negative_pool_must_contain_calibration_sentences():
    d, enc, proj = _tiny_setup()
    with pytest.raises(ValueError, match="missing from negative_pool"):
        fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, negative_pool=np.array([7]))


def test_negative_pool_training_runs_and_is_deterministic():
    d, enc, proj = _tiny_setup()
    kw = dict(epochs=3, negative_pool=np.arange(8), seed=3)
    a = fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, **kw)
    b = fit_adapter(enc, proj, d.x[:16], d.sentence_ids[:16], d.text_emb, **kw)
    assert np.allclose(embed(a, d.x[:8]), embed(b, d.x[:8]))


def test_unknown_component_is_rejected():
    d, enc, proj = _tiny_setup()
    with pytest.raises(ValueError, match="unknown adapter components"):
        fit_adapter(enc, proj, d.x[:4], d.sentence_ids[:4], d.text_emb, components=("everything",))


def test_zero_calibration_equals_base_model():
    d = make_synthetic(n_subjects=2, n_sentences=6, n_channels=8, n_times=64, text_dim=8, seed=0)
    enc = EEGEncoder(8, 64, d_model=32, out_dim=16).eval()
    proj = TextProjector(8, 16)
    adapted = fit_adapter(enc, proj, d.x[:0], d.sentence_ids[:0], d.text_emb)
    assert np.allclose(embed(enc, d.x[:6]), embed(adapted, d.x[:6]), atol=1e-5)


def test_alignment_training_reduces_loss_and_beats_chance_on_easy_synthetic_data():
    d = make_synthetic(n_subjects=6, n_sentences=24, n_channels=8, n_times=64, text_dim=8,
                       n_sources=4, snr=3.0, subject_variability=0.0, seed=0)
    enc = EEGEncoder(8, 64, d_model=32, out_dim=16, n_layers=1)
    proj = TextProjector(8, 16)
    hist = fit_alignment(enc, proj, d.x, d.sentence_ids, d.text_emb, epochs=25, batch_size=48, seed=0)
    assert hist[-1] < hist[0]

    z = embed(enc, d.x)
    cand = proj(torch.as_tensor(d.text_emb)).detach().numpy()
    top1 = (np.argmax(z @ cand.T, axis=1) == d.sentence_ids).mean()
    assert top1 > 3 / 24  # well above chance (1/24) on the training sentences


def test_quantization_shrinks_the_model():
    enc = EEGEncoder(16, 128, d_model=64, out_dim=32)
    measured = quantized_size_bytes(enc)
    if measured is None:
        pytest.skip("this PyTorch build has no working quantization engine")
    assert measured < serialized_size_bytes(enc)


def test_int8_estimate_is_smaller_than_fp32_and_close_to_measurement():
    enc = EEGEncoder(16, 128, d_model=64, out_dim=32)
    est = estimate_int8_linear_bytes(enc)
    assert est < 4 * count_parameters(enc)
    measured = quantized_size_bytes(enc)
    if measured is not None:
        assert abs(est - measured) / measured < 0.1  # serialization overhead only


def test_quantization_without_engine_returns_none(monkeypatch):
    """Simulates the default macOS wheels, where quantize_dynamic raises NoQEngine."""
    def boom(*a, **k):
        raise RuntimeError("Didn't find engine for operation quantized::linear_prepack NoQEngine")

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", boom)
    assert quantized_size_bytes(EEGEncoder(16, 128)) is None
