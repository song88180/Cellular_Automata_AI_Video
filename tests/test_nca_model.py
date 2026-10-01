import numpy as np
import pytest

from sim.nca import model as np_model


def test_numpy_matches_torch_reference_step(tmp_path):
    """The hand-ported numpy runtime must match the torch training model
    exactly (same weights, same update mask). Uses random weights so it's
    independent of any trained checkpoint."""
    torch = pytest.importorskip("torch")
    from sim.nca.train import TorchNCA

    torch.manual_seed(0)
    net = TorchNCA()
    torch.nn.init.normal_(net.w2.weight, std=0.1)  # zero-init would hide bugs
    path = tmp_path / "w.npz"
    net.export(path)
    nca = np_model.NCA.load(path)

    rng = np.random.default_rng(0)
    h, w = 24, 24
    x = np.zeros((h, w, np_model.CHANNELS), dtype=np.float32)
    x[6:18, 6:18] = rng.normal(size=(12, 12, np_model.CHANNELS)).astype(np.float32)
    x[..., 3] = np.abs(x[..., 3])
    for _ in range(3):
        mask = rng.random((h, w)) <= 0.5
        ref = net(
            torch.from_numpy(x).permute(2, 0, 1)[None],
            update_mask=torch.from_numpy(mask)[None, None],
        )[0].permute(1, 2, 0).detach().numpy()
        out = nca.step(x, update_mask=mask)
        np.testing.assert_allclose(out, ref, atol=1e-5)
        x = out


def test_trained_model_grows_and_regenerates():
    if not np_model.WEIGHTS_PATH.exists():
        pytest.skip("no trained weights yet")
    nca = np_model.NCA.load()
    rng = np.random.default_rng(1)
    x = np_model.seed_state(72, 72)
    for _ in range(200):
        x = nca.step(x, rng)
    grown_alpha = x[..., 3] > 0.1
    assert grown_alpha.sum() > 300  # a whole lizard, not a dot / not dead

    x[:, 36:] = 0.0
    for _ in range(200):
        x = nca.step(x, rng)
    regrown_alpha = x[..., 3] > 0.1
    overlap = (grown_alpha & regrown_alpha).sum() / grown_alpha.sum()
    assert overlap > 0.9
    assert regrown_alpha.sum() < 1.3 * grown_alpha.sum()  # didn't explode
