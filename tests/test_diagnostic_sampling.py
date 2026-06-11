import torch

from fm_lab.couplings import MinibatchOTCoupling
from fm_lab.data import TwoMoons
from fm_lab.experiments.sampling import sample_path_batch
from fm_lab.paths import LinearPath
from fm_lab.sources import GaussianSource


def test_sample_path_batch_chunks_minibatch_ot() -> None:
    samples = sample_path_batch(
        source=GaussianSource(dim=2),
        target=TwoMoons(noise=0.0),
        coupling=MinibatchOTCoupling(max_exact_size=8),
        path=LinearPath(),
        n_samples=20,
        t_value=0.5,
        device=torch.device("cpu"),
    )

    assert samples["xt"].shape == (20, 2)
    assert samples["velocities"].shape == (20, 2)
    assert torch.allclose(samples["t"], torch.full((20,), 0.5))
