import numpy as np
import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling, pair_with_condition
from fm_lab.experiments.factory import build_model
from fm_lab.models import ImageUNetVelocity, MLPVelocity, SwitchableLowRankConv2d
from fm_lab.paths import LinearPath
from fm_lab.solvers import EulerSolver
from fm_lab.sources import GaussianSource
from fm_lab.training.losses import FlowMatchingObjective
from fm_lab.training.prediction import classifier_free_guided_prediction
from fm_lab.training.trainer import _sample_training_batch, sample_and_plot


class LabelVelocity(nn.Module):
    is_class_conditional = True

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t
        labels = context["class_labels"].to(x.dtype)
        values = torch.where(labels < 0, torch.zeros_like(labels), labels + 1.0)
        return values[:, None].expand_as(x)


class LabeledTarget:
    dim = 2

    def sample(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.dim, device=device)

    def sample_with_labels(self, n: int, device=None):
        labels = torch.arange(n, device=device) % 2
        return labels[:, None].expand(n, self.dim).float(), labels

    def metadata(self) -> dict:
        return {"name": "labeled", "dim": self.dim}


def test_couplings_preserve_target_condition_alignment() -> None:
    x0 = torch.tensor([[0.0], [10.0], [20.0]])
    x1 = torch.tensor([[19.0], [1.0], [11.0]])
    labels = torch.tensor([19, 1, 11])

    _, paired_x1, paired_labels = pair_with_condition(
        MinibatchOTCoupling(), x0, x1, labels
    )

    assert torch.equal(paired_x1[:, 0].long(), paired_labels)

    torch.manual_seed(4)
    _, shuffled_x1, shuffled_labels = pair_with_condition(
        IndependentCoupling(shuffle_target=True), x0, x1, labels
    )
    assert torch.equal(shuffled_x1[:, 0].long(), shuffled_labels)


def test_conditional_models_accept_classes_and_null_token() -> None:
    mlp = MLPVelocity(
        dim=3,
        hidden_dim=8,
        depth=1,
        time_embedding_dim=4,
        num_classes=2,
    )
    image = ImageUNetVelocity(
        dim=16,
        image_shape=(4, 4),
        base_channels=4,
        time_embedding_dim=8,
        num_classes=2,
    )

    mlp_output = mlp(
        torch.zeros(2, 3), torch.zeros(2), {"class_labels": torch.tensor([0, -1])}
    )
    image_output = image(
        torch.zeros(2, 16), torch.zeros(2), {"class_labels": torch.tensor([1, -1])}
    )
    assert mlp_output.shape == (2, 3)
    assert image_output.shape == (2, 16)

    cifar_image = ImageUNetVelocity(
        dim=3 * 32 * 32,
        image_shape=(3, 32, 32),
        base_channels=32,
        time_embedding_dim=128,
        num_classes=10,
    )
    assert sum(parameter.numel() for parameter in cifar_image.parameters()) == 1_078_569
    assert not any(
        isinstance(module, SwitchableLowRankConv2d)
        for module in cifar_image.modules()
    )
    assert cifar_image.capacity_metadata()["enabled"] is False


def test_flow_matching_objective_passes_class_labels() -> None:
    labels = torch.tensor([0, 1])
    loss, _ = FlowMatchingObjective()(
        model=LabelVelocity(),
        path=LinearPath(),
        x0=torch.zeros(2, 2),
        x1=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        t=torch.full((2,), 0.5),
        class_labels=labels,
    )
    assert torch.allclose(loss, torch.tensor(0.0))


def test_training_batch_preserves_labels_before_cfg_dropout() -> None:
    _, _, _, conditioned_labels, original_labels = _sample_training_batch(
        source=GaussianSource(dim=2),
        target=LabeledTarget(),
        coupling=IndependentCoupling(shuffle_target=False),
        batch_size=4,
        device=torch.device("cpu"),
        class_conditional=True,
        condition_dropout=1.0,
    )

    assert torch.equal(conditioned_labels, torch.full((4,), -1))
    assert torch.equal(original_labels, torch.tensor([0, 1, 0, 1]))


def test_training_batch_supports_official_batch_level_cfg_dropout() -> None:
    _, _, _, kept_labels, kept_original = _sample_training_batch(
        source=GaussianSource(dim=2),
        target=LabeledTarget(),
        coupling=IndependentCoupling(shuffle_target=False),
        batch_size=4,
        device=torch.device("cpu"),
        class_conditional=True,
        condition_dropout=0.0,
        condition_dropout_mode="batch",
    )
    _, _, _, dropped_labels, dropped_original = _sample_training_batch(
        source=GaussianSource(dim=2),
        target=LabeledTarget(),
        coupling=IndependentCoupling(shuffle_target=False),
        batch_size=4,
        device=torch.device("cpu"),
        class_conditional=True,
        condition_dropout=1.0,
        condition_dropout_mode="batch",
    )

    assert torch.equal(kept_labels, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(kept_original, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(dropped_labels, torch.full((4,), -1))
    assert torch.equal(dropped_original, torch.tensor([0, 1, 0, 1]))


def test_classifier_free_guidance_combines_velocity_predictions() -> None:
    prediction = classifier_free_guided_prediction(
        LabelVelocity(),
        torch.zeros(2, 2),
        torch.zeros(2),
        class_labels=torch.tensor([0, 1]),
        guidance_scale=2.0,
    )
    assert torch.equal(prediction, torch.tensor([[2.0, 2.0], [4.0, 4.0]]))


def test_conditional_sampling_saves_requested_generated_labels(tmp_path) -> None:
    config = {
        "experiment": {"seed": 3},
        "conditioning": {"enabled": True, "num_classes": 2},
        "objective": {"name": "flow_matching"},
        "sampling": {
            "n_samples": 5,
            "n_trajectories": 2,
            "nfe": 1,
            "classes": [1, 0],
            "classifier_free_guidance": {"scale": 2.0},
        },
        "solvers": {"nfes": [1]},
    }
    model = build_model(
        {
            "conditioning": {"enabled": True, "num_classes": 2},
            "model": {"name": "mlp", "hidden_dim": 8, "depth": 1},
        },
        dim=2,
    )

    sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=LabeledTarget(),
        source=GaussianSource(dim=2),
        path=LinearPath(),
        model=model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert np.array_equal(
        np.load(tmp_path / "samples" / "generated_labels.npy"),
        np.asarray([1, 0, 1, 0, 1]),
    )
