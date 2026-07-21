from __future__ import annotations

import copy

import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMObjective,
    OfficialImbDiffCMUNet,
    load_official_imbdiff_cm_components,
    sample_official_imbdiff_cm,
)
from fm_lab.paths import DiscreteDDPMPath
from fm_lab.solvers import EulerSolver
from fm_lab.training.trainer import train_flow_matching


def _tiny_model() -> OfficialImbDiffCMUNet:
    return OfficialImbDiffCMUNet(
        dim=3 * 4 * 4,
        image_shape=(3, 4, 4),
        timesteps=8,
        base_channels=32,
        channel_multipliers=(1,),
        attention_levels=(),
        num_res_blocks=1,
        dropout=0.0,
        num_classes=2,
        rank_ratio=0.1,
        capacity_parts=("up",),
    )


def test_official_objective_matches_released_trainer_loss_and_gradients() -> None:
    adapter_model = _tiny_model()
    direct_model = copy.deepcopy(adapter_model)
    counts = (3, 1)
    objective = OfficialImbDiffCMObjective(
        class_counts=counts,
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        consistency_weight=1.0,
        diversity_weight=0.2,
        image_shape=(3, 4, 4),
    )
    # Construct both trainer wrappers before fixing the stochastic training seed.
    objective._trainer_for(adapter_model, torch.device("cpu"))
    probabilities = torch.tensor(counts, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum()
    components = load_official_imbdiff_cm_components()
    direct_trainer = components.cm_trainer(
        model=direct_model,
        beta_1=1e-4,
        beta_T=1e-2,
        T=8,
        dataset=None,
        num_class=2,
        cfg=False,
        weight=probabilities.unsqueeze(0),
        transfer_x0=False,
        transfer_tr_tau=False,
        transfer_mode="t2h",
        label_weight_tr=probabilities.unsqueeze(1) @ probabilities.unsqueeze(0),
        w_con=1.0,
        w_div=0.2,
    )

    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape(2, -1)
    labels = torch.tensor([0, 1])
    torch.manual_seed(17)
    adapter_loss, _ = objective(
        model=adapter_model,
        path=None,
        x0=torch.zeros_like(clean),
        x1=clean,
        t=torch.zeros(2),
        class_labels=labels,
        original_class_labels=labels,
    )
    adapter_loss.backward()

    torch.manual_seed(17)
    direct_loss = direct_trainer(clean.reshape(2, 3, 4, 4), labels)
    direct_loss.backward()

    assert torch.equal(adapter_loss, direct_loss)
    for (adapter_name, adapter_parameter), (direct_name, direct_parameter) in zip(
        adapter_model.named_parameters(),
        direct_model.named_parameters(),
        strict=True,
    ):
        assert adapter_name == direct_name
        assert (adapter_parameter.grad is None) == (direct_parameter.grad is None)
        if adapter_parameter.grad is not None:
            assert direct_parameter.grad is not None
            assert torch.equal(adapter_parameter.grad, direct_parameter.grad)


def test_official_adapter_maps_batch_null_label_to_no_embedding() -> None:
    model = _tiny_model().eval()
    x = torch.randn(2, 3 * 4 * 4)
    t = torch.tensor([2, 5])

    with torch.no_grad():
        adapted = model(x, t, context={"class_labels": torch.tensor([-1, -1])})
        direct = model.network(x.reshape(2, 3, 4, 4), t, y=None).reshape(2, -1)

    assert torch.equal(adapted, direct)


class _RecordingOfficialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.capacity_flags: list[bool] = []

    def forward(self, x, t, y=None, augm=None, use_cm=True):
        del t, y, augm
        self.capacity_flags.append(bool(use_cm))
        return torch.zeros_like(x) + self.anchor * 0.0


def test_release_sampler_uses_capacity_on_default_branch() -> None:
    model = _RecordingOfficialModel()
    samples = sample_official_imbdiff_cm(
        model=model,
        initial_noise=torch.randn(2, 3 * 4 * 4),
        class_labels=torch.tensor([0, 1]),
        timesteps=4,
        beta_end=1e-2,
        omega=0.0,
        method="ddim",
        ddim_skip=2,
        image_shape=(3, 4, 4),
    )

    assert samples.shape == (2, 3 * 4 * 4)
    assert model.capacity_flags
    assert all(model.capacity_flags)


class _TinySource:
    dim = 3 * 4 * 4

    def sample(self, n: int, device=None):
        return torch.randn(n, self.dim, device=device)

    def metadata(self):
        return {"name": "tiny_source", "dim": self.dim}


class _TinyTarget:
    dim = 3 * 4 * 4
    class_counts = (2, 2)

    def sample(self, n: int, device=None):
        return self.sample_with_labels(n, device=device)[0]

    def sample_with_labels(self, n: int, device=None):
        images = torch.rand(n, self.dim, device=device) * 2.0 - 1.0
        labels = torch.arange(n, device=device) % 2
        return images, labels

    def all_samples_with_labels(self, device=None):
        images = torch.linspace(-1.0, 1.0, 4 * self.dim).reshape(4, self.dim)
        labels = torch.tensor([0, 0, 1, 1])
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        return images, labels, torch.arange(4).numpy().astype(str)

    def metadata(self):
        return {
            "name": "tiny_target",
            "dim": self.dim,
            "image_shape": [3, 4, 4],
            "image_value_range": [-1.0, 1.0],
        }


def test_official_path_trains_and_samples_through_fm_lab(tmp_path) -> None:
    config = {
        "experiment": {"seed": 3},
        "source": {"name": "gaussian", "dim": 3 * 4 * 4},
        "coupling": {"name": "independent"},
        "path": {"name": "discrete_ddpm"},
        "model": {"name": "official_imbdiff_cm_unet"},
        "conditioning": {
            "enabled": True,
            "num_classes": 2,
            "dropout_probability": 0.0,
            "dropout_mode": "batch",
        },
        "diffusion": {
            "timesteps": 4,
            "beta_start": 1e-4,
            "beta_end": 1e-2,
            "variance": "fixed_large",
        },
        "objective": {
            "name": "official_imbdiff_cm",
            "image_shape": [3, 4, 4],
            "cfg": False,
            "transfer": {"transfer_x0": False},
        },
        "training": {
            "optimizer": "adam",
            "steps": 1,
            "batch_size": 2,
            "lr": 0.0,
            "warmup_steps": 0,
            "ema_decay": 0.9,
            "early_stopping": {"enabled": False},
        },
        "sampling": {
            "sampler": "ddim",
            "n_samples": 2,
            "sample_batch_size": 2,
            "plot_max_points": 2,
            "ddim_skip": 2,
            "classifier_free_guidance": {"enabled": False},
        },
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=_TinyTarget(),
        source=_TinySource(),
        coupling=IndependentCoupling(),
        path=DiscreteDDPMPath(timesteps=4, beta_end=1e-2),
        model=_tiny_model(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert metrics["sampling"]["implementation"] == "vendored_official_release"
    assert metrics["sampling"]["capacity_branch"] == "on"
    assert (tmp_path / "samples" / "official_ddim.npy").is_file()
