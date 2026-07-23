from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMUNet,
    OfficialImbDiffObjective,
    OfficialImbDiffUNet,
    load_official_imbdiff_cm_components,
    sample_official_imbdiff,
    sample_official_imbdiff_cm,
)
from fm_lab.paths import DiscreteDDPMPath
from fm_lab.solvers import EulerSolver
from fm_lab.training.losses import build_objective
from fm_lab.training.trainer import train_flow_matching


def _tiny_model(*, num_classes: int = 2) -> OfficialImbDiffCMUNet:
    return OfficialImbDiffCMUNet(
        dim=3 * 4 * 4,
        image_shape=(3, 4, 4),
        timesteps=8,
        base_channels=32,
        channel_multipliers=(1,),
        attention_levels=(),
        num_res_blocks=1,
        dropout=0.0,
        num_classes=num_classes,
        rank_ratio=0.1,
        capacity_parts=("up",),
    )


def _tiny_dropout_model() -> OfficialImbDiffCMUNet:
    return OfficialImbDiffCMUNet(
        dim=3 * 4 * 4,
        image_shape=(3, 4, 4),
        timesteps=8,
        base_channels=32,
        channel_multipliers=(1,),
        attention_levels=(),
        num_res_blocks=1,
        dropout=0.5,
        num_classes=2,
        rank_ratio=0.1,
        capacity_parts=("up",),
    )


def _tiny_standard_model() -> OfficialImbDiffUNet:
    return OfficialImbDiffUNet(
        dim=3 * 4 * 4,
        image_shape=(3, 4, 4),
        timesteps=8,
        base_channels=32,
        channel_multipliers=(1,),
        attention_levels=(),
        num_res_blocks=1,
        dropout=0.0,
        num_classes=2,
    )


def test_official_pure_cm_matches_released_trainer_loss_and_gradients() -> None:
    adapter_model = _tiny_model()
    direct_model = copy.deepcopy(adapter_model)
    counts = (3, 1)
    objective = OfficialImbDiffObjective(
        class_counts=counts,
        method="pure_cm",
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


def test_official_released_cm_matches_transfer_trainer_loss_and_gradients() -> None:
    adapter_model = _tiny_model()
    direct_model = copy.deepcopy(adapter_model)
    counts = (3, 1)
    objective = OfficialImbDiffObjective(
        class_counts=counts,
        method="released_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=True,
        transfer_mode="full",
        consistency_weight=1.0,
        diversity_weight=0.2,
        image_shape=(3, 4, 4),
    )
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
        transfer_x0=True,
        transfer_tr_tau=False,
        transfer_mode="full",
        label_weight_tr=probabilities.unsqueeze(1) @ probabilities.unsqueeze(0),
        w_con=1.0,
        w_div=0.2,
    )

    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape(2, -1)
    labels = torch.tensor([0, 1])
    torch.manual_seed(37)
    objective.capture_next_training_terms()
    adapter_loss, adapter_metrics = objective(
        model=adapter_model,
        path=None,
        x0=torch.zeros_like(clean),
        x1=clean,
        t=torch.zeros(2),
        class_labels=labels,
        original_class_labels=labels,
    )
    captured_terms = objective.pop_captured_training_terms()
    assert captured_terms.loss is adapter_loss
    adapter_loss.backward()

    torch.manual_seed(37)
    direct_loss = direct_trainer(clean.reshape(2, 3, 4, 4), labels)
    direct_loss.backward()

    assert torch.equal(adapter_loss, direct_loss)
    assert adapter_metrics["cm_branch_distance"] >= 0.0
    assert adapter_metrics["cm_unconditional_batch"] == 0.0
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


def test_cm_dropout_pairing_separates_capacity_from_mask_noise() -> None:
    model = _tiny_dropout_model().train()
    objective = OfficialImbDiffObjective(
        class_counts=(3, 1),
        method="pure_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        image_shape=(3, 4, 4),
    )
    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    timesteps = torch.tensor([2, 5])
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(7))

    torch.manual_seed(101)
    independent = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
        dropout_mode="independent",
        capacity_on_enabled=False,
        capacity_off_enabled=False,
    )
    torch.manual_seed(101)
    paired = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
        dropout_mode="paired",
        capacity_on_enabled=False,
        capacity_off_enabled=False,
    )
    disabled = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
        dropout_mode="disabled",
        capacity_on_enabled=False,
        capacity_off_enabled=False,
    )

    assert float(independent.distance_per_sample.detach().mean()) > 0.0
    torch.testing.assert_close(
        paired.distance_per_sample,
        torch.zeros_like(paired.distance_per_sample),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        disabled.distance_per_sample,
        torch.zeros_like(disabled.distance_per_sample),
        rtol=0,
        atol=0,
    )
    assert model.training


def test_cm_paired_dropout_preserves_nonzero_expert_response() -> None:
    model = _tiny_dropout_model().train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.01)
    objective = OfficialImbDiffObjective(
        class_counts=(3, 1),
        method="pure_cm",
        timesteps=8,
        cfg=False,
        transfer_x0=False,
        image_shape=(3, 4, 4),
    )
    clean = torch.randn(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    timesteps = torch.tensor([2, 5])
    noise = torch.randn_like(clean)

    torch.manual_seed(103)
    terms = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
        dropout_mode="paired",
    )

    assert terms.dropout_mode == "paired"
    assert float(terms.distance_per_sample.detach().mean()) > 0.0
    assert terms.loss.grad_fn is not None
    gradients = torch.autograd.grad(
        terms.loss,
        tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
        allow_unused=True,
    )
    assert any(gradient is not None for gradient in gradients)


def test_official_ddpm_matches_released_standard_trainer_loss_and_gradients() -> None:
    adapter_model = _tiny_standard_model()
    direct_model = copy.deepcopy(adapter_model)
    counts = (3, 1)
    objective = OfficialImbDiffObjective(
        class_counts=counts,
        method="ddpm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        image_shape=(3, 4, 4),
    )
    objective._trainer_for(adapter_model, torch.device("cpu"))
    probabilities = torch.tensor(counts, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum()
    components = load_official_imbdiff_cm_components()
    direct_trainer = components.trainer(
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
    )

    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape(2, -1)
    labels = torch.tensor([0, 1])
    torch.manual_seed(19)
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

    torch.manual_seed(19)
    direct_loss, direct_auxiliary = direct_trainer(
        clean.reshape(2, 3, 4, 4), labels
    )
    direct_scalar = direct_loss.mean() + direct_auxiliary
    direct_scalar.backward()

    assert torch.equal(adapter_loss, direct_scalar)
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


def test_released_cm_objective_accepts_channels_last_model_outputs() -> None:
    model = _tiny_model().to(memory_format=torch.channels_last)
    objective = OfficialImbDiffObjective(
        class_counts=(3, 1),
        method="released_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=True,
        image_shape=(3, 4, 4),
    )
    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape(2, -1)
    labels = torch.tensor([0, 1])

    torch.manual_seed(31)
    loss, _ = objective(
        model=model,
        path=None,
        x0=torch.zeros_like(clean),
        x1=clean,
        t=torch.zeros(2),
        class_labels=labels,
        original_class_labels=labels,
    )
    loss.backward()

    assert torch.isfinite(loss)
    with torch.no_grad():
        output = model(
            clean.reshape(2, 3, 4, 4).to(memory_format=torch.channels_last),
            torch.tensor([1, 2]),
            y=labels,
        )
    assert output.is_contiguous()


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


class _RecordingStandardModel(nn.Module):
    is_official_imbdiff_cm = False

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, x, t, y=None):
        del t, y
        self.calls += 1
        return torch.zeros_like(x) + self.anchor * 0.0


def test_release_standard_sampler_handles_non_capacity_unet() -> None:
    model = _RecordingStandardModel()
    samples = sample_official_imbdiff(
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
    assert model.calls > 0


class _LabelBiasModel(nn.Module):
    is_official_imbdiff_cm = False

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.2, -0.3]))

    def forward(self, x, t, context=None, *, y=None, augm=None):
        del t, context, augm
        if y is None:
            return torch.zeros_like(x) + self.bias.mean()
        return torch.zeros_like(x) + self.bias[y].reshape(-1, 1, 1, 1)


def test_cbdm_uses_paper_stop_gradient_directions() -> None:
    clean = torch.zeros(4, 3 * 4 * 4)
    labels = torch.zeros(4, dtype=torch.long)

    without_commitment = _LabelBiasModel()
    objective = OfficialImbDiffObjective(
        class_counts=(1, 1_000_000),
        method="cbdm",
        timesteps=8,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        cbdm_tau=0.001,
        cbdm_gamma=0.0,
        image_shape=(3, 4, 4),
    )
    torch.manual_seed(23)
    loss, metrics = objective(
        model=without_commitment,
        path=None,
        x0=torch.zeros_like(clean),
        x1=clean,
        t=torch.zeros(4),
        class_labels=labels,
        original_class_labels=labels,
    )
    loss.backward()

    assert metrics["cbdm_regularizer"] > 0
    assert metrics["cbdm_commitment"] == 0
    assert without_commitment.bias.grad is not None
    assert without_commitment.bias.grad[0] != 0
    assert without_commitment.bias.grad[1] == 0

    with_commitment = _LabelBiasModel()
    committed_objective = OfficialImbDiffObjective(
        class_counts=(1, 1_000_000),
        method="cbdm",
        timesteps=8,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        cbdm_tau=0.001,
        cbdm_gamma=0.25,
        image_shape=(3, 4, 4),
    )
    torch.manual_seed(23)
    committed_loss, committed_metrics = committed_objective(
        model=with_commitment,
        path=None,
        x0=torch.zeros_like(clean),
        x1=clean,
        t=torch.zeros(4),
        class_labels=labels,
        original_class_labels=labels,
    )
    committed_loss.backward()

    assert committed_metrics["cbdm_commitment"] > 0
    assert with_commitment.bias.grad is not None
    assert with_commitment.bias.grad[1] != 0


@pytest.mark.parametrize(
    ("method", "transfer_x0", "capacity"),
    [
        ("ddpm", False, False),
        ("cbdm", False, False),
        ("oc", True, False),
        ("released_cm", True, True),
        ("pure_cm", False, True),
        ("oc_capacity_only", True, True),
    ],
)
def test_all_matrix_methods_compute_finite_gradients(
    method: str,
    transfer_x0: bool,
    capacity: bool,
) -> None:
    model = _tiny_model() if capacity else _tiny_standard_model()
    objective = OfficialImbDiffObjective(
        class_counts=(3, 1),
        method=method,
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=transfer_x0,
        image_shape=(3, 4, 4),
    )
    clean = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape(2, -1)
    labels = torch.tensor([0, 1])
    torch.manual_seed(29)
    loss, metrics = objective(
        model=model,
        path=None,
        x0=torch.zeros_like(clean),
        x1=clean,
        t=torch.zeros(2),
        class_labels=labels,
        original_class_labels=labels,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(float(loss.detach()))
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("method", "transfer_x0"),
    [
        ("ddpm", False),
        ("cbdm", False),
        ("oc", True),
        ("released_cm", True),
        ("pure_cm", False),
        ("oc_capacity_only", True),
    ],
)
def test_checkpoint_serialized_objective_names_rebuild(
    method: str,
    transfer_x0: bool,
) -> None:
    objective = build_objective(
        {
            "name": f"official_imbdiff_{method}",
            "method": method,
            "image_shape": [3, 4, 4],
            "transfer": {"transfer_x0": transfer_x0},
        },
        diffusion_config={"timesteps": 8, "beta_end": 1e-2},
        class_counts=(3, 1),
    )

    assert isinstance(objective, OfficialImbDiffObjective)
    assert objective.method == method


def test_checkpoint_serialized_objective_rejects_method_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        build_objective(
            {
                "name": "official_imbdiff_released_cm",
                "method": "pure_cm",
            },
            class_counts=(3, 1),
        )


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


class _TinyTarget3(_TinyTarget):
    class_counts = (2, 2, 2)

    def sample_with_labels(self, n: int, device=None):
        images = torch.rand(n, self.dim, device=device) * 2.0 - 1.0
        labels = torch.arange(n, device=device) % 3
        return images, labels

    def all_samples_with_labels(self, device=None):
        images = torch.linspace(-1.0, 1.0, 6 * self.dim).reshape(6, self.dim)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        return images, labels, torch.arange(6).numpy().astype(str)


@pytest.mark.parametrize(
    ("method", "model_name", "transfer_x0", "capacity_branch"),
    [
        ("ddpm", "official_imbdiff_unet", False, "not_applicable"),
        ("pure_cm", "official_imbdiff_cm_unet", False, "on"),
    ],
)
def test_official_path_trains_and_samples_through_fm_lab(
    tmp_path,
    method: str,
    model_name: str,
    transfer_x0: bool,
    capacity_branch: str,
) -> None:
    config = {
        "experiment": {"seed": 3},
        "source": {"name": "gaussian", "dim": 3 * 4 * 4},
        "coupling": {"name": "independent"},
        "path": {"name": "discrete_ddpm"},
        "model": {"name": model_name},
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
            "name": "official_imbdiff",
            "method": method,
            "image_shape": [3, 4, 4],
            "cfg": False,
            "transfer": {"transfer_x0": transfer_x0},
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
        model=_tiny_model() if capacity_branch == "on" else _tiny_standard_model(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert metrics["sampling"]["implementation"] == "vendored_official_release"
    assert metrics["sampling"]["capacity_branch"] == capacity_branch
    assert (tmp_path / "samples" / "official_ddim.npy").is_file()


def test_official_cm_training_loop_records_live_dynamics(tmp_path) -> None:
    config = {
        "experiment": {"seed": 5},
        "source": {"name": "gaussian", "dim": 3 * 4 * 4},
        "coupling": {"name": "independent"},
        "path": {"name": "discrete_ddpm"},
        "model": {"name": "official_imbdiff_cm_unet"},
        "conditioning": {
            "enabled": True,
            "num_classes": 3,
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
            "name": "official_imbdiff",
            "method": "pure_cm",
            "image_shape": [3, 4, 4],
            "cfg": False,
            "transfer": {"transfer_x0": False},
        },
        "training": {
            "optimizer": "adam",
            "steps": 1,
            "batch_size": 3,
            "lr": 1e-4,
            "warmup_steps": 0,
            "ema_decay": 0.9,
            "compile": {"enabled": False},
            "early_stopping": {"enabled": False},
            "cm_dynamics": {
                "enabled": True,
                "steps": [1],
                "max_layers": 1,
            },
        },
        "sampling": {
            "sampler": "ddim",
            "n_samples": 3,
            "sample_batch_size": 3,
            "plot_max_points": 3,
            "ddim_skip": 2,
            "classifier_free_guidance": {"enabled": False},
        },
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=_TinyTarget3(),
        source=_TinySource(),
        coupling=IndependentCoupling(),
        path=DiscreteDDPMPath(timesteps=4, beta_end=1e-2),
        model=_tiny_model(num_classes=3),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert metrics["cm_dynamics"]["observed_steps"] == [1]
    assert (tmp_path / "cm_dynamics" / "gradient_components.csv").is_file()
    assert (tmp_path / "cm_dynamics" / "layer_updates.csv").is_file()
    assert (tmp_path / "cm_dynamics" / "functional_updates.csv").is_file()
