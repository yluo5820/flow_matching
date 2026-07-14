import tomllib
from pathlib import Path


def test_cli_docs_mention_all_console_scripts() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    docs = Path("docs/cli.md").read_text(encoding="utf-8")

    for name in scripts:
        assert name in docs


def test_diagnostics_docs_cover_core_outputs() -> None:
    docs = Path("docs/diagnostics.md").read_text(encoding="utf-8")
    required_terms = [
        "ambiguity_heatmap",
        "ambiguity_time.csv",
        "field_stats.csv",
        "solver_sensitivity_nfe",
        "geometry_time.csv",
        "summary.csv",
    ]

    for term in required_terms:
        assert term in docs


def test_fashion_mnist_long_tail_docs_define_balanced_reference_protocol() -> None:
    docs = Path("docs/cli.md").read_text(encoding="utf-8")

    assert "fm-lab-fashion-mnist-lt-eval" in docs
    assert "1,000 generated samples per class" in docs
    assert "official Fashion-MNIST test split" in docs


def test_continuous_fashion_mnist_suite_is_documented_in_readme_and_cli() -> None:
    config_stems = [
        "fashion_mnist_lt_ir100_x_vloss",
        "fashion_mnist_lt_ir100_x_vloss_cbdm",
        "fashion_mnist_lt_ir100_x_vloss_oc",
        "fashion_mnist_lt_ir100_x_vloss_cm",
    ]
    generation_methods = ["x_vloss", "x_vloss_cbdm", "x_vloss_oc", "x_vloss_cm"]

    for path in (Path("README.md"), Path("docs/cli.md")):
        docs = path.read_text(encoding="utf-8")
        for stem in config_stems:
            assert f"configs/fashion_mnist_lt/{stem}.yaml" in docs
            method = stem.removeprefix("fashion_mnist_lt_ir100_")
            assert f"runs/fashion_mnist_lt_ir100/{method}" in docs
        for method in generation_methods:
            assert f"--generation-method {method}" in docs
        assert "samples/euler_nfe64.npy" in docs
        assert "samples/generated_labels.npy" in docs
        assert "checkpoint.pt" in docs
        assert "--sampler euler" in docs
        assert "--nfe 64" in docs
        assert "--guidance-scale 2.0" in docs
        assert "--generation-seed 0" in docs
        assert "logit-normal" in docs
        assert "(-0.8, 0.8)" in docs
        assert "0.05" in docs
        assert "Euler/NFE-64" in docs


def test_continuous_fashion_mnist_docs_do_not_recommend_discrete_training() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8") for path in (Path("README.md"), Path("docs/cli.md"))
    )

    assert "fm-lab-train" in docs
    assert "--diffusion-prediction-type" not in docs
    assert "--ddpm" not in docs.lower()
    assert "--ddim" not in docs.lower()
