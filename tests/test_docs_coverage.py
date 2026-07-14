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
