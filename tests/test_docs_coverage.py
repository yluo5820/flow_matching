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
