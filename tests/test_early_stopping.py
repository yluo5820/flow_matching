import pytest

from fm_lab.training.trainer import _build_early_stopping


def test_early_stopping_stops_after_patience_without_improvement() -> None:
    stopper = _build_early_stopping(
        {
            "enabled": True,
            "warmup_steps": 0,
            "patience_steps": 4,
            "min_delta": 0.1,
            "ema_alpha": 1.0,
        }
    )

    assert not stopper.update({"step": 1, "loss": 1.0})
    assert not stopper.update({"step": 3, "loss": 0.95})
    assert stopper.update({"step": 5, "loss": 0.96})

    summary = stopper.summary()
    assert summary["stopped"] is True
    assert summary["best_step"] == 1
    assert summary["stop_step"] == 5


def test_early_stopping_warmup_delays_monitoring() -> None:
    stopper = _build_early_stopping(
        {
            "enabled": True,
            "warmup_steps": 10,
            "patience_steps": 4,
            "min_delta": 0.1,
            "ema_alpha": 1.0,
        }
    )

    assert not stopper.update({"step": 5, "loss": 1.0})
    assert stopper.summary()["best_step"] is None
    assert not stopper.update({"step": 10, "loss": 0.9})
    assert stopper.summary()["best_step"] == 10


def test_early_stopping_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="ema_alpha"):
        _build_early_stopping({"enabled": True, "ema_alpha": 0.0})
