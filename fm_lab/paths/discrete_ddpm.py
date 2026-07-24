"""Marker path for finite-step DDPM objectives and samplers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiscreteDDPMPath:
    """Record that interpolation is owned by a discrete diffusion objective.

    Unlike continuous ``FlowPath`` implementations, a discrete trainer samples
    integer timesteps and constructs ``x_t`` internally. The methods below fail
    closed if a continuous solver accidentally tries to use this marker.
    """

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    name: str = "discrete_ddpm"

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del x0, x1, t, kwargs
        raise RuntimeError(
            "DiscreteDDPMPath is evaluated by its discrete objective, not sample_xt()."
        )

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del x0, x1, t, kwargs
        raise RuntimeError("DiscreteDDPMPath has no continuous target velocity.")

    def metadata(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }
