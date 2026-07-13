import torch

from fm_lab.diffusion import DiscreteDiffusion


def test_linear_schedule_matches_requested_endpoints() -> None:
    diffusion = DiscreteDiffusion(timesteps=4, beta_start=1e-4, beta_end=2e-2)

    assert diffusion.betas.dtype == torch.float64
    assert torch.isclose(diffusion.betas[0], torch.tensor(1e-4, dtype=torch.float64))
    assert torch.isclose(diffusion.betas[-1], torch.tensor(2e-2, dtype=torch.float64))
    assert torch.all(diffusion.alpha_bars[1:] < diffusion.alpha_bars[:-1])


def test_q_sample_uses_selected_cumulative_alpha() -> None:
    diffusion = DiscreteDiffusion(timesteps=4, beta_start=0.1, beta_end=0.4)
    x0 = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    noise = torch.tensor([[0.5, 1.0], [-1.0, 0.25]])
    t = torch.tensor([0, 3])

    actual = diffusion.q_sample(x0, t, noise=noise)
    alpha_bar = diffusion.alpha_bars[t].to(x0.dtype)[:, None]
    expected = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    assert torch.allclose(actual, expected)


def test_prediction_parameterizations_round_trip() -> None:
    diffusion = DiscreteDiffusion(timesteps=10)
    x0 = torch.randn(3, 5)
    epsilon = torch.randn_like(x0)
    t = torch.tensor([0, 4, 9])
    xt = diffusion.q_sample(x0, t, noise=epsilon)

    recovered_x0 = diffusion.predict_x0_from_epsilon(xt, t, epsilon)
    recovered_epsilon = diffusion.predict_epsilon_from_x0(xt, t, x0)
    velocity = diffusion.velocity_target(x0, epsilon, t)

    assert torch.allclose(recovered_x0, x0, atol=1e-5)
    assert torch.allclose(recovered_epsilon, epsilon, atol=1e-5)
    assert torch.allclose(diffusion.predict_x0_from_velocity(xt, t, velocity), x0, atol=1e-5)


def test_q_posterior_matches_closed_form_coefficients() -> None:
    diffusion = DiscreteDiffusion(timesteps=5, beta_start=0.01, beta_end=0.05)
    x0 = torch.tensor([[2.0]])
    xt = torch.tensor([[-1.0]])
    t = torch.tensor([3])

    mean, variance, _ = diffusion.q_posterior(x0, xt, t)
    beta_t = diffusion.betas[3]
    alpha_t = diffusion.alphas[3]
    alpha_bar_t = diffusion.alpha_bars[3]
    alpha_bar_prev = diffusion.alpha_bars[2]
    expected_mean = (
        beta_t * alpha_bar_prev.sqrt() / (1.0 - alpha_bar_t) * x0
        + (1.0 - alpha_bar_prev) * alpha_t.sqrt() / (1.0 - alpha_bar_t) * xt
    )
    expected_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

    assert torch.allclose(mean, expected_mean.float())
    assert torch.allclose(variance, expected_variance.float().reshape(1, 1))


def test_ddpm_final_step_has_no_added_noise() -> None:
    diffusion = DiscreteDiffusion(timesteps=5, beta_start=0.01, beta_end=0.05)
    xt = torch.randn(2, 3)
    epsilon_prediction = torch.randn_like(xt)
    t = torch.zeros(2, dtype=torch.long)

    first = diffusion.p_sample(
        xt, t, epsilon_prediction, prediction_type="epsilon", noise=torch.randn_like(xt)
    )
    second = diffusion.p_sample(
        xt, t, epsilon_prediction, prediction_type="epsilon", noise=torch.randn_like(xt)
    )

    assert torch.equal(first, second)


def test_ddim_eta_zero_is_deterministic_and_reaches_predicted_x0() -> None:
    diffusion = DiscreteDiffusion(timesteps=10)
    xt = torch.randn(2, 3)
    x0_prediction = torch.randn_like(xt)
    t = torch.tensor([7, 7])

    first = diffusion.ddim_step(
        xt, t, torch.tensor([3, 3]), x0_prediction, prediction_type="x", eta=0.0
    )
    second = diffusion.ddim_step(
        xt, t, torch.tensor([3, 3]), x0_prediction, prediction_type="x", eta=0.0
    )
    final = diffusion.ddim_step(
        xt,
        t,
        torch.tensor([-1, -1]),
        x0_prediction,
        prediction_type="x",
        eta=0.0,
        clip_x0=False,
    )

    assert torch.equal(first, second)
    assert torch.allclose(final, x0_prediction)
