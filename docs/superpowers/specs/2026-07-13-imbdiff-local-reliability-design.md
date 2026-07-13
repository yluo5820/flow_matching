# Local ImbDiff Reliability Design

## Objective

Make the five local CIFAR-10 diffusion experiments produce diagnostically
useful samples and expose undertraining or stale EMA weights before aggregate
loss creates false confidence. Paper-scale configs remain unchanged.

## Failure evidence

The completed 5000-step local DDPM run reached live-model loss `0.0237`, but
normal sampling was white noise. Sampling uses EMA weights, and the local
profile inherited paper-scale decay `0.9999`; after 5000 steps, that EMA still
retained approximately 60.7% of its initial parameters. Paired sampling from
identical noise showed structure from live weights and white noise from EMA.

Fixed-timestep validation also showed live-model MSE `0.770` at timestep 0 but
`0.002` at timestep 999. Aggregate random-timestep loss therefore hid severe
undertraining at the low-noise end of the reverse trajectory. Conditional
generation was not the root cause.

## Local defaults

Update all five files under `configs/imbdiff/local/` to use:

```yaml
training:
  steps: 10000
  ema_decay: 0.999
  early_stopping:
    enabled: true
    warmup_steps: 5000
    patience_steps: 2000
    min_delta: 0.0001
    ema_alpha: 0.3

sampling:
  ddim_skip: 20
  classifier_free_guidance:
    enabled: true
    convention: fm_lab
    scale: 1.0
    paper_omega: 0.0
  live_ema_comparison:
    enabled: true
    n_samples: 100
```

Optimizer warmup remains 500 steps. Normal sampling still produces 1000 EMA
samples. Guidance scale 1.0 means class-conditional prediction without
classifier-free amplification. The earliest early stop is step 7000.

## Timestep-resolved training diagnostics

When training diagnostics are requested, derive per-sample diffusion loss from
the same forward pass used by the objective and report three ranges:

- `diffusion_loss_low_noise` for the first third of discrete timesteps;
- `diffusion_loss_mid_noise` for the middle third;
- `diffusion_loss_high_noise` for the final third.

Log a corresponding `_count` metric for every range. Use zero for the loss only
when its count is zero, so CSV columns remain stable. The scalar optimization
loss remains the mean over all samples and must be numerically unchanged for
epsilon and x-vloss objectives and for DDPM, CBDM, OC, and CM methods. No extra
model forward passes are allowed.

## Paired live-versus-EMA sampling

The normal discrete sampler continues to use EMA when available. When
`sampling.live_ema_comparison.enabled` is true and an EMA model exists, generate
the configured number of diagnostic samples from both live and EMA weights.
Use the same labels and identical initial Gaussian noise for each pair and the
same sampler, DDIM skip, eta, and guidance settings as normal generation.

Save:

- `samples/live_diagnostic.npy`;
- `samples/ema_diagnostic.npy`;
- `plots/live_vs_ema.png`.

Record paths and diagnostic sample count under
`metrics.json -> sampling.live_ema_comparison`. If comparison is enabled while
EMA is disabled, raise a clear configuration error instead of silently
skipping it. Paper-scale configs omit the section and incur no extra sampling.

## Verification

Use test-driven development for per-sample loss preservation, timestep buckets,
paired initial noise, artifact creation, and the missing-EMA error. Config tests
must enforce the new values across exactly five local files. Run focused tests,
Ruff, and the full Conda suite, then run a one-step/two-sample local DDPM smoke
with a two-sample live/EMA comparison and inspect the generated plot and
metrics. Commit and push each verified nontrivial checkpoint.
