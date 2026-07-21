# Official discrete ImbDiff comparison matrix

This matrix is a reduced-budget, controlled CIFAR-100-LT IR100 comparison. It
uses the standard and capacity U-Nets vendored from ImbDiff-CM, the release's
discrete VP/DDPM schedule and DDIM samplers, and a shared 60,000-update protocol.
It is not a numerical reproduction of the release's 300,001-update result.

## Methods

| Method | Endpoint transfer | CM-capable U-Net | CM distance loss | Interpretation |
| --- | --- | --- | --- | --- |
| `ddpm` | no | no | no | ordinary conditional DDPM |
| `cbdm` | no | no | no | DDPM plus the CBDM distribution-adjustment loss |
| `oc` | yes | no | no | released endpoint-transfer method |
| `released_cm` | yes | yes | yes | the authors' released CM recipe |
| `pure_cm` | no | yes | yes | capacity manipulation without endpoint transfer |
| `oc_capacity_only` | yes | yes | no | controls for added capacity without the CM distance loss |

The released CM YAML sets `transfer_x0: true`. Thus `released_cm` is the
composition of endpoint transfer, the CM-capable U-Net, and the CM loss; it is
not the capacity-only intervention. The two additional ablations make the
components identifiable:

- `oc` versus `oc_capacity_only` measures the effect of the expanded model;
- `oc_capacity_only` versus `released_cm` measures the effect of the CM loss;
- `ddpm` versus `pure_cm` measures CM without OC transfer.

The ImbDiff-CM repository releases runnable OC and CM trainers but no CBDM
trainer. The CBDM row therefore implements Eq. (4) of Qin et al., *Class-
Balancing Diffusion Models* (CVPR 2023), directly in discrete epsilon space:
training-distribution auxiliary labels, `tau=0.001`, `gamma=0.25`, and the
paper's stop-gradient directions. It uses the same released standard U-Net,
DDPM schedule, data order, optimizer, EMA, and sampler as DDPM and OC.

## Shared 60k protocol

- CIFAR-100-LT exponential IR100 split, seed 0;
- batch size 64, Adam at `2e-4`, 5,000-step official warmup;
- 1,000 diffusion steps with linear beta `1e-4` to `0.02`;
- 60,000 updates, EMA `0.9999`, checkpoints at 20k, 40k, and 60k;
- 10,000 balanced conditional samples using 50-step DDIM;
- CFG `paper_omega=1.5` for the controlled first comparison;
- full precision, channels-last off, and compile off for reproduction fidelity.

Guidance strength is a sampling-time parameter. After training, checkpoints can
be resampled at other guidance values without retraining; do not mix per-method
guidance tuning into the first controlled result.

## Configurations

All configurations are under `configs/cifar100_lt/autodl_matrix60k/`:

- `cifar100_lt_ir100_official_ddpm_60k.yaml`
- `cifar100_lt_ir100_official_cbdm_60k.yaml`
- `cifar100_lt_ir100_official_oc_60k.yaml`
- `cifar100_lt_ir100_official_released_cm_60k.yaml`
- `cifar100_lt_ir100_official_pure_cm_60k.yaml`
- `cifar100_lt_ir100_official_oc_capacity_only_60k.yaml`

Each run writes to a separate directory below
`/root/autodl-tmp/runs/cifar100_lt_ir100_official_matrix60k/`.

## References

- [Capacity Manipulation for Imbalanced Image Generation](https://openreview.net/forum?id=wSGle6ag5I)
- [Authors' ImbDiff-CM release](https://github.com/Feng-Hong/ImbDiff-CM)
- [Class-Balancing Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Qin_Class-Balancing_Diffusion_Models_CVPR_2023_paper.html)
