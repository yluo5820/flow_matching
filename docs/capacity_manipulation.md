# Capacity Manipulation Baseline

This repository retains Capacity Manipulation as a literature baseline. The
implemented method is a continuous-flow adaptation of Hong et al., not an exact
reproduction of their released CIFAR-100-LT experiment.

## Correspondence

The retained implementation includes:

- switchable, zero-initialized low-rank capacity;
- the distance between capacity-on and capacity-off predictions;
- class-frequency consistency and diversity weighting;
- the released core defaults of `rank_ratio=0.1`, up-block placement,
  `w_con=1.0`, and `w_div=0.2`.

The local adaptation differs in three material ways:

- it trains a continuous flow-matching objective instead of discrete noise
  prediction;
- it uses conventional flattened-kernel low-rank factors rather than the
  authors' spatially structured convolution factorization;
- its canonical CM-only configuration does not enable the endpoint-transfer
  component used by the authors' released CIFAR-100-LT recipe.

The canonical local configuration is
`configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml`.

Primary references:

- [OpenReview paper](https://openreview.net/forum?id=wSGle6ag5I)
- [Authors' implementation](https://github.com/Feng-Hong/ImbDiff-CM)
- [Released CM configuration](https://github.com/Feng-Hong/ImbDiff-CM/blob/main/configs/cifar100lt_ir100/cm.yaml)
