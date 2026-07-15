# Full-network CM capacity design

## Goal

Test whether the weak continuous Capacity Manipulation (CM) signal comes from
reserving capacity in only four decoder convolutions. Replace the decoder-only
adapter placement with a paper-aligned low-rank decomposition throughout the
Fashion-MNIST image U-Net while keeping the data, flow path, prediction/loss
spaces, CM weights, optimizer, and pilot duration fixed.

## Chosen approach

Use switchable low-rank residuals for every suitable convolution and linear
weight matrix in the image U-Net. Convolution kernels are flattened to
`[out_channels, in_channels * kernel_height * kernel_width]` before applying
the low-rank product. Linear layers use their native two-dimensional weight
shape. Biases, embeddings, and normalization parameters stay shared.

The alternatives considered were adapting all convolutions but not conditioning
linear layers, or retaining decoder-only placement and increasing its rank. The
chosen approach is closest to the CM decomposition `W = W_g + BA` and directly
tests the suspected capacity-coverage mismatch.

## Model structure

The existing capacity sections remain meaningful and gain one additional
`conditioning` section:

- `conditioning`: time MLP and class projection linear weights
- `head`: input residual block convolutions
- `down`: downsampling and encoder residual convolutions
- `middle`: bottleneck residual convolutions
- `up`: decoder residual convolutions
- `tail`: output convolution

The production CM config selects all six sections. Individual sections remain
selectable for future ablations. A capacity-off forward uses only shared base
weights; a capacity-on forward uses `W_g + BA` everywhere selected.

## Initialization and data flow

For each adapter, `A` receives Kaiming initialization and `B` is zero. Adapter
initialization occurs in a forked RNG context so adding capacity does not alter
the baseline model's downstream shared parameters. Consequently:

1. same-seed baseline and capacity models have identical shared parameters;
2. capacity-on and capacity-off outputs are exactly equal at initialization;
3. the ordinary base loss on the capacity-on branch supplies the first adapter
   update, after which CM can allocate the resulting branch difference.

Sampling and checkpoint loading continue to use the capacity-on branch by
default, so no inference interface changes are required.

## Tests

Tests are written before implementation and must demonstrate:

- canonical convolution factor shapes and switching behavior;
- switchable linear initialization and switching behavior;
- full section coverage in the image U-Net;
- exact initial capacity-on/off output parity;
- same-seed shared-parameter parity against a non-capacity model;
- configuration parsing and metadata for all selected sections;
- compatibility with the existing CM objective and sampling path.

The focused tests must first fail because full-network coverage and linear
capacity do not yet exist. After implementation, the complete test suite, Ruff,
and `git diff --check` must pass.

## Pilot and decision criterion

Run the existing 1,000-step Fashion-MNIST IR100 CM pilot with only capacity
coverage changed. Record base loss, CM/base ratio, mean/max and frequency-group
capacity distances, gradient norm, and wall-clock time. Compare it with the
decoder-only pilot. A full 20,000-step experiment is justified only if the
full-network pilot creates a materially stronger tail-specific capacity gap
without unstable total loss or excessive runtime.
