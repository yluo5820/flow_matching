# Released-CM live training dynamics: 2k smoke

**Run:** `released_cm_smoke2k`  
**Training:** CIFAR-100-LT IR100, released CM, batch 64, 2,000 steps  
**Purpose:** validate the observer and identify gross early-training behavior,
not establish the final CM mechanism.

## Validation

- Completed all 13 requested observations at steps
  `1,2,4,8,16,32,64,128,256,512,1000,1500,2000`.
- Wall time was 4m06s on the CUDA server.
- All recorded values were finite.
- The ordinary instrumented Adam step exactly matched an uninstrumented
  controlled step in the integration test.
- The effective expert update identity
  \(\Delta(BA)=B\Delta A+\Delta BA+\Delta B\Delta A\) passed at every observed
  layer/step. The largest numerical relative residual was
  \(4.02\times10^{-5}\), occurring when subtracting very small updates.
- The run wrote 2,239 measurement rows across the eight CSV tables.

Step 1 produced no raw parameter update because the faithful configuration's
5,000-step learning-rate warmup begins at zero. The zero-initialized \(B\)
factor also caused the expected LoRA asymmetry: \(\nabla_A=0\), while
\(\nabla_B\ne0\). The first realized \(BA\) update occurred at step 2.

## Early observations

### Expert gradients become nonzero but remain smaller globally

| Step | Expert gradient energy fraction | Expert/general per-parameter gradient RMS |
| ---: | ---: | ---: |
| 1 | \(1.65\times10^{-11}\) | \(1.37\times10^{-5}\) |
| 128 | 0.000124 | 0.0377 |
| 256 | 0.00157 | 0.134 |
| 512 | 0.00588 | 0.260 |
| 1,000 | 0.00472 | 0.233 |
| 2,000 | 0.00259 | 0.172 |

The energy fraction is affected by the much larger number of general
parameters, so the per-parameter RMS comparison is the more useful companion.
Neither quantity alone identifies stored knowledge.

### Effective expert kernels move, but their immediate functional contribution is small

Across the five representative adapted layers, the median
\(\lVert\Delta BA\rVert/\lVert\Delta W_g\rVert\) was approximately 0.14 at step
2 and remained around 0.18--0.20 from steps 128 through 2,000. Meanwhile, on
the same live noisy batch the expert-effect share of the full functional
update was:

| Step | \(\mathrm{RMS}(\Delta f_e)/\mathrm{RMS}(\Delta f_{\mathrm{full}})\) |
| ---: | ---: |
| 2 | 0.0056 |
| 128 | 0.0483 |
| 256 | 0.0886 |
| 512 | 0.0428 |
| 1,000 | 0.0316 |
| 1,500 | 0.0381 |
| 2,000 | 0.0396 |

Thus the effective expert kernels are not frozen, but early functional motion
is dominated by the general/shared network. This does **not** yet establish
that the final model stores little tail knowledge in the expert: it measures
single-step local effects during only the first 2,000 steps.

### Adam materially transforms the raw effective-expert direction

The median cosine between raw effective-weight gradient descent and the
realized Adam \(BA\) update fell from 0.93 at step 2 to about 0.37--0.49 from
steps 128 through 2,000. The median stable rank of selected \(BA\) kernels grew
from about 2 at step 2 to 14.1 at step 2,000. Direct \(A/B\) gradients,
effective \(BA\) updates, and functional changes therefore provide
non-equivalent views of the learning dynamics.

### The two auxiliary terms often oppose one another

On selected adapted parameters, the consistency/diversity gradient cosine
became strongly negative. Examples:

| Step | General cosine | Expert cosine |
| ---: | ---: | ---: |
| 128 | -0.716 | -0.965 |
| 512 | -0.833 | -0.896 |
| 1,000 | -0.739 | -0.489 |
| 2,000 | -0.816 | -0.661 |

This verifies that the two terms exert competing local pressures, but does not
by itself show that the diversity term routes tail knowledge into the expert.

## Why this run cannot answer the tail-routing question

The observed frequency mix fluctuated sharply. Some batches had only one Few
example and step 512 had none. Consequently, a Few/Many gradient ratio from a
single batch is unstable: the exposure-weighted expert contribution can be
small while its group-mean, per-example counterpart is large.

The promoted 30k run therefore reports both quantities and observes nine early
logarithmic steps plus every 500 steps through 30,000 (69 observations). The
scientific decision should be based on temporal aggregation and direction
coherence, not any isolated batch.

## Provisional conclusion

The observer is valid and reveals genuine structure that endpoint FID cannot:
the LoRA initialization phase, optimizer reorientation, competing auxiliary
gradients, and a gap between kernel-space and function-space update magnitude.
The smoke suggests shared-network motion dominates early training. It does not
yet decide whether CM later develops coherent tail-directed expert updates,
whether the expert stores particular spatial frequencies, or whether its
benefit is mainly generic regularization. Those are the registered questions
for the 30k run.
