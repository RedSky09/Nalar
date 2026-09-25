# Design decisions

Each decision here is part of the project's claims. Update this document
whenever a decision changes.

## 1. Language: Rust, `std` only
Reasons: `cargo test` and cross-OS CI are cheap, memory bugs cannot masquerade
as numerical bugs, and Rust does not fuse `a*b+c` into FMA automatically (as
far as I know; verify against the compiler version in use).
Consequence: no `rand` crate, so the RNG is written in-house (PCG32).

## 2. Autograd: hybrid
- `scalar_ad.rs`: scalar tape (micrograd style), f64. Serves as the oracle.
- M2 onward: manual per-layer backprop on tensors (simpler and faster).
- Three-way cross-check: manual vs finite differences vs scalar autograd.
Cost: two code paths to maintain. Benefit: an independent reference.

## 3. Determinism policy
- Single thread in the core version.
- Fixed reduction order (sequential summation).
- No `-ffast-math` (not available on stable Rust); FMA only via an explicit
  `mul_add`, and until there is a conscious decision, do not use it.
- **Resolved: own `exp`/`ln`.** The platform libm gives different bits for
  `exp`/`ln` on different operating systems. Measured with
  `examples/libm_probe.rs` on identical inputs (20,000 each): Linux (my sandbox
  and CI `ubuntu-latest`, which agree even though their Rust versions differ),
  macOS, and Windows produced three different hashes for both functions. That
  also made `trained_weights_golden` fail on macOS and Windows. Decision:
  implement them ourselves in `src/math.rs` from `+ - * /` and integer bit
  manipulation only. The loss layer uses them. `scalar_ad` (the oracle) and the
  gradcheck tests still use libm on purpose: they are references, not part of
  the claim.
- **Verified in CI:** our own `exp`/`ln` are bit-identical on Linux, macOS, and
  Windows (same toolchain, `stable`): the probe's "own exp"/"own ln" hashes
  match on all three, and `tests/math_golden.rs`, `tests/training.rs`, and
  `tests/golden_hash.rs` pass on all three. Scope: 20,000 inputs per function
  and a 200-step training run. That is strong evidence, not a proof for every
  input. The runners' CPU architectures are whatever `rustc -vV` prints in the
  CI log; I have not independently checked them. The claim does not extend to
  compilers or flags that enable FMA contraction, nor to f32.

## 4. Gradcheck
Central difference, f64, h = 1e-6. Error = |a-n| / max(1, |a|, |n|).
Avoid kink points (ReLU at 0). A test ensures the checker can actually fail.

## 5. Precision: f64 behind an alias
`pub type Real = f64;` lives in `lib.rs`; tensor, layers, and optimizer use
`Real`. `scalar_ad`, the RNG, and the gradcheck helpers stay f64 on purpose.
Reason: gradchecks are clean (errors around 1e-9) and bit-level hashes are
easier to defend. Revisit f32 at M3 with measured time and memory. If `Real`
changes, `hash::hash_f64s` and the gradcheck helpers need f32 variants, and
golden hashes must be recorded per type.

## 6. Layer conventions
- Manual backprop. `forward` caches what `backward` needs, so layers take
  `&mut self`.
- `backward` returns the gradient w.r.t. the input and OVERWRITES the layer's
  parameter gradients: one backward call corresponds to one batch, and no
  `zero_grad` is needed.
- Xavier-uniform init from the PCG32 stream (uses `sqrt` only), zero bias,
  bias shape `[out]`.
- The loss (`SoftmaxCrossEntropy`) is separate from the model, so `Mlp` can
  be used for inference alone.

## 7. Reduction order (the golden hashes depend on it)
Every sum runs sequentially in ascending index order on one thread, starting
from 0.0: all three matmul variants, the bias addition (after the inner sum),
and the bias gradient (rows in ascending order). `matmul_tn` and `matmul_nt`
are bit-identical to an explicit transpose followed by `matmul`; a test pins
this. Changing any of these orders changes the golden hashes, so treat it as a
breaking change. (A 4x4 register-tiled version of the three `matmul*` methods
was tried and reverted; see section 11. It preserved this exact order -- every
golden hash passed with unchanged constants both when it was introduced and
when it was reverted -- so the tiling episode never put reproducibility at
risk. It was reverted because it measured slower, not because it was wrong.)

## 8. Accuracy of `math::exp` and `math::ln`
Measured on Linux against a 200-bit mpmath reference, over 180,000 inputs for
`exp` (full range, the softmax range [-30, 0], and near zero) and 240,000 for
`ln` (random bit patterns including subnormals, [0.5, 2], very close to 1, and
[1, 10]):

| | max error (ulp, vs true value) | inputs above 1 ulp |
|---|---|---|
| `math::exp` | 0.76 | 0 |
| `math::ln` | 0.84 | 0 |
| glibc `exp` / `ln` (for comparison) | 0.51 / 0.51 | 0 |

This is a sampled measurement, not a proof of a bound. The first version of
`ln` reached 1.95 ulp near x = 1 (rounding of `s = f/(2+f)` entered the
leading term); it was rewritten so the exact `f` is the leading term. Unit
tests pin a table of correctly rounded reference values (computed offline) and
check special values (NaN, infinities, zero, subnormals, overflow and
underflow thresholds).

## 9. Profiler (`src/profile.rs`)
Timing via `Instant`, no dependencies. Deliberately does NOT measure process
memory (RSS): `std` has no portable call for that, and a platform-specific one
would break the "no dependencies, one afternoon" scope. `memory_report`
instead sums tensor byte sizes (parameters + cached forward activations) from
`Tensor::len()`, which is exact for what the engine allocates but excludes
allocator overhead, the input tensor, and the loss layer. `forward_profiled` /
`backward_profiled` are separate methods, checked to be bit-identical to
`forward` / `backward` (see `mlp::tests`), so profiling cannot silently change
a training run's numerics.

## 10. Benchmark methodology (`examples/bench.rs`, `benches/compare_pytorch.py`)
One fixed synthetic batch, generated once outside the timed region and reused
every iteration (only the weights change, as in real training): this isolates
the cost of the operations (matmul, ReLU, softmax+CE, SGD) from data loading
and from any change in the computation as training progresses. A warm-up phase
is discarded before timing. Statistics are population mean/stdev/min/median/max
over the measured iterations, computed with only `std` (Rust) or the standard
library (Python); no external stats package.

The PyTorch comparison necessarily includes PyTorch's own per-step Python and
autograd overhead; it answers "would PyTorch be faster for a model this small
on this machine", not "which matmul kernel is faster". Both scripts use
float64 and one thread to match nalar's current core (see section 5); revisit
if nalar ever gains f32 or multithreading.

## 11. The nalar-vs-PyTorch speed gap
Measured (see README, M3) on one Windows machine (i5-12450H), one training
step of the 784-128-10 MLP:
- Default build, PyTorch with `torch.set_num_threads(1)` only: nalar 6.5x
  slower (11.09 ms vs 1.71 ms).
- `-C target-cpu=native` (enables AVX2/FMA on that machine): nalar drops to
  6.65 ms, narrowing the gap to 3.9x against the same PyTorch number.
- Pinning `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS=1` made
  PyTorch itself ~2x FASTER (0.83 ms), the opposite of what "hypothesis 2"
  predicted: for a matrix this small, PyTorch's BLAS spawning more than one
  thread apparently cost more in overhead than it gained. Against this truer
  single-thread PyTorch number, nalar (native, naive matmul) is 8.0x slower.
- **Tried and reverted: 4x4 register tiling for `Tensor::matmul` /
  `matmul_tn` / `matmul_nt`.** Verified bit-identical to the naive loops (see
  section 7) both when introduced and when reverted -- every golden hash
  passed throughout, unchanged. Measured on the i5-12450H with
  `target-cpu=native`, three clean runs (stdev 3-5% of the mean, so not
  noise): 7.52-7.61 ms, about 14% SLOWER than the naive loops' 6.65 ms. Likely
  explanation, not confirmed: the naive loop's simple, contiguous access
  pattern (`for j in 0..n { c[j] += a * b[j] }`) is easy for LLVM to
  auto-vectorize into wide AVX2/FMA instructions; the tiled version's nested
  loops over small fixed-size accumulator arrays (`acc[4][4]`) with
  runtime-variable bounds at tile edges are more likely to defeat that
  auto-vectorization, so the "optimization" traded away something the
  compiler was already doing well. Reverted to the naive loops, which remain
  the fastest configuration measured (6.65 ms, `target-cpu=native`, 8.0x
  slower than PyTorch's true single-thread number).
- Not attempted: hand-written SIMD (`unsafe`, outside this project's stated
  scope so far), or a real BLAS-style blocking scheme validated by profiling
  the generated assembly rather than guessing. Either could plausibly do
  better than the reverted attempt, but neither has been tried.
