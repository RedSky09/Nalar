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
- **Open:** `exp`/`ln`/`tanh` from libm may differ across platforms.
  Options: (a) implement them ourselves from `+ - * /`, or (b) narrow the
  claim to "identical on a single platform". Decide before M4. Until then,
  the main path should avoid transcendental functions (e.g. weight init uses
  uniform, not Box-Muller).

Golden tests that reach libm (`mlp_loss_and_grads_golden`,
`trained_weights_golden`) were recorded on Linux only. CI on macOS and Windows
is the first real test of this open item.

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
breaking change.
