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

## 4. Gradcheck
Central difference, f64, h = 1e-6. Error = |a-n| / max(1, |a|, |n|).
Avoid kink points (ReLU at 0). A test ensures the checker can actually fail.
