# nalar

An ML engine from scratch in Rust: full training + inference, no dependencies
(`std` only), readable in one afternoon, with reproducible results.

## Claims, and how each one is backed

| Claim | Evidence | Status |
|---|---|---|
| No dependencies | `[dependencies]` in `Cargo.toml` is empty | satisfied |
| Gradients are correct | `tests/gradcheck.rs`: autograd vs finite differences | M1 passes |
| Results are reproducible | `tests/golden_hash.rs`: bit-for-bit hash, compared in CI on Linux/macOS/Windows | RNG stream verified in CI, float training not yet covered |
| Readable in one afternoon | code size (target: set a line-count budget here) | not yet measured |

A claim without evidence must not be stated as fact in this README.

## Milestones

- [x] M0 scaffold, deterministic RNG, CI
- [x] M1 scalar autograd + gradcheck
- [ ] M2 tensor, layers, MLP on MNIST
- [ ] M3 benchmark (time/epoch, peak memory)
- [ ] M4 bit-for-bit determinism across platforms + per-layer profiler

## Running

    cargo test
    cargo test --release

License: not chosen yet (MIT or Apache-2.0) before the repo goes public.
