# nalar

An ML engine from scratch in Rust: full training + inference, no dependencies
(`std` only), readable in one afternoon, with reproducible results.

[![ci](https://github.com/RedSky09/Nalar/actions/workflows/ci.yml/badge.svg)](https://github.com/RedSky09/Nalar/actions/workflows/ci.yml)

## Claims, and how each one is backed

| Claim | Evidence | Status |
|---|---|---|
| No dependencies | `[dependencies]` in `Cargo.toml` is empty | satisfied |
| Gradients are correct | `tests/gradcheck.rs`, `tests/layers_gradcheck.rs`, `tests/crosscheck_scalar_ad.rs`: manual backprop vs scalar autograd vs finite differences | passes for Linear, ReLU, softmax+CE, and a full MLP (inputs and parameters) |
| Results are reproducible | golden hashes in `tests/golden_hash.rs`, `tests/math_golden.rs`, `tests/training.rs`, and the libm probe, compared in CI on Linux/macOS/Windows | **Verified in CI** on all three OSes: the RNG stream, our own `exp`/`ln` (`src/math.rs`, 20,000 inputs each), and the weights after 200 training steps. Not covered: other compilers or CPUs (e.g. builds that enable FMA), longer training runs, f32. Platform libm `exp`/`ln` were shown to differ across OSes, which is why we do not use them in the engine |
| An MLP learns MNIST | `cargo run --release --example mnist` | training loop verified on synthetic data only; not yet run on real MNIST |
| Readable in one afternoon | code size (target: set a line-count budget here) | not yet measured |

A claim without evidence must not be stated as fact in this README.

## Milestones

- [x] M0 scaffold, deterministic RNG, CI
- [x] M1 scalar autograd + gradcheck
- [ ] M2 tensor, layers, SGD, MLP, gradchecks, IDX loader, MNIST driver
      (done); accuracy on real MNIST (pending)
- [ ] M3 benchmark (time/epoch, peak memory)
- [x] M4a bit-for-bit determinism across platforms (CI-verified, scope in the table above)
- [ ] M4b per-layer profiler (time and memory per layer)

## Running

    cargo test
    cargo test --release
    cargo run --release --example mnist [DATA_DIR]     # default: data/

MNIST: put the four IDX files, already decompressed with `gunzip`, in `data/`
(git-ignored): `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
`t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`. Download them from a
source you trust and record their SHA-256 sums alongside your results, so runs
are comparable.

License: not chosen yet (MIT or Apache-2.0) before the repo goes public.
