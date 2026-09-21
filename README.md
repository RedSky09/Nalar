# nalar

An ML engine from scratch in Rust: full training + inference, no dependencies
(`std` only), readable in one afternoon, with reproducible results.

## Claims, and how each one is backed

| Claim                     | Evidence                                                                        | Status                                 |
| ------------------------- | ------------------------------------------------------------------------------- | -------------------------------------- |
| No dependencies           | `[dependencies]` in `Cargo.toml` is empty                                       | satisfied                              |
| Gradients are correct     | `tests/gradcheck.rs`: autograd vs finite differences                            | M1 passes                              |
| Results are reproducible  | `tests/golden_hash.rs`: bit-for-bit hash, compared in CI on Linux/macOS/Windows | RNG only; not yet verified across OSes |
| Readable in one afternoon | code size (target: set a line-count budget here)                                | not yet measured                       |

A claim without evidence must not be stated as fact in this README.

## Milestones

- [x] M0 scaffold, deterministic RNG, CI
- [x] M1 scalar autograd + gradcheck
- [ ] M2 tensor, layers, SGD, MLP, gradchecks, IDX loader, MNIST driver
      (done); accuracy on real MNIST (pending)
- [ ] M3 benchmark (time/epoch, peak memory)
- [ ] M4 bit-for-bit determinism across platforms + per-layer profiler

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
