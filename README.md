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
| An MLP learns MNIST | `cargo run --release --example mnist` | **96.74% test accuracy** after 5 epochs on real MNIST (one run, seed 1, no tuning); see Results |
| Readable in one afternoon | code size (target: set a line-count budget here) | not yet measured |

A claim without evidence must not be stated as fact in this README.

## Milestones

- [x] M0 scaffold, deterministic RNG, CI
- [x] M1 scalar autograd + gradcheck
- [x] M2 tensor, layers, SGD, MLP, gradchecks, IDX loader, MNIST driver; 96.74% test accuracy on real MNIST (see Results)
- [ ] M3 benchmark (time/epoch, peak memory)
- [x] M4a bit-for-bit determinism across platforms (CI-verified, scope in the table above)
- [x] M4b per-layer profiler: `cargo run --release --example profile [DATA_DIR] [N_BATCHES]`

## Results

MNIST, one run of `cargo run --release --example mnist`: MLP 784-128-10 (ReLU),
SGD with learning rate 0.1, batch size 64, 5 epochs, seed 1, Xavier-uniform
init, f64, single thread. These hyperparameters were fixed in advance and were
not tuned against the test set.

| epoch | train loss | test accuracy |
|---|---|---|
| 1 | 0.3689 | 93.53% |
| 2 | 0.2014 | 94.78% |
| 3 | 0.1535 | 95.81% |
| 4 | 0.1243 | 96.53% |
| 5 | 0.1045 | **96.74%** |

The curve is still rising at epoch 5, so this is a starting point, not a
ceiling.

Machine: Windows, 12th Gen Intel Core i5-12450H (8 cores, 12 threads; the
engine uses one). Time per epoch was 5.5 to 6.1 s. This is a single run, not a
benchmark: proper time and memory measurements are milestone M3.

Data (SHA-256 of the decompressed IDX files used):

    t10k-images-idx3-ubyte   0FA7898D509279E482958E8CE81C8E77DB3F2F8254E26661CEB7762C4D494CE7
    t10k-labels-idx1-ubyte   FF7BCFD416DE33731A308C3F266CC351222C34898ECBEAF847F06E48F7EC33F2
    train-images-idx3-ubyte  BA891046E6505D7AADCBBE25680A0738AD16AEC93BDE7F9B65E87A2FC25776DB
    train-labels-idx1-ubyte  65A50CBBF4E906D70832878AD85CCDA5333A97F0F4C3DD2EF09A8A9EEF7101C5

The first three match the checksums listed for the same files in several
third-party copies of MNIST; I did not compare them with the original
distribution, and I found no third-party listing for the fourth.

The example also prints a `final weights hash`, a bit-level fingerprint of the
trained weights. On the machine above, two consecutive runs printed the same
value:

    final weights hash: 0x30a6ff4e911bc6ea

Read this carefully: it shows the run is repeatable on that machine (Windows,
x86_64). It has NOT been compared on another machine, because CI does not have
the MNIST files. Anyone with the same four data files can check it: it holds
only for this code, this seed, and these hyperparameters, so any change to the
numerics (initialisation, reduction order, `src/math.rs`) legitimately changes
it and the value must be re-recorded.

One run of `cargo run --release --example profile` (default: 50 batches of the
first 3200 training images), same machine as above:

| label | calls | total ms | µs/call |
|---|---|---|---|
| linear0.fwd | 50 | 147.7 | 2953 |
| relu0.fwd | 50 | 4.3 | 85 |
| linear1.fwd | 50 | 3.9 | 77 |
| loss.fwd | 50 | 1.0 | 19 |
| loss.bwd | 50 | 0.2 | 4 |
| linear1.bwd | 50 | 8.3 | 167 |
| relu0.bwd | 50 | 3.0 | 60 |
| linear0.bwd | 50 | 446.5 | 8929 |
| **forward total** | 50 | 166.4 | 3328 |
| **backward total** | 50 | 463.2 | 9264 |

Estimated memory at batch=64 (tensor sizes, not measured RSS): `linear0`
803,840 bytes of parameters + 65,536 bytes of cached activations; `linear1`
10,320 + 5,120. `linear0` accounts for over 95% of both time and memory. That
matches what the shapes predict: `linear0`'s cost scales with
`n_in × n_out × batch` = 784×128×64, over an order of magnitude more than
`linear1`'s 128×10×64, and its backward pass does two matmuls of that size
(`dW` and `dx`) against one matmul in the forward pass, which is consistent
with `linear0.bwd` costing about 3x `linear0.fwd`.

The gap between `forward_total` (166.4 ms) and the sum of its three rows
(155.8 ms) is about 10.6 ms over 50 calls, roughly 0.2 ms/call: likely the
`x.clone()` at the top of `forward_profiled` (401 KB per batch: 64×784 `f64`)
plus the profiler's own bookkeeping. This has not been measured separately, so
treat it as a plausible explanation, not a confirmed one.

This is one run's wall-clock time on one machine, not a benchmark: no
repetitions, no variance, no comparison to another implementation. M3 below
covers that.

## M3: benchmark

    cargo run --release --example bench [N_ITERS] [WARMUP]     # this engine
    python benches/compare_pytorch.py [N_ITERS] [WARMUP]        # PyTorch, for comparison

Both run the same shape (784-128-10, ReLU, batch 64, float64, single thread)
on one fixed synthetic batch, with a warm-up phase discarded before timing, and
report mean/stdev/min/median/max over many repetitions. For a fair comparison,
run both back to back on the same machine. `benches/compare_pytorch.py` needs
`pip install torch --index-url https://download.pytorch.org/whl/cpu` (the
CPU-only build; the default `pip install torch` pulls a large set of CUDA
packages you don't need for this).

**Caveat on the PyTorch script:** I could not install PyTorch in the sandbox I
worked in (see the script's docstring for why), so, unlike every other file in
this repo, it has not actually been run before being committed. The API it
uses is standard, but treat its numbers as unverified until someone runs it.

**Results:** not yet run. This section will hold nalar's and PyTorch's
mean/stdev per step, and the machine they were measured on, once both have
been run back to back on the same machine.

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
