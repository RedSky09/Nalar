"""Same-shape PyTorch benchmark, to compare against `cargo run --release
--example bench` on the SAME machine, back to back.

NOTE ON VERIFICATION: this script was written but NOT executed by the
assistant that wrote it. PyTorch could not be installed in that sandbox (the
default `pip install torch` pulls a large set of CUDA/GPU packages, and the
CPU-only wheel index, download.pytorch.org, was not reachable from there). The
PyTorch API used here (nn.Linear with dtype=torch.float64, nn.CrossEntropyLoss,
optim.SGD) is standard and has been stable for a long time, but treat this
script as unverified until you run it and see sane output; if it errors,
please share the traceback.

Install (CPU only, much smaller than the default CUDA build):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

Usage:
    python compare_pytorch.py [N_ITERS] [WARMUP]
    (defaults: 200, 20 -- same defaults as examples/bench.rs)

Design choices made to match examples/bench.rs as closely as possible:
  - Same shape: 784-128-10, ReLU, batch 64.
  - float64, matching nalar's `Real` (see docs/design.md, section 5).
  - Single thread (torch.set_num_threads(1)), matching nalar's single-threaded
    core.
  - One fixed random batch, generated once outside the timed region and reused
    for every iteration; only the weights change between iterations, exactly
    as in the Rust harness.
  - Warm-up iterations are discarded before timing, same as the Rust harness.

What this does NOT control for, and that a strict comparison should note:
  - PyTorch's per-step Python-interpreter and autograd-graph overhead is
    included in every timing. That is realistic if the question is "would
    PyTorch be faster for a model this small", but it is not a measurement of
    matmul throughput alone.
  - Weight initialisation differs (PyTorch's default `nn.Linear` init vs
    nalar's Xavier-uniform from PCG32). This does not affect step time.
  - CPU frequency scaling, other processes, and thermal state are not
    controlled here or in the Rust harness. Run both back to back, and prefer
    the median over the mean if you see outliers.
"""

import sys
import time

import torch

N_IN, HIDDEN, N_OUT, BATCH = 784, 128, 10, 64
LR = 0.1
SEED = 1


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(N_IN, HIDDEN, dtype=torch.float64)
        self.l2 = torch.nn.Linear(HIDDEN, N_OUT, dtype=torch.float64)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, var**0.5, xs[0], xs[n // 2], xs[-1]


def main():
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if n_iters < 2:
        raise SystemExit("need at least 2 measured iterations for a standard deviation")

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(BATCH, N_IN, dtype=torch.float64)
    y = torch.randint(0, N_OUT, (BATCH,))

    def one_step():
        optimizer.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        optimizer.step()

    print(f"pytorch bench: {N_IN}-{HIDDEN}-{N_OUT}, batch={BATCH}, float64, "
          f"1 thread, one fixed batch, torch {torch.__version__}")
    print(f"warmup={warmup} iters (discarded), measured={n_iters} iters")

    for _ in range(warmup):
        one_step()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        one_step()
        times.append((time.perf_counter() - t0) * 1e3)

    mean, stdev, lo, median, hi = stats(times)
    print(f"per-step ms: mean={mean:.4f} stdev={stdev:.4f} min={lo:.4f} "
          f"median={median:.4f} max={hi:.4f}")
    print(f"(stdev/mean = {100 * stdev / mean:.1f}%; a high ratio means other load "
          f"on the machine, or too few warm-up iterations)")


if __name__ == "__main__":
    main()
