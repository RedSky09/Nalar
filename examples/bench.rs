//! Formal benchmark: one fixed synthetic batch, repeated training steps
//! (forward + loss + backward + SGD step), with a warm-up phase and basic
//! statistics (mean, standard deviation, min, median, max).
//!
//! This is a same-machine, same-process measurement: it says how this
//! implementation performs here, now. It is NOT a cross-machine number by
//! itself. For a comparison against another implementation (e.g. PyTorch, see
//! benches/compare_pytorch.py), run both on the same machine, back to back,
//! with nothing else competing for the CPU, and report both.
//!
//! Data is synthetic (not MNIST) on purpose: this measures the cost of the
//! operations (matmul, ReLU, softmax+CE, SGD) for a fixed shape, not learning.
//! The one input batch is generated once, outside the timed region, and reused
//! for every iteration; only the weights change between iterations, exactly
//! as in real training.
//!
//! Usage: `cargo run --release --example bench [N_ITERS] [WARMUP]`
//! (defaults: 200, 20)

use nalar::layers::SoftmaxCrossEntropy;
use nalar::mlp::Mlp;
use nalar::optim::Sgd;
use nalar::rng::Pcg32;
use nalar::tensor::Tensor;
use nalar::Real;
use std::time::Instant;

const N_IN: usize = 784;
const HIDDEN: usize = 128;
const N_OUT: usize = 10;
const BATCH: usize = 64;
const LR: Real = 0.1;

fn synthetic_batch(rng: &mut Pcg32) -> (Tensor, Vec<usize>) {
    let x = Tensor::from_vec(&[BATCH, N_IN], (0..BATCH * N_IN).map(|_| rng.uniform(-1.0, 1.0)).collect());
    let y = (0..BATCH).map(|_| rng.below(N_OUT as u32) as usize).collect();
    (x, y)
}

/// Population mean, population standard deviation, min, median, max, in that
/// order. `xs` is sorted in place.
fn stats(xs: &mut [f64]) -> (f64, f64, f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt(), xs[0], xs[xs.len() / 2], *xs.last().unwrap())
}

fn one_step(mlp: &mut Mlp, sgd: &Sgd, x: &Tensor, y: &[usize]) {
    let logits = mlp.forward(x);
    let mut ce = SoftmaxCrossEntropy::default();
    ce.forward(&logits, y);
    mlp.backward(&ce.backward());
    sgd.step(mlp.params_and_grads());
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let warmup: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    assert!(n_iters >= 2, "need at least 2 measured iterations for a standard deviation");

    let mut rng = Pcg32::new(1, 1);
    let mut mlp = Mlp::new(&[N_IN, HIDDEN, N_OUT], &mut rng);
    let sgd = Sgd::new(LR);
    let (x, y) = synthetic_batch(&mut rng);

    println!("nalar bench: {N_IN}-{HIDDEN}-{N_OUT}, batch={BATCH}, f64, single thread, one fixed batch");
    println!("warmup={warmup} iters (discarded), measured={n_iters} iters");

    for _ in 0..warmup {
        one_step(&mut mlp, &sgd, &x, &y);
    }

    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        one_step(&mut mlp, &sgd, &x, &y);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let (mean, stdev, min, median, max) = stats(&mut times);
    println!("per-step ms: mean={mean:.4} stdev={stdev:.4} min={min:.4} median={median:.4} max={max:.4}");
    println!("(stdev/mean = {:.1}%; a high ratio means other load on the machine, or too few warm-up iterations)", 100.0 * stdev / mean);
}
