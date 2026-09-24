//! Per-layer time and memory profiling.
//!
//! Usage: `cargo run --release --example profile [DATA_DIR] [N_BATCHES]`
//! (defaults: `data`, 50). Uses the same MLP shape and batch size as
//! `examples/mnist.rs` so the two are comparable. Needs the same MNIST files
//! as `examples/mnist.rs`; see its header for the expected file names.
//!
//! This measures wall-clock time in this process, on this machine, for this
//! run: expect run-to-run noise, and treat relative time between layers as
//! more meaningful than the absolute numbers. It does not measure or control
//! for other load on the machine.

use nalar::data::{parse_idx, Dataset};
use nalar::layers::SoftmaxCrossEntropy;
use nalar::mlp::Mlp;
use nalar::profile::Profiler;
use nalar::rng::Pcg32;
use std::fs;
use std::time::Instant;

const HIDDEN: usize = 128;
const BATCH: usize = 64;

fn load(dir: &str, images: &str, labels: &str) -> Result<Dataset, String> {
    let read = |name: &str| {
        fs::read(format!("{dir}/{name}")).map_err(|e| format!("cannot read {dir}/{name}: {e}"))
    };
    Dataset::from_idx(&parse_idx(&read(images)?)?, &parse_idx(&read(labels)?)?)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "data".to_string());
    let n_batches: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);

    let train = load(&dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        .unwrap_or_else(|e| panic!("{e}"));
    let n_in = train.images.dims2().1;
    let n_batches = n_batches.min(train.len() / BATCH).max(1);

    let mut rng = Pcg32::new(1, 1);
    let mut mlp = Mlp::new(&[n_in, HIDDEN, 10], &mut rng);
    let mut prof = Profiler::new();
    let mut order: Vec<usize> = (0..train.len()).collect();
    rng.shuffle(&mut order);

    println!(
        "profiling {n_batches} batches of {BATCH} ({n_in}-{HIDDEN}-10, f64, single thread, release: {})",
        cfg!(debug_assertions).then(|| "no").unwrap_or("yes")
    );

    for chunk in order.chunks(BATCH).take(n_batches) {
        let (x, y) = train.batch(chunk);

        let t0 = Instant::now();
        let logits = mlp.forward_profiled(&x, &mut prof);
        prof.record("forward_total", t0.elapsed());

        let mut ce = SoftmaxCrossEntropy::default();
        let _loss = prof.time("loss.fwd", || ce.forward(&logits, &y));
        let dloss = prof.time("loss.bwd", || ce.backward());

        let t1 = Instant::now();
        mlp.backward_profiled(&dloss, &mut prof);
        prof.record("backward_total", t1.elapsed());
    }

    println!("\n-- time (sum over {n_batches} batches) --");
    println!("{}", prof.report());
    println!("-- estimated memory at batch={BATCH} (tensor sizes, not measured RSS) --");
    println!("{}", mlp.memory_report(BATCH));
}
