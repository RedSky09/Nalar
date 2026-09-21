//! Trains an MLP on MNIST-format IDX files.
//!
//! Usage: `cargo run --release --example mnist [DATA_DIR]` (default `data`).
//! Expects these four decompressed files in DATA_DIR:
//!   train-images-idx3-ubyte, train-labels-idx1-ubyte,
//!   t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte

use nalar::data::{parse_idx, Dataset};
use nalar::layers::SoftmaxCrossEntropy;
use nalar::mlp::Mlp;
use nalar::optim::Sgd;
use nalar::rng::Pcg32;
use nalar::Real;
use std::fs;
use std::time::Instant;

const SEED: u64 = 1;
const HIDDEN: usize = 128;
const EPOCHS: usize = 5;
const BATCH: usize = 64;
const LR: Real = 0.1;

fn load(dir: &str, images: &str, labels: &str) -> Result<Dataset, String> {
    let read = |name: &str| {
        fs::read(format!("{dir}/{name}")).map_err(|e| format!("cannot read {dir}/{name}: {e}"))
    };
    Dataset::from_idx(&parse_idx(&read(images)?)?, &parse_idx(&read(labels)?)?)
}

fn accuracy(mlp: &mut Mlp, ds: &Dataset) -> f64 {
    let mut correct = 0;
    let all: Vec<usize> = (0..ds.len()).collect();
    for chunk in all.chunks(1000) {
        let (x, y) = ds.batch(chunk);
        correct += mlp.predict(&x).iter().zip(&y).filter(|(a, b)| a == b).count();
    }
    correct as f64 / ds.len() as f64
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data".to_string());
    let train = load(&dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        .unwrap_or_else(|e| panic!("{e}"));
    let test = load(&dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        .unwrap_or_else(|e| panic!("{e}"));
    let n_in = train.images.dims2().1;
    println!("train: {} samples, test: {} samples, {} inputs", train.len(), test.len(), n_in);

    let mut rng = Pcg32::new(SEED, 1);
    let mut mlp = Mlp::new(&[n_in, HIDDEN, 10], &mut rng);
    let sgd = Sgd::new(LR);
    let mut order: Vec<usize> = (0..train.len()).collect();

    for epoch in 1..=EPOCHS {
        let t0 = Instant::now();
        rng.shuffle(&mut order);
        let mut loss_sum = 0.0;
        for chunk in order.chunks(BATCH) {
            let (x, y) = train.batch(chunk);
            let logits = mlp.forward(&x);
            let mut ce = SoftmaxCrossEntropy::default();
            loss_sum += ce.forward(&logits, &y) * chunk.len() as Real;
            mlp.backward(&ce.backward());
            sgd.step(mlp.params_and_grads());
        }
        println!(
            "epoch {epoch}: train loss {:.4}, test acc {:.4}, {:.2}s",
            loss_sum / train.len() as Real,
            accuracy(&mut mlp, &test),
            t0.elapsed().as_secs_f64()
        );
    }
}
