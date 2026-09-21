//! End-to-end training on a small synthetic task (XOR of the coordinate
//! signs): not linearly separable, so it needs the hidden layer to work.

use nalar::hash::hash_f64s;
use nalar::layers::SoftmaxCrossEntropy;
use nalar::mlp::Mlp;
use nalar::optim::Sgd;
use nalar::rng::Pcg32;
use nalar::tensor::Tensor;
use nalar::Real;

/// Points in [-1, 1]^2, kept at least 0.1 away from both axes.
fn xor_data(n: usize, rng: &mut Pcg32) -> (Tensor, Vec<usize>) {
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    while y.len() < n {
        let (a, b) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
        if a.abs() < 0.1 || b.abs() < 0.1 {
            continue;
        }
        x.push(a);
        x.push(b);
        y.push(((a > 0.0) != (b > 0.0)) as usize);
    }
    (Tensor::from_vec(&[n, 2], x), y)
}

/// Full-batch SGD. Returns the model plus the first and last training loss.
fn train(steps: usize, lr: Real) -> (Mlp, Real, Real) {
    let mut rng = Pcg32::new(1, 1);
    let (x, y) = xor_data(200, &mut rng);
    let mut mlp = Mlp::new(&[2, 16, 2], &mut rng);
    let sgd = Sgd::new(lr);
    let (mut first, mut last) = (0.0, 0.0);
    for step in 0..steps {
        let logits = mlp.forward(&x);
        let mut ce = SoftmaxCrossEntropy::default();
        let loss = ce.forward(&logits, &y);
        if step == 0 {
            first = loss;
        }
        last = loss;
        mlp.backward(&ce.backward());
        sgd.step(mlp.params_and_grads());
    }
    (mlp, first, last)
}

#[test]
fn learns_xor_quadrants() {
    let (mut mlp, first, last) = train(1000, 0.5);
    assert!(last < first * 0.2, "loss did not fall enough: {first} -> {last}");

    // Held-out points from a different stream.
    let (tx, ty) = xor_data(500, &mut Pcg32::new(99, 7));
    let pred = mlp.predict(&tx);
    let correct = pred.iter().zip(&ty).filter(|(a, b)| a == b).count();
    let acc = correct as f64 / ty.len() as f64;
    assert!(acc >= 0.95, "held-out accuracy too low: {acc}");
}

#[test]
fn training_is_repeatable_within_one_process() {
    let a = train(50, 0.5).0.flat_params();
    let b = train(50, 0.5).0.flat_params();
    assert_eq!(a, b);
}

/// Golden hash of the weights after 200 training steps. This is the first test
/// that exercises libm (`exp`/`ln` in the loss), so it is the real test of the
/// cross-platform determinism claim. The constant was recorded on Linux only.
/// If CI fails on another OS, that is a finding about libm, not a test bug:
/// see docs/design.md, section 3.
const GOLDEN_TRAINED_WEIGHTS_200_STEPS: u64 = 0xd3cb372abf8cba76;

#[test]
fn trained_weights_golden() {
    let w = train(200, 0.5).0.flat_params();
    assert_eq!(hash_f64s(&w), GOLDEN_TRAINED_WEIGHTS_200_STEPS, "actual hash: {:#018x}", hash_f64s(&w));
}
