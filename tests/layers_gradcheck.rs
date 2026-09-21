//! Per-layer gradient checks: manual backprop vs finite differences (f64).

use nalar::gradcheck::{max_error, numeric_grad};
use nalar::layers::{Linear, Relu, SoftmaxCrossEntropy};
use nalar::mlp::Mlp;
use nalar::rng::Pcg32;
use nalar::tensor::Tensor;
use nalar::Real;

const H: Real = 1e-6;
const TOL: Real = 1e-6;

fn rand_tensor(shape: &[usize], rng: &mut Pcg32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(shape, (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect())
}

fn dot(a: &Tensor, b: &Tensor) -> Real {
    let mut s = 0.0;
    for (x, y) in a.data().iter().zip(b.data()) {
        s += x * y;
    }
    s
}

fn assert_close(name: &str, analytic: &[Real], numeric: &[Real]) {
    let err = max_error(analytic, numeric);
    assert!(err < TOL, "{name}: error {err:e}\n analytic={analytic:?}\n numeric ={numeric:?}");
}

#[test]
fn softmax_cross_entropy_matches_numeric() {
    let mut rng = Pcg32::new(1, 1);
    let (n, c) = (4, 5);
    let logits = rand_tensor(&[n, c], &mut rng);
    let labels: Vec<usize> = (0..n).map(|_| rng.below(c as u32) as usize).collect();

    let mut ce = SoftmaxCrossEntropy::default();
    ce.forward(&logits, &labels);
    let analytic = ce.backward();

    let f = |v: &[Real]| {
        SoftmaxCrossEntropy::default().forward(&Tensor::from_vec(&[n, c], v.to_vec()), &labels)
    };
    assert_close("softmax_ce", analytic.data(), &numeric_grad(&f, logits.data(), H));
}

#[test]
fn linear_matches_numeric() {
    let mut rng = Pcg32::new(2, 1);
    let (batch, n_in, n_out) = (5, 3, 4);
    let x = rand_tensor(&[batch, n_in], &mut rng);
    let w = rand_tensor(&[n_in, n_out], &mut rng);
    let b = rand_tensor(&[n_out], &mut rng);
    let r = rand_tensor(&[batch, n_out], &mut rng); // loss = sum(y * r), so dy = r

    let mut layer = Linear::from_params(w.clone(), b.clone());
    layer.forward(&x);
    let dx = layer.backward(&r);

    let loss = |x: &Tensor, w: &Tensor, b: &Tensor| {
        let mut l = Linear::from_params(w.clone(), b.clone());
        dot(&l.forward(x), &r)
    };
    let nx = numeric_grad(
        &|v: &[Real]| loss(&Tensor::from_vec(x.shape(), v.to_vec()), &w, &b),
        x.data(),
        H,
    );
    let nw = numeric_grad(
        &|v: &[Real]| loss(&x, &Tensor::from_vec(w.shape(), v.to_vec()), &b),
        w.data(),
        H,
    );
    let nb = numeric_grad(
        &|v: &[Real]| loss(&x, &w, &Tensor::from_vec(b.shape(), v.to_vec())),
        b.data(),
        H,
    );
    assert_close("linear dx", dx.data(), &nx);
    assert_close("linear dw", layer.dw.data(), &nw);
    assert_close("linear db", layer.db.data(), &nb);
}

#[test]
fn relu_matches_numeric_away_from_kink() {
    let mut rng = Pcg32::new(3, 1);
    let mut x = rand_tensor(&[3, 6], &mut rng);
    // Finite differences are invalid at the kink, so keep every value at least 0.05 from 0.
    for v in x.data_mut() {
        if v.abs() < 0.05 {
            *v += 0.1;
        }
    }
    let r = rand_tensor(&[3, 6], &mut rng);

    let mut relu = Relu::default();
    relu.forward(&x);
    let dx = relu.backward(&r);

    let f = |v: &[Real]| dot(&Relu::default().forward(&Tensor::from_vec(x.shape(), v.to_vec())), &r);
    assert_close("relu dx", dx.data(), &numeric_grad(&f, x.data(), H));
}

#[test]
fn mlp_matches_numeric_for_params_and_input() {
    let sizes = [3, 5, 4, 2];
    let batch = 6;
    let mut rng = Pcg32::new(4, 1);
    let mut p = Mlp::new(&sizes, &mut rng).flat_params();
    for v in p.iter_mut() {
        *v += rng.uniform(-0.2, 0.2); // makes the (zero-initialised) biases non-trivial
    }
    let x = rand_tensor(&[batch, sizes[0]], &mut rng);
    let labels: Vec<usize> = (0..batch).map(|_| rng.below(2) as usize).collect();

    let mut mlp = Mlp::from_flat_params(&sizes, &p);
    let logits = mlp.forward(&x);
    let mut ce = SoftmaxCrossEntropy::default();
    ce.forward(&logits, &labels);
    let dx = mlp.backward(&ce.backward());

    let loss = |p: &[Real], x: &Tensor| {
        let mut m = Mlp::from_flat_params(&sizes, p);
        let lg = m.forward(x);
        SoftmaxCrossEntropy::default().forward(&lg, &labels)
    };
    let np = numeric_grad(&|v: &[Real]| loss(v, &x), &p, H);
    let nx = numeric_grad(
        &|v: &[Real]| loss(&p, &Tensor::from_vec(x.shape(), v.to_vec())),
        x.data(),
        H,
    );
    assert_close("mlp params", &mlp.flat_grads(), &np);
    assert_close("mlp input", dx.data(), &nx);
}
