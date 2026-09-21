//! Shared fixture: a tiny MLP `Linear -> ReLU -> Linear -> SoftmaxCE`, evaluated
//! two independent ways: with the manual-backprop layers, and with the scalar
//! tape (`scalar_ad`). Everything is packed into one flat parameter vector so
//! finite differences can perturb any entry, including the input.
//!
//! Tests pin f64 explicitly (gradcheck is always f64). If `nalar::Real` ever
//! changes, the conversions in `manual` need updating.
#![allow(dead_code)]

use nalar::layers::{Linear, Relu, SoftmaxCrossEntropy};
use nalar::rng::Pcg32;
use nalar::scalar_ad::{Tape, Var};
use nalar::tensor::Tensor;

pub const B: usize = 4; // batch
pub const IN: usize = 3;
pub const H: usize = 5;
pub const C: usize = 3; // classes
pub const LABELS: [usize; B] = [0, 2, 1, 2];

// Flat layout: x | w1 | b1 | w2 | b2
pub const N_X: usize = B * IN;
pub const N_W1: usize = IN * H;
pub const N_B1: usize = H;
pub const N_W2: usize = H * C;
pub const N_B2: usize = C;
pub const N_TOTAL: usize = N_X + N_W1 + N_B1 + N_W2 + N_B2;

pub fn init_params() -> Vec<f64> {
    let mut rng = Pcg32::new(7, 1);
    (0..N_TOTAL).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

fn split<T>(flat: &[T]) -> (&[T], &[T], &[T], &[T], &[T]) {
    assert_eq!(flat.len(), N_TOTAL);
    let (x, r) = flat.split_at(N_X);
    let (w1, r) = r.split_at(N_W1);
    let (b1, r) = r.split_at(N_B1);
    let (w2, b2) = r.split_at(N_W2);
    (x, w1, b1, w2, b2)
}

fn layer1(w1: &[f64], b1: &[f64]) -> Linear {
    Linear::from_params(Tensor::from_vec(&[IN, H], w1.to_vec()), Tensor::from_vec(&[H], b1.to_vec()))
}

/// Smallest |pre-activation| of the hidden layer. Finite differences are
/// invalid near the ReLU kink, so tests assert this stays well away from 0.
pub fn min_abs_preactivation(flat: &[f64]) -> f64 {
    let (x, w1, b1, _, _) = split(flat);
    let h = layer1(w1, b1).forward(&Tensor::from_vec(&[B, IN], x.to_vec()));
    h.data().iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min)
}

/// Loss and gradient (same flat layout) from the manual-backprop layers.
pub fn manual(flat: &[f64]) -> (f64, Vec<f64>) {
    let (x, w1, b1, w2, b2) = split(flat);
    let x = Tensor::from_vec(&[B, IN], x.to_vec());
    let mut l1 = layer1(w1, b1);
    let mut act = Relu::default();
    let mut l2 = Linear::from_params(
        Tensor::from_vec(&[H, C], w2.to_vec()),
        Tensor::from_vec(&[C], b2.to_vec()),
    );
    let mut ce = SoftmaxCrossEntropy::default();

    let h = l1.forward(&x);
    let a = act.forward(&h);
    let z = l2.forward(&a);
    let loss = ce.forward(&z, &LABELS);

    let dz = ce.backward();
    let da = l2.backward(&dz);
    let dh = act.backward(&da);
    let dx = l1.backward(&dh);

    let mut g = Vec::with_capacity(N_TOTAL);
    g.extend_from_slice(dx.data());
    g.extend_from_slice(l1.dw.data());
    g.extend_from_slice(l1.db.data());
    g.extend_from_slice(l2.dw.data());
    g.extend_from_slice(l2.db.data());
    (loss, g)
}

fn tape_linear(
    t: &mut Tape,
    x: &[Var],
    rows: usize,
    in_dim: usize,
    w: &[Var],
    b: &[Var],
    out_dim: usize,
) -> Vec<Var> {
    let mut y = Vec::with_capacity(rows * out_dim);
    for i in 0..rows {
        for j in 0..out_dim {
            let mut acc = b[j];
            for k in 0..in_dim {
                let p = t.mul(x[i * in_dim + k], w[k * out_dim + j]);
                acc = t.add(acc, p);
            }
            y.push(acc);
        }
    }
    y
}

/// Loss and gradient (same flat layout) from the scalar tape. This is the
/// independent oracle: it shares no backprop code with the layers.
pub fn tape(flat: &[f64]) -> (f64, Vec<f64>) {
    let mut t = Tape::new();
    let vars: Vec<Var> = flat.iter().map(|&v| t.var(v)).collect();
    let (x, w1, b1, w2, b2) = split(&vars);

    let h = tape_linear(&mut t, x, B, IN, w1, b1, H);
    let a: Vec<Var> = h.iter().map(|&v| t.relu(v)).collect();
    let z = tape_linear(&mut t, &a, B, H, w2, b2, C);

    let mut total: Option<Var> = None;
    for i in 0..B {
        let mut s: Option<Var> = None;
        for j in 0..C {
            let e = t.exp(z[i * C + j]);
            s = Some(match s {
                None => e,
                Some(p) => t.add(p, e),
            });
        }
        let lse = t.ln(s.unwrap());
        let li = t.sub(lse, z[i * C + LABELS[i]]);
        total = Some(match total {
            None => li,
            Some(p) => t.add(p, li),
        });
    }
    let nb = t.var(B as f64);
    let loss = t.div(total.unwrap(), nb);

    let g = t.backward(loss);
    (t.value(loss), vars.iter().map(|&v| g.wrt(v)).collect())
}
