//! Multi-layer perceptron built from the layers in `layers/`:
//! Linear -> ReLU -> ... -> Linear (logits). The loss lives outside the model
//! (`SoftmaxCrossEntropy`), so the model can be used for inference alone.

use crate::layers::{Linear, Relu};
use crate::rng::Pcg32;
use crate::tensor::Tensor;
use crate::Real;

pub struct Mlp {
    sizes: Vec<usize>,
    linears: Vec<Linear>,
    /// One ReLU per hidden layer, so `relus.len() == linears.len() - 1`.
    relus: Vec<Relu>,
}

impl Mlp {
    /// `sizes = [n_in, hidden1, ..., n_out]`.
    pub fn new(sizes: &[usize], rng: &mut Pcg32) -> Self {
        assert!(sizes.len() >= 2, "need at least an input and an output size");
        let linears = sizes.windows(2).map(|w| Linear::new(w[0], w[1], rng)).collect();
        Self::assemble(sizes, linears)
    }

    /// Builds a model from flat parameters in the layout of `flat_params`.
    pub fn from_flat_params(sizes: &[usize], p: &[Real]) -> Self {
        assert!(sizes.len() >= 2, "need at least an input and an output size");
        let mut off = 0;
        let mut linears = Vec::new();
        for w in sizes.windows(2) {
            let (n_in, n_out) = (w[0], w[1]);
            let wt = Tensor::from_vec(&[n_in, n_out], p[off..off + n_in * n_out].to_vec());
            off += n_in * n_out;
            let bt = Tensor::from_vec(&[n_out], p[off..off + n_out].to_vec());
            off += n_out;
            linears.push(Linear::from_params(wt, bt));
        }
        assert_eq!(off, p.len(), "parameter vector has the wrong length for sizes {sizes:?}");
        Self::assemble(sizes, linears)
    }

    fn assemble(sizes: &[usize], linears: Vec<Linear>) -> Self {
        let relus = (1..linears.len()).map(|_| Relu::default()).collect();
        Mlp { sizes: sizes.to_vec(), linears, relus }
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// `x`: `[batch, n_in]` -> logits `[batch, n_out]`.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let last = self.linears.len() - 1;
        let mut h = x.clone();
        for i in 0..self.linears.len() {
            h = self.linears[i].forward(&h);
            if i < last {
                h = self.relus[i].forward(&h);
            }
        }
        h
    }

    /// Backprop from the gradient w.r.t. the logits; returns the gradient
    /// w.r.t. the input. Parameter gradients are overwritten (see `layers`).
    pub fn backward(&mut self, dlogits: &Tensor) -> Tensor {
        let last = self.linears.len() - 1;
        let mut d = dlogits.clone();
        for i in (0..self.linears.len()).rev() {
            if i < last {
                d = self.relus[i].backward(&d);
            }
            d = self.linears[i].backward(&d);
        }
        d
    }

    /// Argmax over classes for each row (ties resolve to the lowest index).
    pub fn predict(&mut self, x: &Tensor) -> Vec<usize> {
        let logits = self.forward(x);
        let (n, c) = logits.dims2();
        (0..n)
            .map(|i| {
                let row = &logits.data()[i * c..(i + 1) * c];
                let mut best = 0;
                for j in 1..c {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                best
            })
            .collect()
    }

    /// (parameter, gradient) pairs in a fixed order: for each layer, w then b.
    pub fn params_and_grads(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut v = Vec::new();
        for l in self.linears.iter_mut() {
            v.push((&mut l.w, &l.dw));
            v.push((&mut l.b, &l.db));
        }
        v
    }

    pub fn param_count(&self) -> usize {
        self.linears.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    /// All parameters flattened: for each layer, w (row-major `[n_in, n_out]`) then b.
    pub fn flat_params(&self) -> Vec<Real> {
        let mut v = Vec::with_capacity(self.param_count());
        for l in &self.linears {
            v.extend_from_slice(l.w.data());
            v.extend_from_slice(l.b.data());
        }
        v
    }

    /// Gradients from the last backward pass, in the same layout as `flat_params`.
    pub fn flat_grads(&self) -> Vec<Real> {
        let mut v = Vec::with_capacity(self.param_count());
        for l in &self.linears {
            v.extend_from_slice(l.dw.data());
            v.extend_from_slice(l.db.data());
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_count_and_layout_roundtrip() {
        let mut rng = Pcg32::new(1, 1);
        let m = Mlp::new(&[3, 4, 2], &mut rng);
        assert_eq!(m.param_count(), 3 * 4 + 4 + 4 * 2 + 2);
        let p = m.flat_params();
        let m2 = Mlp::from_flat_params(m.sizes(), &p);
        assert_eq!(m2.flat_params(), p);
    }

    #[test]
    fn forward_shape_and_deterministic_init() {
        let a = Mlp::new(&[5, 7, 3], &mut Pcg32::new(9, 9));
        let b = Mlp::new(&[5, 7, 3], &mut Pcg32::new(9, 9));
        assert_eq!(a.flat_params(), b.flat_params());
        let mut a = a;
        let y = a.forward(&Tensor::zeros(&[4, 5]));
        assert_eq!(y.shape(), &[4, 3]);
    }

    #[test]
    fn predict_takes_first_max() {
        // Identity-like 1-layer "model": logits equal the input.
        let w = Tensor::from_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let mut m = Mlp::from_flat_params(&[2, 2], &[w.data(), &[0.0, 0.0]].concat());
        let x = Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 5.0, 4.0, 3.0, 3.0]);
        assert_eq!(m.predict(&x), vec![1, 0, 0]);
    }
}
