//! Multi-layer perceptron built from the layers in `layers/`:
//! Linear -> ReLU -> ... -> Linear (logits). The loss lives outside the model
//! (`SoftmaxCrossEntropy`), so the model can be used for inference alone.

use crate::layers::{Linear, Relu};
use crate::profile::{bytes_f64, Profiler};
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

    /// Same computation as `forward`, timing each layer into `prof` under
    /// labels `"linear{i}.fwd"` / `"relu{i}.fwd"`. Adds `Instant::now()`
    /// overhead per layer per call; for the batch sizes in `examples/mnist.rs`
    /// that overhead was negligible relative to the work, but it has not been
    /// measured at very small batch sizes.
    pub fn forward_profiled(&mut self, x: &Tensor, prof: &mut Profiler) -> Tensor {
        let last = self.linears.len() - 1;
        let mut h = x.clone();
        for i in 0..self.linears.len() {
            let l = &mut self.linears[i];
            h = prof.time(format!("linear{i}.fwd"), || l.forward(&h));
            if i < last {
                let r = &mut self.relus[i];
                h = prof.time(format!("relu{i}.fwd"), || r.forward(&h));
            }
        }
        h
    }

    /// Same computation as `backward`, timing each layer into `prof` under
    /// labels `"linear{i}.bwd"` / `"relu{i}.bwd"`.
    pub fn backward_profiled(&mut self, dlogits: &Tensor, prof: &mut Profiler) -> Tensor {
        let last = self.linears.len() - 1;
        let mut d = dlogits.clone();
        for i in (0..self.linears.len()).rev() {
            if i < last {
                let r = &mut self.relus[i];
                d = prof.time(format!("relu{i}.bwd"), || r.backward(&d));
            }
            let l = &mut self.linears[i];
            d = prof.time(format!("linear{i}.bwd"), || l.backward(&d));
        }
        d
    }

    /// Estimated bytes for parameters (fixed) and forward-pass activations
    /// (depends on `batch`), per layer. See `profile::bytes_f64`: this is a
    /// size estimate from tensor shapes, not measured process memory, and it
    /// does not include the input `x` itself or the loss layer.
    pub fn memory_report(&self, batch: usize) -> String {
        let mut out = String::new();
        out.push_str(&format!("{:<10} {:>14} {:>18}\n", "layer", "params_bytes", "activation_bytes"));
        let (mut p_total, mut a_total) = (0usize, 0usize);
        for (i, l) in self.linears.iter().enumerate() {
            let pbytes = bytes_f64(l.w.len() + l.b.len());
            let abytes = bytes_f64(batch * l.b.len()); // output of this layer, cached for backward
            p_total += pbytes;
            a_total += abytes;
            out.push_str(&format!("linear{i:<3} {:>14} {:>18}\n", pbytes, abytes));
        }
        out.push_str(&format!("{:<10} {:>14} {:>18}\n", "TOTAL", p_total, a_total));
        out
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

    #[test]
    fn profiled_forward_and_backward_match_unprofiled() {
        let sizes = [4, 6, 3];
        let p = Mlp::new(&sizes, &mut Pcg32::new(1, 1)).flat_params();
        let mut a = Mlp::from_flat_params(&sizes, &p);
        let mut b = Mlp::from_flat_params(&sizes, &p);
        let x = Tensor::from_vec(&[5, 4], (0..20).map(|i| i as Real * 0.07 - 0.6).collect());

        let y_plain = a.forward(&x);
        let mut prof = Profiler::new();
        let y_prof = b.forward_profiled(&x, &mut prof);
        assert_eq!(y_plain, y_prof, "profiled forward must be bit-identical to forward");

        let dy = Tensor::from_vec(y_plain.shape(), (0..y_plain.len()).map(|i| i as Real * 0.03 - 0.1).collect());
        let dx_plain = a.backward(&dy);
        let dx_prof = b.backward_profiled(&dy, &mut prof);
        assert_eq!(dx_plain, dx_prof, "profiled backward must be bit-identical to backward");
        assert_eq!(a.flat_grads(), b.flat_grads());

        for i in 0..sizes.len() - 1 {
            assert!(prof.total_ms(&format!("linear{i}.fwd")).is_some());
            assert!(prof.total_ms(&format!("linear{i}.bwd")).is_some());
        }
    }

    #[test]
    fn memory_report_scales_with_batch() {
        let m = Mlp::new(&[10, 20, 5], &mut Pcg32::new(2, 2));
        let r1 = m.memory_report(1);
        let r32 = m.memory_report(32);
        assert!(r1.contains("linear0"));
        // Activation bytes should be exactly 32x larger for linear0 (batch * n_out * 8).
        let bytes_for = |report: &str, layer: &str| -> usize {
            report.lines().find(|l| l.starts_with(layer)).unwrap()
                .split_whitespace().nth(2).unwrap().parse().unwrap()
        };
        assert_eq!(bytes_for(&r32, "linear0"), bytes_for(&r1, "linear0") * 32);
    }
}
