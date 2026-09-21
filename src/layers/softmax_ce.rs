//! Softmax + cross-entropy, fused (the fused gradient is simply
//! `(softmax - onehot) / batch`), averaged over the batch.
//!
//! Open determinism item: this calls `exp` and `ln` from the platform libm.
//! See docs/design.md, section 3.

use crate::tensor::Tensor;
use crate::Real;

#[derive(Default)]
pub struct SoftmaxCrossEntropy {
    probs: Option<Tensor>,
    labels: Vec<usize>,
}

impl SoftmaxCrossEntropy {
    /// `logits: [batch, classes]`, `labels[i]` in `0..classes`. Returns the mean
    /// loss. Uses the max-subtraction trick, so large logits do not overflow.
    pub fn forward(&mut self, logits: &Tensor, labels: &[usize]) -> Real {
        assert_eq!(logits.shape().len(), 2, "logits must be 2-D");
        let (b, c) = (logits.shape()[0], logits.shape()[1]);
        assert_eq!(labels.len(), b, "one label per batch row");

        let mut probs = Tensor::zeros(&[b, c]);
        let mut total = 0.0;
        for i in 0..b {
            let y = labels[i];
            assert!(y < c, "label {y} out of range for {c} classes");
            let row = &logits.data()[i * c..(i + 1) * c];
            let m = row.iter().cloned().fold(Real::NEG_INFINITY, Real::max);

            let out = &mut probs.data_mut()[i * c..(i + 1) * c];
            let mut s = 0.0;
            for (o, &z) in out.iter_mut().zip(row) {
                let e = (z - m).exp();
                *o = e;
                s += e;
            }
            for o in out.iter_mut() {
                *o /= s;
            }
            total += (m + s.ln()) - row[y];
        }
        self.probs = Some(probs);
        self.labels = labels.to_vec();
        total / b as Real
    }

    /// Gradient of the mean loss w.r.t. the logits, `[batch, classes]`.
    pub fn backward(&self) -> Tensor {
        let probs = self.probs.as_ref().expect("SoftmaxCrossEntropy::backward called before forward");
        let (b, c) = (probs.shape()[0], probs.shape()[1]);
        let mut g = probs.clone();
        for (i, &y) in self.labels.iter().enumerate() {
            g.data_mut()[i * c + y] -= 1.0;
        }
        for v in g.data_mut() {
            *v /= b as Real;
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_logits_give_log_classes() {
        let mut ce = SoftmaxCrossEntropy::default();
        let loss = ce.forward(&Tensor::zeros(&[2, 4]), &[1, 3]);
        assert!((loss - (4.0 as Real).ln()).abs() < 1e-12);
    }

    #[test]
    fn gradient_rows_sum_to_zero() {
        let mut ce = SoftmaxCrossEntropy::default();
        let logits = Tensor::from_vec(&[2, 3], vec![0.3, -1.2, 2.0, 5.0, 4.9, -3.0]);
        ce.forward(&logits, &[2, 0]);
        let g = ce.backward();
        for row in g.data().chunks(3) {
            assert!(row.iter().sum::<Real>().abs() < 1e-12);
        }
    }

    #[test]
    fn large_logits_do_not_overflow() {
        let mut ce = SoftmaxCrossEntropy::default();
        let logits = Tensor::from_vec(&[1, 3], vec![1000.0, 999.0, -1000.0]);
        let loss = ce.forward(&logits, &[0]);
        assert!(loss.is_finite() && loss > 0.0 && loss < 1.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_bad_label() {
        SoftmaxCrossEntropy::default().forward(&Tensor::zeros(&[1, 3]), &[3]);
    }
}
