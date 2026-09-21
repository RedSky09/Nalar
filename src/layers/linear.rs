//! Fully connected layer: `y = x·W + b`, with `W: [in, out]`, `b: [out]`.

use crate::rng::Pcg32;
use crate::tensor::Tensor;
use crate::Real;

pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
    pub dw: Tensor,
    pub db: Tensor,
    cache_x: Option<Tensor>,
}

impl Linear {
    /// Xavier-uniform weights, zero bias. Uses `sqrt`, which IEEE-754 requires
    /// to be correctly rounded, so init does not depend on the platform libm.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut Pcg32) -> Self {
        let limit = (6.0 / (in_dim + out_dim) as Real).sqrt();
        let data: Vec<Real> = (0..in_dim * out_dim).map(|_| rng.uniform(-limit, limit)).collect();
        Self::from_params(Tensor::from_vec(&[in_dim, out_dim], data), Tensor::zeros(&[out_dim]))
    }

    pub fn from_params(w: Tensor, b: Tensor) -> Self {
        assert_eq!(w.shape().len(), 2, "weight must be 2-D");
        assert_eq!(b.shape(), &[w.shape()[1]], "bias must have one entry per output");
        let dw = Tensor::zeros(w.shape());
        let db = Tensor::zeros(b.shape());
        Linear { w, b, dw, db, cache_x: None }
    }

    /// `x: [batch, in]` -> `[batch, out]`. The bias is added after the matrix
    /// product's inner sum has completed (fixed order, see `tensor.rs`).
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut y = x.matmul(&self.w);
        let out = self.b.len();
        for row in y.data_mut().chunks_mut(out) {
            for (v, bj) in row.iter_mut().zip(self.b.data()) {
                *v += *bj;
            }
        }
        self.cache_x = Some(x.clone());
        y
    }

    /// `dy: [batch, out]` -> `dx: [batch, in]`. Sets `dw = xᵀ·dy` and
    /// `db = sum over batch rows of dy` (rows summed in ascending order).
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        let x = self.cache_x.as_ref().expect("Linear::backward called before forward");
        self.dw = x.matmul_tn(dy);

        let out = self.b.len();
        let mut db = vec![0.0; out];
        for row in dy.data().chunks(out) {
            for (acc, g) in db.iter_mut().zip(row) {
                *acc += *g;
            }
        }
        self.db = Tensor::from_vec(&[out], db);

        dy.matmul_nt(&self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_deterministic_and_bounded() {
        let a = Linear::new(4, 6, &mut Pcg32::new(3, 3));
        let b = Linear::new(4, 6, &mut Pcg32::new(3, 3));
        assert_eq!(a.w, b.w);
        let limit = (6.0_f64 / 10.0).sqrt();
        assert!(a.w.data().iter().all(|v| v.abs() <= limit));
        assert!(a.b.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_by_hand() {
        // x = [[1, 2]], W = [[1, 0, 2], [0, 1, 3]], b = [10, 20, 30]
        let w = Tensor::from_vec(&[2, 3], vec![1.0, 0.0, 2.0, 0.0, 1.0, 3.0]);
        let b = Tensor::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        let mut l = Linear::from_params(w, b);
        let y = l.forward(&Tensor::from_vec(&[1, 2], vec![1.0, 2.0]));
        assert_eq!(y.data(), &[11.0, 22.0, 38.0]);
    }

    #[test]
    #[should_panic(expected = "before forward")]
    fn backward_before_forward_panics() {
        let mut l = Linear::new(2, 2, &mut Pcg32::new(1, 1));
        l.backward(&Tensor::zeros(&[1, 2]));
    }
}
