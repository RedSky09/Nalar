//! Optimizers. SGD first; Adam comes later.

use crate::tensor::Tensor;
use crate::Real;

pub struct Sgd {
    pub lr: Real,
}

impl Sgd {
    pub fn new(lr: Real) -> Self {
        Sgd { lr }
    }

    /// `p -= lr * g` for every (parameter, gradient) pair, in the given order.
    pub fn step(&self, params: Vec<(&mut Tensor, &Tensor)>) {
        for (p, g) in params {
            assert_eq!(p.shape(), g.shape());
            for (pv, gv) in p.data_mut().iter_mut().zip(g.data()) {
                *pv -= self.lr * *gv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_step_by_hand() {
        let mut p = Tensor::from_vec(&[2], vec![1.0, -2.0]);
        let g = Tensor::from_vec(&[2], vec![0.5, 4.0]);
        Sgd::new(0.5).step(vec![(&mut p, &g)]);
        assert_eq!(p.data(), &[0.75, -4.0]);
    }
}
