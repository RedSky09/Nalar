//! ReLU. The derivative at exactly 0 is defined as 0, matching `scalar_ad`.

use crate::tensor::Tensor;
use crate::Real;

#[derive(Default)]
pub struct Relu {
    cache_x: Option<Tensor>,
}

impl Relu {
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut y = x.clone();
        for v in y.data_mut() {
            *v = v.max(0.0);
        }
        self.cache_x = Some(x.clone());
        y
    }

    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        let x = self.cache_x.as_ref().expect("Relu::backward called before forward");
        assert_eq!(x.shape(), dy.shape(), "gradient shape must match input shape");
        let mut dx = dy.clone();
        for (g, &xv) in dx.data_mut().iter_mut().zip(x.data()) {
            if xv <= 0.0 as Real {
                *g = 0.0;
            }
        }
        dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_backward_by_hand() {
        let mut r = Relu::default();
        let x = Tensor::from_vec(&[1, 4], vec![-2.0, 0.0, 0.5, 3.0]);
        assert_eq!(r.forward(&x).data(), &[0.0, 0.0, 0.5, 3.0]);
        let dy = Tensor::from_vec(&[1, 4], vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(r.backward(&dy).data(), &[0.0, 0.0, 1.0, 1.0]);
    }
}
