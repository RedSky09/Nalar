//! N-dimensional tensor with shape and strides.
//!
//! Precision is chosen in one place: `crate::Real` (see lib.rs). Everything in
//! the engine core (tensor, layers, optimizer) uses it.
//!
//! Current limitation: storage is always contiguous and row-major, so strides
//! are fully determined by the shape. They are stored explicitly because they
//! are the basis for indexing today and for non-contiguous views later.
//!
//! Determinism policy for reductions: every sum runs in ascending index order,
//! sequentially, on a single thread. Rust does not fuse `a * b + c` into an FMA
//! on its own (as far as I know; verify for the compiler version you use).

use crate::Real;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<Real>,
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_vec(shape, vec![0.0; n])
    }

    pub fn from_vec(shape: &[usize], data: Vec<Real>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "shape {shape:?} needs {n} elements, got {}", data.len());
        Tensor { shape: shape.to_vec(), strides: row_major_strides(shape), data }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn data(&self) -> &[Real] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [Real] {
        &mut self.data
    }
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat offset of a multi-index, computed from the strides.
    pub fn offset(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len(), "index rank mismatch");
        idx.iter()
            .zip(&self.shape)
            .zip(&self.strides)
            .map(|((&i, &dim), &st)| {
                assert!(i < dim, "index {i} out of bounds for dimension of size {dim}");
                i * st
            })
            .sum()
    }

    pub fn get(&self, idx: &[usize]) -> Real {
        self.data[self.offset(idx)]
    }

    pub fn set(&mut self, idx: &[usize], v: Real) {
        let o = self.offset(idx);
        self.data[o] = v;
    }

    /// Returns a copy with a new shape (same element order).
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        Tensor::from_vec(shape, self.data.clone())
    }

    /// (rows, cols) of a 2-D tensor; panics otherwise.
    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(self.shape.len(), 2, "expected a 2-D tensor, got shape {:?}", self.shape);
        (self.shape[0], self.shape[1])
    }

    /// Transposed copy of a 2-D tensor.
    pub fn transpose(&self) -> Tensor {
        let (m, n) = self.dims2();
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::from_vec(&[n, m], out)
    }

    /// `self · b` for `[m,k] · [k,n]`. Each output element is a sequential sum
    /// over k in ascending order, starting from 0.0.
    pub fn matmul(&self, b: &Tensor) -> Tensor {
        let (m, k) = self.dims2();
        let (k2, n) = b.dims2();
        assert_eq!(k, k2, "matmul inner dimensions differ: {k} vs {k2}");
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a * b.data[p * n + j];
                }
            }
        }
        Tensor::from_vec(&[m, n], c)
    }

    /// `selfᵀ · b` for `[k,m]ᵀ · [k,n]`. Bit-identical to
    /// `self.transpose().matmul(b)` because the summation order is the same.
    pub fn matmul_tn(&self, b: &Tensor) -> Tensor {
        let (k, m) = self.dims2();
        let (k2, n) = b.dims2();
        assert_eq!(k, k2, "matmul_tn shared dimensions differ: {k} vs {k2}");
        let mut c = vec![0.0; m * n];
        for p in 0..k {
            for i in 0..m {
                let a = self.data[p * m + i];
                for j in 0..n {
                    c[i * n + j] += a * b.data[p * n + j];
                }
            }
        }
        Tensor::from_vec(&[m, n], c)
    }

    /// `self · bᵀ` for `[m,k] · [n,k]ᵀ`. Bit-identical to
    /// `self.matmul(&b.transpose())`.
    pub fn matmul_nt(&self, b: &Tensor) -> Tensor {
        let (m, k) = self.dims2();
        let (n, k2) = b.dims2();
        assert_eq!(k, k2, "matmul_nt shared dimensions differ: {k} vs {k2}");
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += self.data[i * k + p] * b.data[j * k + p];
                }
                c[i * n + j] = acc;
            }
        }
        Tensor::from_vec(&[m, n], c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], v: &[Real]) -> Tensor {
        Tensor::from_vec(shape, v.to_vec())
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(Tensor::zeros(&[2, 3, 4]).strides(), &[12, 4, 1]);
        assert_eq!(Tensor::zeros(&[5]).strides(), &[1]);
        assert_eq!(Tensor::zeros(&[]).strides(), &[] as &[usize]);
    }

    #[test]
    fn indexing_uses_strides() {
        let x = Tensor::from_vec(&[2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(x.get(&[0, 2]), 2.0);
        assert_eq!(x.get(&[1, 0]), 3.0);
        let mut y = x.clone();
        y.set(&[1, 2], 9.0);
        assert_eq!(y.data()[5], 9.0);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_index_panics() {
        Tensor::zeros(&[2, 2]).get(&[0, 2]);
    }

    #[test]
    fn reshape_keeps_order() {
        let x = t(&[2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape(&[3, 2]);
        assert_eq!(x.get(&[2, 1]), 5.0);
    }

    #[test]
    fn matmul_by_hand() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = t(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(a.matmul(&b).data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_rectangular() {
        // [1,3] · [3,2]
        let a = t(&[1, 3], &[1.0, 2.0, 3.0]);
        let b = t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(a.matmul(&b).data(), &[4.0, 5.0]);
    }

    #[test]
    fn transposed_variants_are_bit_identical() {
        // Non-trivial values so that summation order would show up in the bits.
        let a: Vec<Real> = (0..12).map(|i| ((i * 7 + 3) % 11) as Real * 0.1 - 0.37).collect();
        let b: Vec<Real> = (0..12).map(|i| ((i * 5 + 1) % 13) as Real * 0.07 - 0.21).collect();
        let a = Tensor::from_vec(&[3, 4], a);
        let b = Tensor::from_vec(&[4, 3], b);
        // aᵀ·b'  with a:[3,4] -> tn needs shared first dim: use a:[3,4], c:[3,2]
        let c = Tensor::from_vec(&[3, 2], (0..6).map(|i| i as Real * 0.31 - 0.5).collect());
        assert_eq!(a.matmul_tn(&c), a.transpose().matmul(&c));
        // a·bᵀ with b:[2,4]
        let d = Tensor::from_vec(&[2, 4], (0..8).map(|i| i as Real * 0.17 - 0.4).collect());
        assert_eq!(a.matmul_nt(&d), a.matmul(&d.transpose()));
        let _ = b;
    }
}
