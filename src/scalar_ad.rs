//! M1: scalar tape-based autograd (micrograd style).
//!
//! Role in this project: REFERENCE ORACLE. The manual per-layer backprop (M2)
//! is cross-checked against this and against finite differences.
//!
//! Design: nodes live in a `Vec` and are referenced by index (`Var`), not by
//! pointer. A node is always created after its operands, so index order is
//! already a topological order and backward is a single reverse loop.
//!
//! Note: `exp`, `ln`, `tanh` call the platform libm. That is fine for an
//! oracle; for the cross-platform determinism claim, the main code path must
//! not depend on them (see docs/design.md).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

#[derive(Clone, Copy, Debug)]
enum Op {
    Leaf,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Neg(usize),
    Exp(usize),
    Ln(usize),
    Tanh(usize),
    Relu(usize),
}

#[derive(Default)]
pub struct Tape {
    values: Vec<f64>,
    ops: Vec<Op>,
}

/// Result of backward: gradient of the output with respect to every node.
pub struct Grads(Vec<f64>);

impl Grads {
    pub fn wrt(&self, x: Var) -> f64 {
        self.0[x.0]
    }
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, v: f64, op: Op) -> Var {
        self.values.push(v);
        self.ops.push(op);
        Var(self.values.len() - 1)
    }

    fn v(&self, x: Var) -> f64 {
        self.values[x.0]
    }

    pub fn var(&mut self, v: f64) -> Var {
        self.push(v, Op::Leaf)
    }

    pub fn value(&self, x: Var) -> f64 {
        self.v(x)
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let v = self.v(a) + self.v(b);
        self.push(v, Op::Add(a.0, b.0))
    }
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        let v = self.v(a) - self.v(b);
        self.push(v, Op::Sub(a.0, b.0))
    }
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let v = self.v(a) * self.v(b);
        self.push(v, Op::Mul(a.0, b.0))
    }
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let v = self.v(a) / self.v(b);
        self.push(v, Op::Div(a.0, b.0))
    }
    pub fn neg(&mut self, a: Var) -> Var {
        let v = -self.v(a);
        self.push(v, Op::Neg(a.0))
    }
    pub fn exp(&mut self, a: Var) -> Var {
        let v = self.v(a).exp();
        self.push(v, Op::Exp(a.0))
    }
    pub fn ln(&mut self, a: Var) -> Var {
        let v = self.v(a).ln();
        self.push(v, Op::Ln(a.0))
    }
    pub fn tanh(&mut self, a: Var) -> Var {
        let v = self.v(a).tanh();
        self.push(v, Op::Tanh(a.0))
    }
    /// The ReLU derivative at 0 is defined as 0 (a common convention, not the only one).
    pub fn relu(&mut self, a: Var) -> Var {
        let v = self.v(a).max(0.0);
        self.push(v, Op::Relu(a.0))
    }

    pub fn backward(&self, out: Var) -> Grads {
        let mut g = vec![0.0; self.values.len()];
        g[out.0] = 1.0;
        for i in (0..=out.0).rev() {
            let gi = g[i];
            match self.ops[i] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    g[a] += gi;
                    g[b] += gi;
                }
                Op::Sub(a, b) => {
                    g[a] += gi;
                    g[b] -= gi;
                }
                Op::Mul(a, b) => {
                    g[a] += gi * self.values[b];
                    g[b] += gi * self.values[a];
                }
                Op::Div(a, b) => {
                    let (va, vb) = (self.values[a], self.values[b]);
                    g[a] += gi / vb;
                    g[b] -= gi * va / (vb * vb);
                }
                Op::Neg(a) => g[a] -= gi,
                Op::Exp(a) => g[a] += gi * self.values[i],
                Op::Ln(a) => g[a] += gi / self.values[a],
                Op::Tanh(a) => {
                    let t = self.values[i];
                    g[a] += gi * (1.0 - t * t);
                }
                Op::Relu(a) => {
                    if self.values[a] > 0.0 {
                        g[a] += gi;
                    }
                }
            }
        }
        Grads(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_rule_by_hand() {
        let mut t = Tape::new();
        let x = t.var(3.0);
        let y = t.var(4.0);
        let z = t.mul(x, y);
        let g = t.backward(z);
        assert_eq!(t.value(z), 12.0);
        assert_eq!(g.wrt(x), 4.0);
        assert_eq!(g.wrt(y), 3.0);
    }

    #[test]
    fn variable_used_twice_accumulates() {
        // f = x * x -> df/dx = 2x
        let mut t = Tape::new();
        let x = t.var(5.0);
        let f = t.mul(x, x);
        assert_eq!(t.backward(f).wrt(x), 10.0);
    }
}
