//! Numerical gradient checking (central difference).
//!
//! Use f64 for gradchecks even if the model later runs in f32: the tolerance
//! can be much tighter, and a failure almost certainly means a backprop bug
//! rather than rounding noise.

/// Numerical gradient of `f` at point `x`, with step size `h`.
pub fn numeric_grad<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<f64> {
    let mut xs = x.to_vec();
    let mut g = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let orig = xs[i];
        xs[i] = orig + h;
        let fp = f(&xs);
        xs[i] = orig - h;
        let fm = f(&xs);
        xs[i] = orig;
        g.push((fp - fm) / (2.0 * h));
    }
    g
}

/// Maximum error, a mix of absolute and relative:
/// |a - n| / max(1, |a|, |n|).
/// For small gradients this behaves like an absolute error, for large ones
/// like a relative error. If a case does not fit, change the formula here
/// instead of loosening the tolerance in individual tests.
pub fn max_error(analytic: &[f64], numeric: &[f64]) -> f64 {
    assert_eq!(analytic.len(), numeric.len());
    analytic
        .iter()
        .zip(numeric)
        .map(|(a, n)| (a - n).abs() / 1.0_f64.max(a.abs()).max(n.abs()))
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_grad_of_quadratic() {
        // f(x, y) = x^2 + 3xy -> grad = (2x + 3y, 3x)
        let f = |v: &[f64]| v[0] * v[0] + 3.0 * v[0] * v[1];
        let g = numeric_grad(&f, &[2.0, -1.0], 1e-6);
        assert!((g[0] - 1.0).abs() < 1e-6);
        assert!((g[1] - 6.0).abs() < 1e-6);
    }
}
