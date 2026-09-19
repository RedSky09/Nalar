//! Cross-check: scalar autograd gradients vs finite differences.

use nalar::gradcheck::{max_error, numeric_grad};
use nalar::scalar_ad::{Tape, Var};

type Build = fn(&mut Tape, &[Var]) -> Var;

fn analytic_and_numeric(build: Build, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut t = Tape::new();
    let vars: Vec<Var> = x.iter().map(|&v| t.var(v)).collect();
    let out = build(&mut t, &vars);
    let g = t.backward(out);
    let analytic = vars.iter().map(|&v| g.wrt(v)).collect();

    let f = |xs: &[f64]| {
        let mut t = Tape::new();
        let vs: Vec<Var> = xs.iter().map(|&v| t.var(v)).collect();
        let o = build(&mut t, &vs);
        t.value(o)
    };
    (analytic, numeric_grad(&f, x, 1e-6))
}

fn check(name: &str, build: Build, x: &[f64]) {
    let (a, n) = analytic_and_numeric(build, x);
    let err = max_error(&a, &n);
    assert!(err < 1e-6, "{name}: error {err:e}\n analytic={a:?}\n numeric ={n:?}");
}

#[test]
fn polynomial() {
    check("poly", |t, v| {
        let xy = t.mul(v[0], v[1]);
        let xx = t.mul(v[0], v[0]);
        let s = t.add(xy, xx);
        t.sub(s, v[1])
    }, &[1.3, -0.7]);
}

#[test]
fn rational_exp_ln() {
    check("exp/div/ln", |t, v| {
        let one = t.var(1.0);
        let yy = t.mul(v[1], v[1]);
        let den = t.add(one, yy);
        let e = t.exp(v[0]);
        let q = t.div(e, den);
        let zz = t.mul(v[2], v[2]);
        let arg = t.add(zz, one);
        let l = t.ln(arg);
        t.add(q, l)
    }, &[0.4, -1.2, 0.9]);
}

#[test]
fn tanh_of_affine() {
    check("tanh", |t, v| {
        let xy = t.mul(v[0], v[1]);
        let s = t.add(xy, v[2]);
        t.tanh(s)
    }, &[0.5, -0.8, 0.3]);
}

#[test]
fn relu_away_from_kink() {
    // Avoid the point 0: finite differences are not valid at the kink.
    check("relu", |t, v| {
        let r1 = t.relu(v[0]);
        let m = t.mul(r1, v[1]);
        let r2 = t.relu(v[2]);
        t.add(m, r2)
    }, &[0.7, 1.5, -0.4]);
}

#[test]
fn fan_out_and_neg() {
    check("fan-out", |t, v| {
        let s = t.add(v[0], v[0]);
        let q = t.mul(v[0], v[0]);
        let p = t.mul(s, q);
        let d = t.sub(v[0], v[1]);
        let n = t.neg(d);
        t.mul(p, n)
    }, &[1.1, 0.6]);
}

/// A test for the tests: make sure the gradcheck can actually fail.
/// If this passes while the gradient is deliberately corrupted, the whole
/// suite is meaningless.
#[test]
fn checker_detects_wrong_gradient() {
    let build: Build = |t, v| t.mul(v[0], v[1]);
    let (mut a, n) = analytic_and_numeric(build, &[2.0, 3.0]);
    a[0] += 1e-3; // corrupt it slightly
    assert!(max_error(&a, &n) > 1e-6);
}
