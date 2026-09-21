//! Three-way cross-check on the shared fixture in `common`:
//!   manual backprop  vs  scalar tape autograd  vs  finite differences.
//! The fixture packs the input and all parameters into one flat vector, so the
//! input gradient is checked as well.

mod common;

use nalar::gradcheck::{max_error, numeric_grad};

const H: f64 = 1e-6;

#[test]
fn fixture_is_away_from_relu_kink() {
    let p = common::init_params();
    let m = common::min_abs_preactivation(&p);
    assert!(m > 1e-3, "smallest |pre-activation| is {m:e}; finite differences would be unreliable");
}

#[test]
fn manual_matches_scalar_tape() {
    let p = common::init_params();
    let (loss_m, g_m) = common::manual(&p);
    let (loss_t, g_t) = common::tape(&p);
    assert!((loss_m - loss_t).abs() < 1e-12, "loss differs: {loss_m} vs {loss_t}");
    let err = max_error(&g_m, &g_t);
    assert!(err < 1e-10, "manual vs tape gradients: max error {err:e}");
}

#[test]
fn manual_matches_finite_differences() {
    let p = common::init_params();
    let (_, g) = common::manual(&p);
    let num = numeric_grad(&|v: &[f64]| common::manual(v).0, &p, H);
    let err = max_error(&g, &num);
    assert!(err < 1e-6, "manual vs finite differences: max error {err:e}");
}

#[test]
fn scalar_tape_matches_finite_differences() {
    let p = common::init_params();
    let (_, g) = common::tape(&p);
    let num = numeric_grad(&|v: &[f64]| common::tape(v).0, &p, H);
    let err = max_error(&g, &num);
    assert!(err < 1e-6, "tape vs finite differences: max error {err:e}");
}
