//! Golden hashes of our own `exp`/`ln` (src/math.rs) over the same inputs that
//! `examples/libm_probe.rs` uses. Unlike the libm hashes, these must be the same
//! on every operating system: CI compares Linux, macOS, and Windows. The
//! constants were recorded on Linux.

use nalar::hash::hash_f64s;
use nalar::math;
use nalar::rng::Pcg32;

const N: usize = 20_000;
const GOLDEN_EXP: u64 = 0xc52cc763eb29344f;
const GOLDEN_LN: u64 = 0x18e01805855c6814;

fn inputs() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Pcg32::new(2024, 1);
    let xs = (0..N).map(|_| rng.uniform(-30.0, 0.0)).collect();
    let ys = (0..N).map(|_| rng.uniform(1.0, 10.0)).collect();
    (xs, ys)
}

#[test]
fn own_exp_golden() {
    let (xs, _) = inputs();
    let out: Vec<f64> = xs.iter().map(|&x| math::exp(x)).collect();
    assert_eq!(hash_f64s(&out), GOLDEN_EXP, "actual hash: {:#018x}", hash_f64s(&out));
}

#[test]
fn own_ln_golden() {
    let (_, ys) = inputs();
    let out: Vec<f64> = ys.iter().map(|&y| math::ln(y)).collect();
    assert_eq!(hash_f64s(&out), GOLDEN_LN, "actual hash: {:#018x}", hash_f64s(&out));
}
