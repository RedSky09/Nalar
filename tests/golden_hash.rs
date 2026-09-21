//! Golden test: bit-for-bit hash of output that should be deterministic.
//! The constant was recorded once (on Linux); CI then compares it on Linux,
//! macOS, and Windows. Later, add hashes of the weights after N training steps.

mod common;

use nalar::hash::hash_f64s;
use nalar::rng::Pcg32;

const GOLDEN_RNG_F64_1000: u64 = 0x9da8fdfdf482cf2c;

#[test]
fn rng_stream_golden() {
    let mut r = Pcg32::new(42, 54);
    let xs: Vec<f64> = (0..1000).map(|_| r.next_f64()).collect();
    assert_eq!(hash_f64s(&xs), GOLDEN_RNG_F64_1000, "actual hash: {:#018x}", hash_f64s(&xs));
}

/// libm canary. Unlike the RNG stream, this path calls `exp` and `ln` from the
/// platform libm. Passing on all OSes is weak evidence (few inputs), but a
/// failure on one OS only is a strong finding: it means libm differs, and the
/// open item in docs/design.md section 3 must be resolved before M4.
const GOLDEN_MLP_LOSS_AND_GRADS: u64 = 0x2232d732a1d84408;

#[test]
fn mlp_loss_and_grads_golden() {
    let (loss, grads) = common::manual(&common::init_params());
    let mut all = vec![loss];
    all.extend(grads);
    assert_eq!(hash_f64s(&all), GOLDEN_MLP_LOSS_AND_GRADS, "actual hash: {:#018x}", hash_f64s(&all));
}
