//! Golden test: bit-for-bit hash of output that should be deterministic.
//! The constant was recorded once (on Linux); CI then compares it on Linux,
//! macOS, and Windows. Later, add hashes of the weights after N training steps.

use nalar::hash::hash_f64s;
use nalar::rng::Pcg32;

const GOLDEN_RNG_F64_1000: u64 = 0x9da8fdfdf482cf2c;

#[test]
fn rng_stream_golden() {
    let mut r = Pcg32::new(42, 54);
    let xs: Vec<f64> = (0..1000).map(|_| r.next_f64()).collect();
    assert_eq!(hash_f64s(&xs), GOLDEN_RNG_F64_1000, "actual hash: {:#018x}", hash_f64s(&xs));
}
