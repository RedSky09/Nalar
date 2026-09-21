//! Diagnostic for the open libm question in docs/design.md, section 3.
//!
//! Hashes the bit patterns of `exp` and `ln` over a fixed set of inputs that is
//! identical on every platform (integer-only PCG32, exact scaling). Run it on
//! two machines and compare the printed hashes: if they differ, the platform
//! libm gives different bits for at least one input.
//!
//! The input ranges mirror what softmax + cross-entropy feed into these
//! functions: `exp` gets max-shifted logits (<= 0), `ln` gets a sum of
//! exponentials in [1, n_classes].
//!
//! The last two lines hash the outputs of our own `nalar::math` on the same
//! inputs; unlike the libm lines above, those must match on every platform.
//!
//! Usage: `cargo run --release --example libm_probe`

use nalar::hash::hash_f64s;
use nalar::math;
use nalar::rng::Pcg32;

const N: usize = 20_000;

fn main() {
    let mut rng = Pcg32::new(2024, 1);
    let xs: Vec<f64> = (0..N).map(|_| rng.uniform(-30.0, 0.0)).collect();
    let ys: Vec<f64> = (0..N).map(|_| rng.uniform(1.0, 10.0)).collect();

    // Sanity: the inputs themselves must hash identically everywhere.
    println!("platform:     {}-{}", std::env::consts::OS, std::env::consts::ARCH);
    println!("inputs xs:    {:#018x}", hash_f64s(&xs));
    println!("inputs ys:    {:#018x}", hash_f64s(&ys));

    let e: Vec<f64> = xs.iter().map(|x| x.exp()).collect();
    let l: Vec<f64> = ys.iter().map(|y| y.ln()).collect();
    println!("exp outputs:  {:#018x}   ({N} inputs in [-30, 0))", hash_f64s(&e));
    println!("ln  outputs:  {:#018x}   ({N} inputs in [1, 10))", hash_f64s(&l));

    // Our own implementations (src/math.rs). These two lines must be identical on every platform.
    let me: Vec<f64> = xs.iter().map(|&x| math::exp(x)).collect();
    let ml: Vec<f64> = ys.iter().map(|&y| math::ln(y)).collect();
    println!("own exp:      {:#018x}", hash_f64s(&me));
    println!("own ln:       {:#018x}", hash_f64s(&ml));
}
