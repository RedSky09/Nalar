//! Deterministic `exp` and `ln`, built only from IEEE-754 basic operations.
//!
//! Why this exists: the platform libm gives different bits for `exp`/`ln` on
//! different operating systems (measured: Linux, macOS, and Windows produce
//! three different hashes over the same 20,000 inputs; see
//! `examples/libm_probe.rs` and docs/design.md, section 3). Addition,
//! subtraction, multiplication, division, and comparisons are exactly specified
//! by IEEE-754, and integer bit manipulation is exact, so a function built only
//! from those produces the same bits everywhere, provided the compiler does not
//! fuse or reassociate operations. Rust does neither on its own (as far as I
//! know; verify for the compiler version you use).
//!
//! What is deliberately NOT used: `f64::exp`, `f64::ln`, `powi`, `floor`,
//! `round`, `mul_add`, or anything else that reaches libm or hardware-specific
//! instructions.
//!
//! Accuracy is not perfect and is measured, not assumed: see the tests in this
//! module and the numbers in docs/design.md, section 8.

const LN2_HI: f64 = 0.6931471803691238; // ln 2 with 32 significant bits, so k * LN2_HI is exact
const LN2_LO: f64 = 1.9082149292705877e-10; // ln 2 - LN2_HI
const INV_LN2: f64 = 1.4426950408889634;
const SQRT2: f64 = 1.4142135623730951;
const TWO54: f64 = 1.8014398509481984e16;

/// Largest x for which the correctly rounded exp(x) is finite.
const EXP_OVERFLOW: f64 = 709.782712893384;
/// Smallest x for which the correctly rounded exp(x) is not 0.
const EXP_UNDERFLOW: f64 = -745.1332191019411;

const MANT_MASK: u64 = 0x000f_ffff_ffff_ffff;
const ONE_EXP: u64 = 0x3ff0_0000_0000_0000;

/// 2^k for k in [-1022, 1023], built directly from the exponent field.
#[inline]
fn pow2(k: i64) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// `p * 2^k` with at most one rounding, including results that overflow or are
/// subnormal. Every intermediate product is exact except the last one.
#[inline]
fn scale(p: f64, k: i64) -> f64 {
    if k > 1023 {
        (p * pow2(1023)) * pow2(k - 1023)
    } else if k < -1022 {
        (p * pow2(-500)) * pow2(k + 500)
    } else {
        p * pow2(k)
    }
}

/// e^x.
///
/// Method: x = k*ln2 + r with |r| <= ~0.35, then exp(r) from a degree-13 Taylor
/// polynomial, then scaling by 2^k. The sum 1 + r is kept exact as a
/// (hi, lo) pair so only one rounding error enters at the end.
pub fn exp(x: f64) -> f64 {
    if x != x {
        return x; // NaN
    }
    if x > EXP_OVERFLOW {
        return f64::INFINITY;
    }
    if x < EXP_UNDERFLOW {
        return 0.0;
    }
    // k = x / ln2 rounded to nearest, via truncation toward zero (a defined,
    // deterministic conversion), not via `round()`.
    let t = x * INV_LN2;
    let k = (if t >= 0.0 { t + 0.5 } else { t - 0.5 }) as i64;
    let kf = k as f64;
    let r = (x - kf * LN2_HI) - kf * LN2_LO;

    // 1/n! for n = 2..=13, Horner in r.
    let q = 0.5
        + r * (0.16666666666666666
            + r * (0.041666666666666664
                + r * (0.008333333333333333
                    + r * (0.001388888888888889
                        + r * (0.0001984126984126984
                            + r * (2.48015873015873e-05
                                + r * (2.7557319223985893e-06
                                    + r * (2.755731922398589e-07
                                        + r * (2.505210838544172e-08
                                            + r * (2.08767569878681e-09
                                                + r * 1.6059043836821613e-10))))))))));

    // exp(r) = 1 + r + r^2 * q, with 1 + r as an exact (hi, lo) pair (Fast2Sum).
    let hi = 1.0 + r;
    let lo = r - (hi - 1.0);
    let p = hi + (lo + (r * r) * q);
    scale(p, k)
}

/// Natural logarithm.
///
/// Method: x = m * 2^e with m in [sqrt(2)/2, sqrt(2)], then
/// ln(m) = ln(1 + f) with f = m - 1, evaluated through s = f / (2 + f) and the
/// odd series 2 * atanh(s), arranged so that f (which is exact) is the leading
/// term; plus e * ln2 (split in two parts).
pub fn ln(x: f64) -> f64 {
    if x != x {
        return x; // NaN
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }
    let mut bits = x.to_bits();
    let mut e: i64 = 0;
    if (bits >> 52) == 0 {
        // Subnormal: scaling by 2^54 is exact and makes it normal.
        bits = (x * TWO54).to_bits();
        e = -54;
    }
    e += ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & MANT_MASK) | ONE_EXP); // in [1, 2)
    if m > SQRT2 {
        m *= 0.5; // exact
        e += 1;
    }
    let f = m - 1.0; // exact (Sterbenz)
    let s = f / (2.0 + f);
    let z = s * s;

    // Tail of the series: r = 2/3 z + 2/5 z^2 + ... + 2/21 z^10, i.e. everything
    // beyond the leading 2s term is s * r. Coefficients 2/(2n+1), n = 1..=10.
    let r = z
        * (0.6666666666666666
            + z * (0.4
                + z * (0.2857142857142857
                    + z * (0.2222222222222222
                        + z * (0.18181818181818182
                            + z * (0.15384615384615385
                                + z * (0.13333333333333333
                                    + z * (0.11764705882352941
                                        + z * (0.10526315789473684
                                            + z * 0.09523809523809523)))))))));

    // Identity: 2s = f - f^2/2 + s * f^2/2. So
    //   ln(1 + f) = f - (hfsq - s * (hfsq + r)),  hfsq = f^2 / 2.
    // f is exact, so the rounding error of s (from the division) only enters
    // through the small correction terms, not the leading term.
    let hfsq = (0.5 * f) * f;
    if e == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        let ef = e as f64;
        ef * LN2_HI - ((hfsq - (s * (hfsq + r) + ef * LN2_LO)) - f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg32;

    /// Maps a double to an integer such that neighbouring doubles differ by 1
    /// (works across zero as well).
    fn ordered(x: f64) -> i64 {
        let b = x.to_bits() as i64;
        if b < 0 {
            i64::MIN - b
        } else {
            b
        }
    }

    fn ulp_diff(a: f64, b: f64) -> u64 {
        ordered(a).abs_diff(ordered(b))
    }

    #[test]
    fn constants_are_consistent() {
        // k * LN2_HI must be exact for every k we can produce (|k| < 2^11).
        assert_eq!(LN2_HI.to_bits() & ((1 << 21) - 1), 0);
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(1), 2.0);
        assert_eq!(pow2(-1), 0.5);
        assert_eq!(pow2(-1022), f64::MIN_POSITIVE);
        assert_eq!(TWO54, 18014398509481984.0);
    }

    #[test]
    fn exp_special_values() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert!(exp(f64::NAN).is_nan());
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(exp(-746.0), 0.0);
        assert!(exp(EXP_OVERFLOW).is_finite());
        assert_eq!(exp(f64::from_bits(EXP_OVERFLOW.to_bits() + 1)), f64::INFINITY);
        assert!(exp(EXP_UNDERFLOW) > 0.0);
        // EXP_UNDERFLOW is negative, so +1 in the bit pattern is the next double DOWN.
        assert_eq!(exp(f64::from_bits(EXP_UNDERFLOW.to_bits() + 1)), 0.0);
        assert!((exp(1.0) - std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn exp_subnormal_range() {
        // Results below 2^-1022 are subnormal; they must be positive, and must
        // decrease monotonically.
        let mut prev = f64::INFINITY;
        let mut x = -708.0;
        while x > -745.0 {
            let y = exp(x);
            assert!(y > 0.0 && y <= prev, "not monotone or not positive at x = {x}");
            prev = y;
            x -= 0.25;
        }
    }

    #[test]
    fn ln_special_values() {
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert_eq!(ln(-0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert!(ln(f64::NAN).is_nan());
        assert_eq!(ln(f64::INFINITY), f64::INFINITY);
        assert!((ln(std::f64::consts::E) - 1.0).abs() < 1e-15);
        assert_eq!(ln(2.0), LN2_HI + LN2_LO);
        // The smallest subnormal, 2^-1074, has ln = -1074 * ln 2 = -744.4400719213812...
        assert!((ln(f64::from_bits(1)) + 744.4400719213812).abs() < 1e-10);
    }

    #[test]
    fn exp_and_ln_are_inverse_within_a_few_ulps() {
        let mut rng = Pcg32::new(5, 5);
        for _ in 0..20_000 {
            let x = rng.uniform(-700.0, 700.0);
            let back = ln(exp(x));
            // Relative error of exp(x) becomes an absolute error of about 1e-16 in ln.
            assert!((back - x).abs() <= 4e-16 * x.abs().max(1.0), "x = {x}, ln(exp(x)) = {back}");
        }
    }

    /// Cross-check against the platform libm. The two implementations are both
    /// accurate to about 1 ulp, so they should never be far apart. This test
    /// checks closeness only; it must NOT be used to assert bit equality.
    #[test]
    fn close_to_platform_libm() {
        let mut rng = Pcg32::new(21, 3);
        let mut worst_exp = 0;
        let mut worst_ln = 0;
        for _ in 0..200_000 {
            let x = rng.uniform(-745.0, 709.7);
            worst_exp = worst_exp.max(ulp_diff(exp(x), x.exp()));
            let y = f64::from_bits(rng.next_u64() & 0x7fef_ffff_ffff_ffff);
            if y > 0.0 {
                worst_ln = worst_ln.max(ulp_diff(ln(y), y.ln()));
            }
        }
        assert!(worst_exp <= MAX_ULP_VS_LIBM, "exp differs from libm by {worst_exp} ulp");
        assert!(worst_ln <= MAX_ULP_VS_LIBM, "ln differs from libm by {worst_ln} ulp");
    }

    // Measured on glibc: ours <= 0.84 ulp from the true value, glibc <= 0.52.
    // The limit of 3 leaves room for other libms, whose accuracy I have not measured;
    // this test only catches gross bugs and is not an accuracy certificate.
    const MAX_ULP_VS_LIBM: u64 = 3;

    /// Reference values: the correctly rounded double of the true result,
    /// computed offline with 300-bit arithmetic (mpmath). They do not depend on
    /// any libm. Measured worst error of `exp` is 0.76 ulp from the true value,
    /// so it must land within 1 ulp of the correctly rounded double.
    const EXP_REF: &[(u64, u64)] = &[
        (0xc087480000000000, 0x0000000000000001), // f(-745.0) = 5e-324
        (0xc086261e8a428d69, 0x000b118351bb24c7), // f(-708.7649121474652) = 1.5392518623268697e-308
        (0xc03e000000000000, 0x3d3a56e0c2ac7f75), // f(-30.0) = 9.357622968840175e-14
        (0xbff0000000000000, 0x3fd78b56362cef38), // f(-1.0) = 0.36787944117144233
        (0xbfe0000000000000, 0x3fe368b2fc6f960a), // f(-0.5) = 0.6065306597126334
        (0x3ddb7cdfd9d7bdbb, 0x3ff000000006df38), // f(1e-10) = 1.0000000001
        (0x3fd3333333333333, 0x3ff599058c8c1a96), // f(0.3) = 1.3498588075760032
        (0x3ff0000000000000, 0x4005bf0a8b145769), // f(1.0) = 2.718281828459045
        (0x4004000000000000, 0x40285d6fd931e0bb), // f(2.5) = 12.182493960703473
        (0x4059000000000000, 0x48f3494a9b171bf5), // f(100.0) = 2.6881171418161356e+43
        (0x4086280000000000, 0x7fdd422d2be5dc9b), // f(709.0) = 8.218407461554972e+307
        (0x40862e42fefa39ef, 0x7fefffffffffff2a), // f(709.782712893384) = 1.7976931348622732e+308
        (0x3fe62e42fefa39ef, 0x4000000000000000), // f(0.6931471805599453) = 2.0
        (0xbfe62e42fefa39ef, 0x3fe0000000000000), // f(-0.6931471805599453) = 0.5
        (0x3ee4f8b588e368f1, 0x3ff0000a7c5e340e), // f(1e-05) = 1.00001000005
    ];

    /// Same for `ln` (measured worst error 0.84 ulp).
    const LN_REF: &[(u64, u64)] = &[
        (0x3fe69105560a0b43, 0xbfd65a70521436c5), // f(0.7052027397588535) = -0.3492699433853484
        (0x3ff000f99175c8fb, 0x3f2f313b76b71ee1), // f(1.000238006785877) = 0.00023797846675520983
        (0x3fe0000000000000, 0xbfe62e42fefa39ef), // f(0.5) = -0.6931471805599453
        (0x4000000000000000, 0x3fe62e42fefa39ef), // f(2.0) = 0.6931471805599453
        (0x4008000000000000, 0x3ff193ea7aad030b), // f(3.0) = 1.0986122886681098
        (0x4024000000000000, 0x40026bb1bbb55516), // f(10.0) = 2.302585092994046
        (0x01a56e1fc2f8f359, 0xc085963447f87fb5), // f(1e-300) = -690.7755278982137
        (0x7e37e43c8800759c, 0x4085963447f87fb5), // f(1e+300) = 690.7755278982137
        (0x0000000000000001, 0xc0874385446d71c3), // f(5e-324) = -744.4400719213812
        (0x3ff8000000000000, 0x3fd9f323ecbf984c), // f(1.5) = 0.4054651081081644
        (0x3fefffffffffffff, 0xbca0000000000000), // f(0.9999999999999999) = -1.1102230246251565e-16
        (0x3ff0000000000001, 0x3cafffffffffffff), // f(1.0000000000000002) = 2.2204460492503128e-16
        (0x3ff6a09e667f3bcd, 0x3fd62e42fefa39f0), // f(1.4142135623730951) = 0.3465735902799727
        (0x3fe6a09e667f3bcd, 0xbfd62e42fefa39ee), // f(0.7071067811865476) = -0.3465735902799726
        (0x419d6f3454000000, 0x4032a1a38bd0540a), // f(123456789.0) = 18.63140176616802
    ];

    #[test]
    fn matches_high_precision_reference() {
        for &(xb, yb) in EXP_REF {
            let (x, want) = (f64::from_bits(xb), f64::from_bits(yb));
            let got = exp(x);
            assert!(ulp_diff(got, want) <= 1, "exp({x:e}) = {got:e}, want {want:e}");
        }
        for &(xb, yb) in LN_REF {
            let (x, want) = (f64::from_bits(xb), f64::from_bits(yb));
            let got = ln(x);
            assert!(ulp_diff(got, want) <= 1, "ln({x:e}) = {got:e}, want {want:e}");
        }
    }
}
