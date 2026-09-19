//! PCG32 (XSH-RR 64/32 variant).
//!
//! Uses integer operations only, so the sequence is identical on every
//! platform. There is deliberately NO normal-distribution generator here:
//! Box-Muller needs `ln`/`cos`, whose results can differ between libm
//! implementations. For weight initialization, use `uniform` (e.g.
//! Xavier-uniform) until our own `ln`/`exp` exist.

const MULT: u64 = 6364136223846793005;

#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// `seed` is the initial state, `stream` selects the sequence (same roles
    /// as `initstate` and `initseq` in the reference PCG implementation).
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Uniform in [0, 1) with 53 bits of mantissa.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform integer in [0, n) without modulo bias (rejection sampling).
    pub fn below(&mut self, n: u32) -> u32 {
        assert!(n > 0, "below(0) is undefined");
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Fisher-Yates. The result is fully determined by the RNG state.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below((i + 1) as u32) as usize;
            xs.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector from the official PCG demo (pcg32, seed=42, stream=54).
    /// These values were written from memory, not copied from the source. The
    /// test passes, which suggests they are right, but verify them against the
    /// official demo output before making public claims. If this test ever
    /// fails, check the values before blaming the code.
    #[test]
    fn reference_vector() {
        let mut r = Pcg32::new(42, 54);
        let got: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
        let want = [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e];
        assert_eq!(got, want);
    }

    #[test]
    fn same_seed_same_stream() {
        let mut a = Pcg32::new(1, 1);
        let mut b = Pcg32::new(1, 1);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_differs() {
        let mut a = Pcg32::new(1, 1);
        let mut b = Pcg32::new(2, 1);
        let same = (0..100).filter(|_| a.next_u32() == b.next_u32()).count();
        assert!(same < 5);
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Pcg32::new(7, 3);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn below_in_range_and_covers() {
        let mut r = Pcg32::new(9, 9);
        let mut seen = [false; 10];
        for _ in 0..1000 {
            let v = r.below(10) as usize;
            assert!(v < 10);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn shuffle_is_deterministic_permutation() {
        let mut a: Vec<u32> = (0..50).collect();
        let mut b = a.clone();
        Pcg32::new(5, 5).shuffle(&mut a);
        Pcg32::new(5, 5).shuffle(&mut b);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..50).collect::<Vec<u32>>());
        assert_ne!(a, (0..50).collect::<Vec<u32>>());
    }
}
