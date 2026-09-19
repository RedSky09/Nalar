//! 64-bit FNV-1a for golden tests. Not a cryptographic hash; it is enough to
//! detect bit-level changes in weights or outputs.

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hashes the bit patterns explicitly as little-endian, so the result does not
/// depend on host endianness.
pub fn hash_f64s(xs: &[f64]) -> u64 {
    let mut buf = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        buf.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    fnv1a64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_fnv_values() {
        // Commonly used FNV-1a 64 test vectors: the empty string yields the offset basis.
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn sensitive_to_single_bit() {
        let a = [1.0_f64, 2.0];
        let b = [1.0_f64, f64::from_bits(2.0_f64.to_bits() + 1)];
        assert_ne!(hash_f64s(&a), hash_f64s(&b));
    }
}
