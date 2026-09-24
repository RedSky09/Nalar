//! nalar: an ML engine from scratch, `std` only.
//!
//! Status: M0 (scaffold), M1 (scalar autograd as a reference oracle), and the
//! core of M2 (tensor, layers, SGD, MLP). MNIST accuracy on real data is not
//! yet verified; see README.md.

/// Floating-point type used by the engine core (tensor, layers, optimizer).
/// The scalar oracle (`scalar_ad`), the RNG, and the gradcheck helpers stay
/// f64 on purpose. If this alias ever changes, `hash::hash_f64s` and the
/// gradcheck helpers need matching variants.
pub type Real = f64;

pub mod data;
pub mod gradcheck;
pub mod hash;
pub mod layers;
pub mod math;
pub mod mlp;
pub mod optim;
pub mod profile;
pub mod rng;
pub mod scalar_ad;
pub mod tensor;
