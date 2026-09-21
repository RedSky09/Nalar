//! Layers with hand-written backprop.
//!
//! Conventions shared by every layer:
//! - `forward` caches whatever `backward` needs, so it takes `&mut self`.
//! - `backward` takes the gradient w.r.t. the layer's output and returns the
//!   gradient w.r.t. its input.
//! - Parameter gradients are OVERWRITTEN by `backward`, not accumulated:
//!   one backward call corresponds to one batch.
//! - Batch dimension first: activations are `[batch, features]`.

pub mod linear;
pub mod relu;
pub mod softmax_ce;

pub use linear::Linear;
pub use relu::Relu;
pub use softmax_ce::SoftmaxCrossEntropy;
