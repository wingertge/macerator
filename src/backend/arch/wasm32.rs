use crate::{
    backend::scalar::Fallback,
    wasm32::{Simd128Fallback, Simd128Relaxed},
    Simd,
};

use super::WithSimd;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
#[repr(u8)]
pub enum Arch {
    Scalar,
    Simd128Relaxed,
    Simd128Fallback,
}

impl Arch {
    pub fn new() -> Self {
        if Simd128Relaxed::is_available() {
            Self::Simd128Relaxed
        } else if Simd128Fallback::is_available() {
            Self::Simd128Fallback
        } else {
            Self::Scalar
        }
    }

    pub fn dispatch<Op: WithSimd>(self, op: Op) -> Op::Output {
        match self {
            Arch::Scalar => <Fallback as Simd>::vectorize(op),
            Arch::Simd128Relaxed => <Simd128Relaxed as Simd>::vectorize(op),
            Arch::Simd128Fallback => <Simd128Fallback as Simd>::vectorize(op),
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Self::new()
    }
}
