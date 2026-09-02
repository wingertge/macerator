use crate::{backend::scalar::Fallback, wasm32, Simd};

use super::WithSimd;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
#[repr(u8)]
pub enum Arch {
    Scalar,
    #[cfg(relaxed_simd)]
    Simd128Relaxed,
    Simd128Fallback,
}

impl Arch {
    pub fn detect() -> Self {
        #[cfg(relaxed_simd)]
        if wasm32::Simd128Relaxed::is_available() {
            return Self::Simd128Relaxed;
        }
        if wasm32::Simd128Fallback::is_available() {
            Self::Simd128Fallback
        } else {
            Self::Scalar
        }
    }

    /// Dispatch a function on this [`Arch`]
    ///
    /// # Safety
    /// Required features for the [`Arch`] must be available.
    pub unsafe fn dispatch<Op: WithSimd>(self, op: Op) -> Op::Output {
        match self {
            Arch::Scalar => <Fallback as Simd>::vectorize(op),
            #[cfg(relaxed_simd)]
            Arch::Simd128Relaxed => <wasm32::Simd128Relaxed as Simd>::vectorize(op),
            Arch::Simd128Fallback => <wasm32::Simd128Fallback as Simd>::vectorize(op),
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Self::detect()
    }
}
