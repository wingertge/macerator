use crate::{
    backend::{aarch64::*, scalar::Fallback},
    Simd,
};

use super::WithSimd;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
#[repr(u8)]
pub enum Arch {
    Scalar,
    NeonFma,
    #[cfg(feature = "fp16")]
    NeonF16,
}

impl Arch {
    pub fn detect() -> Self {
        #[cfg(feature = "fp16")]
        if NeonFP16::is_available() {
            return Self::NeonF16;
        }
        if NeonFma::is_available() {
            Self::NeonFma
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
            Arch::NeonFma => <NeonFma as Simd>::vectorize(op),
            #[cfg(feature = "fp16")]
            Arch::NeonF16 => <NeonFP16 as Simd>::vectorize(op),
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Self::detect()
    }
}
