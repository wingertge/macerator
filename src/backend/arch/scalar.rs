use crate::Simd;

use super::WithSimd;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
#[repr(u8)]
pub enum Arch {
    Scalar,
}

impl Arch {
    pub fn detect() -> Self {
        Self::Scalar
    }

    /// Dispatch a function on this [`Arch`]
    ///
    /// # Safety
    /// Required features for the [`Arch`] must be available.
    pub unsafe fn dispatch<Op: WithSimd>(self, op: Op) -> Op::Output {
        match self {
            Arch::Scalar => <crate::scalar::Fallback as Simd>::vectorize(op),
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Self::detect()
    }
}
