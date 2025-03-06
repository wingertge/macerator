use crate::{
    Simd,
    backend::{scalar::Fallback, x86::v2::V2, x86::v3::V3},
};

use super::WithSimd;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
#[repr(u8)]
pub enum Arch {
    Scalar,
    V2,
    V3,
}

impl Arch {
    pub fn new() -> Self {
        if V3::is_available() {
            Self::V3
        } else if V2::is_available() {
            Self::V2
        } else {
            Self::Scalar
        }
    }

    pub fn dispatch<Op: WithSimd>(self, op: Op) -> Op::Output {
        match self {
            Arch::Scalar => <Fallback as Simd>::vectorize(op),
            Arch::V2 => <V2 as Simd>::vectorize(op),
            Arch::V3 => <V3 as Simd>::vectorize(op),
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Self::new()
    }
}
