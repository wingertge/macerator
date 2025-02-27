use core::ops::{BitAnd, BitOr, BitXor};

use paste::paste;
use pulp::Simd;

use crate::Vectorizable;

pub trait VBitXor: Vectorizable + BitXor<Output = Self> {
    fn vbitxor<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}
pub trait VBitAnd: Vectorizable + BitAnd<Output = Self> {
    fn vbitand<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}
pub trait VBitOr: Vectorizable + BitOr<Output = Self> {
    fn vbitor<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}

macro_rules! impl_bitwise_binop {
    ($ty: ty) => {
        paste! {
            impl VBitXor for $ty {
                fn vbitxor<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<xor_ $ty s>](lhs, rhs)
                }
            }
            impl VBitAnd for $ty {
                fn vbitand<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<and_ $ty s>](lhs, rhs)
                }
            }
            impl VBitOr for $ty {
                fn vbitor<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<or_ $ty s>](lhs, rhs)
                }
            }
        }
    };
    ($($ty: ty),*) => {
        $(impl_bitwise_binop!($ty);)*
    }
}

impl_bitwise_binop!(u8, i8, u16, i16, u32, i32, u64, i64);
