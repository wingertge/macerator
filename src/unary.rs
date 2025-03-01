use core::ops::Not;

use paste::paste;
use pulp::Simd;

use crate::Vectorizable;

pub trait VRecip: Vectorizable {
    fn vrecip<S: Simd>(simd: S, input: Self::Vector<S>) -> Self::Vector<S>;
}

#[cfg(feature = "std")]
pub trait VSqrt: Vectorizable {
    fn vsqrt<S: Simd>(simd: S, input: Self::Vector<S>) -> Self::Vector<S>;
}

pub trait VAbs: Vectorizable {
    fn vabs<S: Simd>(simd: S, input: Self::Vector<S>) -> Self::Vector<S>;
}

pub trait VBitNot: Vectorizable + Not<Output = Self> {
    fn vbitnot<S: Simd>(simd: S, input: Self::Vector<S>) -> Self::Vector<S>;
}

macro_rules! impl_unop {
    ($trait: ident, $name: ident, $ty: ty) => {
        paste! {
            impl $trait for $ty {
                fn [<$trait:lower>]<S: Simd>(simd: S, input: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<$name _ $ty s>](input)
                }
            }
        }
    };
    ($trait: ident, $name: ident, $($ty: ty),*) => {
        $(impl_unop!($trait, $name, $ty);)*
    }
}

impl_unop!(VRecip, recip, f32);
#[cfg(feature = "std")]
impl_unop!(VSqrt, sqrt, f32, f64);
impl_unop!(VAbs, abs, i8, i16, i32, f32, f64);
impl_unop!(VBitNot, not, i8, u8, i16, u16, i32, u32, i64, u64);
