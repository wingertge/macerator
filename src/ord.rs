use paste::paste;
use pulp::Simd;

use crate::Vectorizable;

pub trait VEq: Vectorizable + PartialEq {
    /// Compare two vectors for elementwise equality
    fn veq<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S>;
}

macro_rules! impl_veq {
    ($ty: ty) => {
        paste! {
            impl VEq for $ty {
                #[inline(always)]
                fn veq<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S> {
                    simd.[<equal_ $ty s>](a, b)
                }
            }
        }
    };
    ($($ty: ty),*) => {
        $(impl_veq!($ty);)*
    }
}

impl_veq!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);

pub trait VOrd: VEq {
    /// Apply elementwise [`PartialOrd::lt`] on two vectors
    fn vlt<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S>;
    /// Apply elementwise [`PartialOrd::le`] on two vectors
    fn vle<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S>;
    /// Apply elementwise [`PartialOrd::gt`] on two vectors
    fn vgt<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S>;
    /// Apply elementwise [`PartialOrd::ge`] on two vectors
    fn vge<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S>;
    /// Apply elementwise [`Ord::min`] to two vectors
    fn vmin<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    /// Apply elementwise [`Ord::max`] on two vectors
    fn vmax<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
}

macro_rules! impl_vord {
    ($ty: ty) => {
        paste! {
            impl VOrd for $ty {
                #[inline(always)]
                fn vlt<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S> {
                    simd.[<less_than_ $ty s>](a, b)
                }
                #[inline(always)]
                fn vle<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S> {
                    simd.[<less_than_or_equal_ $ty s>](a, b)
                }
                #[inline(always)]
                fn vgt<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S> {
                    simd.[<greater_than_ $ty s>](a, b)
                }
                #[inline(always)]
                fn vge<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Mask<S> {
                    simd.[<greater_than_or_equal_ $ty s>](a, b)
                }
                #[inline(always)]
                fn vmin<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<min_ $ty s>](a, b)
                }
                #[inline(always)]
                fn vmax<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<max_ $ty s>](a, b)
                }
            }
        }
    };
    ($($ty: ty),*) => {
        $(impl_vord!($ty);)*
    }
}

impl_vord!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
