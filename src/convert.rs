use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VConvert: Scalar {
    /// The signed integer of the same width.
    type Int: Scalar;
    fn vtrunc_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self::Int>;
    fn vround_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self::Int>;
    fn vfrom_int<S: Simd>(input: Vector<S, Self::Int>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

/// Conversion between float lanes and signed integer lanes of the same width
/// (f32 and i32, f64 and i64), with the results of Rust's `as` on every
/// backend.
impl<S: Simd, T: VConvert> Vector<S, T> {
    /// Elementwise conversion to integers, rounding toward zero like `x as
    /// i32`: out-of-range values saturate to the integer's minimum or
    /// maximum, and NaN becomes 0.
    #[inline(always)]
    pub fn trunc_to_int(self) -> Vector<S, T::Int> {
        T::vtrunc_to_int(self)
    }

    /// Elementwise conversion to integers, rounding to the nearest with ties
    /// to even, like `x.round_ties_even() as i32`: saturating, and NaN
    /// becomes 0.
    #[inline(always)]
    pub fn round_to_int(self) -> Vector<S, T::Int> {
        T::vround_to_int(self)
    }

    /// Elementwise conversion from integers, rounding to the nearest float
    /// with ties to even, like `x as f32`.
    #[inline(always)]
    pub fn from_int(input: Vector<S, T::Int>) -> Self {
        T::vfrom_int(input)
    }
}

macro_rules! impl_convert {
    ($($ty: ty: $int: ty),*) => {
        $(paste! {
            impl VConvert for $ty {
                type Int = $int;
                #[inline(always)]
                fn vtrunc_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, $int> {
                    S::typed(S::[<trunc_to_int_ $ty>](*input))
                }
                #[inline(always)]
                fn vround_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, $int> {
                    S::typed(S::[<round_to_int_ $ty>](*input))
                }
                #[inline(always)]
                fn vfrom_int<S: Simd>(input: Vector<S, $int>) -> Vector<S, Self> {
                    S::typed(S::[<from_int_ $ty>](*input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<trunc_to_int_ $ty _supported>]()
                        && S::[<round_to_int_ $ty _supported>]()
                        && S::[<from_int_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_convert!(f32: i32, f64: i64);
