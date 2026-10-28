use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VTruncToInt: Scalar {
    /// The signed integer of the same width.
    type Int: Scalar;
    fn vtrunc_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self::Int>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VTruncToInt> Vector<S, T> {
    /// Elementwise conversion to the signed integer of the same width,
    /// rounding toward zero like `x as i32` on every backend: out-of-range
    /// values saturate to the integer's minimum or maximum, and NaN becomes 0.
    #[inline(always)]
    pub fn trunc_to_int(self) -> Vector<S, T::Int> {
        T::vtrunc_to_int(self)
    }
}

pub trait VRoundToInt: VTruncToInt {
    fn vround_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self::Int>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VRoundToInt> Vector<S, T> {
    /// Elementwise conversion to the signed integer of the same width,
    /// rounding to the nearest with ties to even, like
    /// `x.round_ties_even() as i32` on every backend: saturating, and NaN
    /// becomes 0. One instruction where the hardware has it, rather than
    /// [`round_ties_even`](Self::round_ties_even) then
    /// [`trunc_to_int`](Self::trunc_to_int).
    #[inline(always)]
    pub fn round_to_int(self) -> Vector<S, T::Int> {
        T::vround_to_int(self)
    }
}

pub trait VToFloat: Scalar {
    /// The float of the same width.
    type Float: Scalar;
    fn vto_float<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self::Float>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VToFloat> Vector<S, T> {
    /// Elementwise conversion to the float of the same width, rounding to the
    /// nearest with ties to even, like `x as f32` on every backend.
    #[inline(always)]
    pub fn to_float(self) -> Vector<S, T::Float> {
        T::vto_float(self)
    }
}

macro_rules! impl_convert {
    ($($ty: ty: $int: ty),*) => {
        $(paste! {
            impl VTruncToInt for $ty {
                type Int = $int;
                #[inline(always)]
                fn vtrunc_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, $int> {
                    S::typed(S::[<trunc_to_int_ $ty>](*input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<trunc_to_int_ $ty _supported>]()
                }
            }
            impl VRoundToInt for $ty {
                #[inline(always)]
                fn vround_to_int<S: Simd>(input: Vector<S, Self>) -> Vector<S, $int> {
                    S::typed(S::[<round_to_int_ $ty>](*input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<round_to_int_ $ty _supported>]()
                }
            }
            impl VToFloat for $int {
                type Float = $ty;
                #[inline(always)]
                fn vto_float<S: Simd>(input: Vector<S, Self>) -> Vector<S, $ty> {
                    S::typed(S::[<from_int_ $ty>](*input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<from_int_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_convert!(f32: i32, f64: i64);
