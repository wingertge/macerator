use half::f16;
use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VRound: Scalar {
    fn vfloor<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn vceil<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn vtrunc<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn vround_ties_even<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

/// Rounding to an integral value. Results are exact and match the `std`
/// methods of the same names on every backend: the sign of zero is kept
/// (`ceil(-0.5)` is `-0.0`), infinities pass through, and NaN stays NaN.
impl<S: Simd, T: VRound> Vector<S, T> {
    /// Elementwise rounding toward negative infinity.
    #[inline(always)]
    pub fn floor(self) -> Self {
        T::vfloor(self)
    }

    /// Elementwise rounding toward positive infinity.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        T::vceil(self)
    }

    /// Elementwise rounding toward zero.
    #[inline(always)]
    pub fn trunc(self) -> Self {
        T::vtrunc(self)
    }

    /// Elementwise rounding to the nearest integer, ties to even, like
    /// [`f32::round_ties_even`]. This is the rounding of hardware "nearest"
    /// modes, not of [`f32::round`], which rounds ties away from zero.
    #[inline(always)]
    pub fn round_ties_even(self) -> Self {
        T::vround_ties_even(self)
    }
}

macro_rules! impl_round {
    ($($ty: ty),*) => {
        $(paste! {
            impl VRound for $ty {
                #[inline(always)]
                fn vfloor<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::[<floor_ $ty>](*input))
                }
                #[inline(always)]
                fn vceil<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::[<ceil_ $ty>](*input))
                }
                #[inline(always)]
                fn vtrunc<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::[<trunc_ $ty>](*input))
                }
                #[inline(always)]
                fn vround_ties_even<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::[<round_ties_even_ $ty>](*input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<floor_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_round!(f16, f32, f64);
