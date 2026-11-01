use core::ops::Neg;

use half::f16;
use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VRecip: Scalar {
    fn vrecip<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VRecip> Vector<S, T> {
    /// Elementwise reciprocal (`1 / x`).
    ///
    /// `f32`/`f64` are accurate within a few ULP using Newton-Raphson refinement.
    /// Special values (`±0`, `±inf`, `NaN`) map accurately across all backends.
    ///
    /// Backends that flush subnormals (e.g. SSE/AVX2 `rcp_ps`) saturate subnormal inputs
    /// or subnormal results to signed `±inf`/`±0`, preserving sign without returning NaN.
    /// `f16` uses unrefined hardware estimates (~8-bit precision) for now.
    #[inline(always)]
    pub fn recip(self) -> Self {
        T::vrecip(self)
    }
}

pub trait VSqrt: Scalar {
    fn vsqrt<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VSqrt> Vector<S, T> {
    /// Elementwise square root. Correctly rounded on every SIMD backend; the
    /// scalar fallback uses the platform's `sqrt`.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        T::vsqrt(self)
    }
}

pub trait VAbs: Scalar {
    fn vabs<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VAbs> Vector<S, T> {
    /// Elementwise absolute value. For signed integers, `MIN` wraps to itself.
    #[inline(always)]
    pub fn abs(self) -> Self {
        T::vabs(self)
    }
}

pub trait VFloor: Scalar {
    fn vfloor<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VFloor> Vector<S, T> {
    /// Elementwise rounding toward negative infinity, like [`f32::floor`] on
    /// every backend: the sign of zero is kept, infinities pass through, and
    /// NaN stays NaN.
    #[inline(always)]
    pub fn floor(self) -> Self {
        T::vfloor(self)
    }
}

pub trait VCeil: Scalar {
    fn vceil<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VCeil> Vector<S, T> {
    /// Elementwise rounding toward positive infinity, like [`f32::ceil`] on
    /// every backend: the sign of zero is kept (`ceil(-0.5)` is `-0.0`),
    /// infinities pass through, and NaN stays NaN.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        T::vceil(self)
    }
}

pub trait VTrunc: Scalar {
    fn vtrunc<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VTrunc> Vector<S, T> {
    /// Elementwise rounding toward zero, like [`f32::trunc`] on every backend:
    /// the sign of zero is kept, infinities pass through, and NaN stays NaN.
    #[inline(always)]
    pub fn trunc(self) -> Self {
        T::vtrunc(self)
    }
}

pub trait VRoundTiesEven: Scalar {
    fn vroundtieseven<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VRoundTiesEven> Vector<S, T> {
    /// Elementwise rounding to the nearest integer, ties to even, like
    /// [`f32::round_ties_even`] on every backend. This is the rounding of
    /// hardware "nearest" modes, not of [`f32::round`], which rounds ties away
    /// from zero.
    #[inline(always)]
    pub fn round_ties_even(self) -> Self {
        T::vroundtieseven(self)
    }
}

pub trait VNeg: Scalar + Neg<Output = Self> {
    fn vneg<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

/// Elementwise negation, like scalar `-x` for floats and `x.wrapping_neg()`
/// for signed integers. Floats flip the sign bit, so `-0.0` and `0.0` swap and
/// NaN stays NaN with its sign flipped. Integers wrap: `-MIN` is `MIN`, where
/// scalar `-x` would panic in debug builds.
impl<S: Simd, T: VNeg> Neg for Vector<S, T> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        T::vneg(self)
    }
}

macro_rules! impl_neg {
    (float $($ty: ty: $bits: literal),*) => {
        $(paste! {
            impl VNeg for $ty {
                #[inline(always)]
                fn vneg<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::bitxor(*input, S::[<splat_i $bits>]([<i $bits>]::MIN)))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::bitxor_supported()
                }
            }
        })*
    };
    (int $($ty: ty),*) => {
        $(paste! {
            impl VNeg for $ty {
                #[inline(always)]
                fn vneg<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::[<sub_ $ty>](S::[<splat_ $ty>](0), *input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<sub_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_neg!(float f16: 16, f32: 32, f64: 64);
impl_neg!(int i8, i16, i32, i64);

macro_rules! impl_unop {
    ($trait: ident, $name: ident, $($ty: ty),*) => {
        $(paste! {
            impl $trait for $ty {
                #[inline(always)]
                fn [<$trait:lower>]<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self> {
                    S::typed(S::[<$name _ $ty>](*input))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<$name _ $ty _supported>]()
                }
            }
        })*
    };
}

impl_unop!(VRecip, recip, f16, f32, f64);
impl_unop!(VAbs, abs, i8, i16, f16, i32, i64, f32, f64);
impl_unop!(VSqrt, sqrt, f16, f32, f64);
impl_unop!(VFloor, floor, f16, f32, f64);
impl_unop!(VCeil, ceil, f16, f32, f64);
impl_unop!(VTrunc, trunc, f16, f32, f64);
impl_unop!(VRoundTiesEven, round_ties_even, f16, f32, f64);
