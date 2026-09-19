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

pub trait VAbs: Scalar {
    fn vabs<S: Simd>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VAbs> Vector<S, T> {
    #[inline(always)]
    pub fn abs(self) -> Self {
        T::vabs(self)
    }
}

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
impl_unop!(VAbs, abs, i8, i16, f16, i32, f32, f64);
