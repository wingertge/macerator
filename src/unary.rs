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
    /// `f32`/`f64` are precise to within a couple ULP on every backend, since
    /// backends without an exact reciprocal refine the hardware *estimate*
    /// with Newton-Raphson steps. `±0`, `±inf` and `NaN` map to `±inf`, `±0`
    /// and `NaN` exactly, everywhere.
    ///
    /// Subnormals are the one place backends differ. A refined estimate can
    /// only reach the subnormal range if the estimate instruction itself
    /// supports it: `frecpe` (aarch64) does, `rcp_ps` (sse/avx2) does not and
    /// flushes both its input and its result. So on sse/avx2 the ends of the
    /// range saturate rather than reaching the extreme finite value: a
    /// subnormal input gives `±inf`, and an input whose reciprocal would be no
    /// larger than the smallest normal gives `±0`. Saturation always goes
    /// towards the true value and keeps the sign of the input; it never
    /// returns `NaN` or the wrong sign.
    ///
    /// `f16` is not yet refined and may only be accurate to ~8 bits on
    /// backends that compute it with a hardware estimate.
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
