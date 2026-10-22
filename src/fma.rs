use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VFma: Scalar {
    fn vfma<S: Simd>(a: Vector<S, Self>, b: Vector<S, Self>, c: Vector<S, Self>)
        -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VFma> Vector<S, T> {
    /// Elementwise `self * b + c` with a single rounding on every backend,
    /// like [`f32::mul_add`]. AVX2, AVX-512, NEON, and LoongArch have an
    /// instruction for it; x86-64 v2, wasm, and the scalar fallback compute it
    /// lane by lane. Use [`mul_add`](Self::mul_add) when either rounding will
    /// do and speed matters more.
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        T::vfma(self, b, c)
    }
}

macro_rules! impl_fma {
    ($($ty: ty),*) => {
        $(paste! {
            impl VFma for $ty {
                #[inline(always)]
                fn vfma<S: Simd>(
                    a: Vector<S, Self>,
                    b: Vector<S, Self>,
                    c: Vector<S, Self>,
                ) -> Vector<S, Self> {
                    S::typed(S::[<fma_ $ty>](*a, *b, *c))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<fma_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_fma!(f32, f64);
