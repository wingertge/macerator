use half::{bf16, f16};
use paste::paste;

use crate::{Mask, Scalar, Simd, Vector};

pub trait VSelect: Scalar {
    fn vselect<S: Simd>(
        mask: Mask<S, Self>,
        if_true: Vector<S, Self>,
        if_false: Vector<S, Self>,
    ) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VSelect> Mask<S, T> {
    /// Elementwise select: lanes where the mask is set come from `if_true`,
    /// the others from `if_false`. Lanes are copied bit for bit, so NaN
    /// payloads and signed zeros pass through unchanged.
    #[inline(always)]
    pub fn select(self, if_true: Vector<S, T>, if_false: Vector<S, T>) -> Vector<S, T> {
        T::vselect(self, if_true, if_false)
    }
}

impl<S: Simd, T: Scalar> Vector<S, T> {
    /// The same register read as lanes of `U`. The bits are unchanged; when
    /// `U` has a different width, the lane count changes with it.
    ///
    /// ```
    /// # use macerator::{Scalar, Simd, Vector};
    /// fn sign_bits<S: Simd>(x: Vector<S, f32>) -> Vector<S, u32> {
    ///     x.reinterpret::<u32>() & (1u32 << 31).splat()
    /// }
    /// ```
    #[inline(always)]
    pub fn reinterpret<U: Scalar>(self) -> Vector<S, U> {
        S::typed(*self)
    }
}

macro_rules! impl_select {
    ($bits: literal, $($ty: ty),*) => {
        $(paste! {
            impl VSelect for $ty {
                #[inline(always)]
                fn vselect<S: Simd>(
                    mask: Mask<S, Self>,
                    if_true: Vector<S, Self>,
                    if_false: Vector<S, Self>,
                ) -> Vector<S, Self> {
                    S::typed(S::[<select_ $bits>](mask.0, *if_true, *if_false))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<select_ $bits _supported>]()
                }
            }
        })*
    };
}

impl_select!(8, u8, i8);
impl_select!(16, u16, i16, f16, bf16);
impl_select!(32, u32, i32, f32);
impl_select!(64, u64, i64, f64);
