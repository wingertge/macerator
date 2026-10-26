use core::ops::{Shl, Shr};

use paste::paste;

use crate::{Scalar, Simd, Vector};

macro_rules! impl_shift {
    ($(#[$attr: meta])* $trait: ident, $std_trait: ident, $name: ident, $($ty: ty),*) => {
        paste!{
            $(#[$attr])*
            pub trait $trait: Scalar + $std_trait<u32, Output = Self> {
                fn [<$trait:lower>]<S: Simd>(lhs: Vector<S, Self>, n: u32) -> Vector<S, Self>;
                fn is_accelerated<S: Simd>() -> bool;
            }
            $(impl $trait for $ty {
                #[inline(always)]
                fn [<$trait:lower>]<S: Simd>(lhs: Vector<S, Self>, n: u32) -> Vector<S, Self> {
                    debug_assert!(n < <$ty>::BITS, "shift must be less than the lane width");
                    // Backends disagree about larger counts, so wrap as release builds of `<<` do.
                    S::typed(S::[<$name _ $ty>](*lhs, n & (<$ty>::BITS - 1)))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<$name _ $ty _supported>]()
                }
            })*
            impl<S: Simd, T: $trait> $std_trait<u32> for Vector<S, T> {
                type Output = Self;

                #[inline(always)]
                fn $name(self, n: u32) -> Self::Output {
                    T::[<$trait:lower>](self, n)
                }
            }
            impl<S: Simd, T: $trait> ::core::ops::[<$std_trait Assign>]<u32> for Vector<S, T> {
                #[inline(always)]
                fn [<$name _assign>](&mut self, n: u32) {
                    *self = T::[<$trait:lower>](*self, n)
                }
            }
        }
    };
}

impl_shift!(
    /// Elementwise `<<` by `n` bits, shifting in zeros. Like the scalar
    /// operator, a shift by the lane width or more panics with debug
    /// assertions on and is taken modulo the lane width otherwise.
    VShl, Shl, shl, u16, i16, u32, i32, u64, i64
);
impl_shift!(
    /// Elementwise `>>` by `n` bits: logical for unsigned lanes, arithmetic
    /// for signed ones. Like the scalar operator, a shift by the lane width or
    /// more panics with debug assertions on and is taken modulo the lane width
    /// otherwise.
    ///
    /// ```
    /// # use macerator::{Simd, Vector};
    /// // The biased exponent and sign of f32 values, given their bits.
    /// fn exponent_bits<S: Simd>(bits: Vector<S, u32>) -> Vector<S, u32> {
    ///     bits >> 23
    /// }
    /// ```
    VShr, Shr, shr, u16, i16, u32, i32, u64, i64
);
