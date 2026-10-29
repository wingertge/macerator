use core::ops::{Shl, Shr};

use paste::paste;

use crate::{backend::seal::Sealed, Scalar, Simd, Vector};

/// The count of a shift of `T` lanes: a `u32` shifts every lane by the same
/// count, and a vector of `T` shifts each lane by the matching lane.
pub trait ShiftRhs<T: Scalar>: Sealed {
    /// The count as a shift of `Vector<S, T>` takes it.
    type Count<S: Simd>: Copy;
}

impl<T: Scalar> ShiftRhs<T> for u32 {
    type Count<S: Simd> = u32;
}

impl<S0: Simd, T: Scalar> ShiftRhs<T> for Vector<S0, T> {
    type Count<S: Simd> = Vector<S, T>;
}

macro_rules! impl_shift {
    ($(#[$attr: meta])* $trait: ident, $std_trait: ident, $name: ident, $($ty: ty),*) => {
        paste!{
            $(#[$attr])*
            pub trait $trait<Rhs: ShiftRhs<Self> = u32>: Scalar + $std_trait<u32, Output = Self> {
                fn [<$trait:lower>]<S: Simd>(lhs: Vector<S, Self>, n: Rhs::Count<S>) -> Vector<S, Self>;
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
            }
            impl<S0: Simd> $trait<Vector<S0, $ty>> for $ty {
                #[inline(always)]
                fn [<$trait:lower>]<S: Simd>(lhs: Vector<S, Self>, n: Vector<S, Self>) -> Vector<S, Self> {
                    debug_assert!(
                        bytemuck::cast_slice::<_, $ty>(core::slice::from_ref(&*n))
                            .iter()
                            .all(|&n| (n as u64) < <$ty>::BITS as u64),
                        "shift must be less than the lane width"
                    );
                    S::typed(S::[<$name _lanes_ $ty>](*lhs, *n))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<$name _lanes_ $ty _supported>]()
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
            impl<S: Simd, T: $trait<Self>> $std_trait<Self> for Vector<S, T> {
                type Output = Self;

                #[inline(always)]
                fn $name(self, n: Self) -> Self::Output {
                    T::[<$trait:lower>](self, n)
                }
            }
            impl<S: Simd, T: $trait<Self>> ::core::ops::[<$std_trait Assign>]<Self> for Vector<S, T> {
                #[inline(always)]
                fn [<$name _assign>](&mut self, n: Self) {
                    *self = T::[<$trait:lower>](*self, n)
                }
            }
        }
    };
}

impl_shift!(
    /// Elementwise `<<`, shifting in zeros. `Rhs` is the count: a `u32`, the
    /// default, shifts every lane by the same count, and a `Vector<S, Self>`
    /// shifts each lane by the matching lane. Like the scalar operator, a
    /// count of the lane width or more, or a negative one, panics with debug
    /// assertions on and is taken modulo the lane width otherwise.
    ///
    /// ```
    /// # use macerator::{Simd, Vector};
    /// // Each lane of `x` shifted by its own count in `n`.
    /// fn shift_each<S: Simd>(x: Vector<S, u32>, n: Vector<S, u32>) -> Vector<S, u32> {
    ///     x << n
    /// }
    /// ```
    VShl, Shl, shl, u8, i8, u16, i16, u32, i32, u64, i64
);
impl_shift!(
    /// Elementwise `>>`: logical for unsigned lanes, arithmetic for signed
    /// ones. `Rhs` is the count, a `u32` for every lane or a vector of a count
    /// per lane, as for [`VShl`]. Like the scalar operator, a count of the lane
    /// width or more, or a negative one, panics with debug assertions on and is
    /// taken modulo the lane width otherwise.
    ///
    /// ```
    /// # use macerator::{Simd, Vector};
    /// // The biased exponent and sign of f32 values, given their bits.
    /// fn exponent_bits<S: Simd>(bits: Vector<S, u32>) -> Vector<S, u32> {
    ///     bits >> 23
    /// }
    /// ```
    VShr, Shr, shr, u8, i8, u16, i16, u32, i32, u64, i64
);
