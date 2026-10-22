use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VShift: Scalar {
    fn vshl<S: Simd, const N: u32>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn vshr<S: Simd, const N: u32>(input: Vector<S, Self>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

/// Shifts by a constant, as Rust's `<<` and `>>`: right shifts are logical
/// for unsigned lanes and arithmetic for signed ones. `N` must be less than
/// the lane width, which is checked at compile time, because backends
/// disagree about larger shifts.
///
/// ```
/// # use macerator::{Simd, Vector};
/// // The biased exponent and sign of f32 values, given their bits.
/// fn exponent_bits<S: Simd>(bits: Vector<S, u32>) -> Vector<S, u32> {
///     bits.shr::<23>()
/// }
/// ```
impl<S: Simd, T: VShift> Vector<S, T> {
    /// Elementwise shift left by `N` bits, shifting in zeros.
    ///
    /// ```no_run
    /// # use macerator::{with_simd, Scalar, Simd};
    /// #[with_simd]
    /// fn top_bit<S: Simd>(x: u32) -> bool {
    ///     let mut out = [0u32; 16];
    ///     let v = x.splat::<S>().shl::<31>();
    ///     unsafe { macerator::vstore_unaligned(out.as_mut_ptr(), v) };
    ///     out[0] == 1 << 31
    /// }
    /// assert!(top_bit(1));
    /// ```
    ///
    /// A shift by the lane width or more does not compile:
    ///
    /// ```compile_fail
    /// # use macerator::{with_simd, Scalar, Simd};
    /// #[with_simd]
    /// fn top_bit<S: Simd>(x: u32) -> bool {
    ///     let mut out = [0u32; 16];
    ///     let v = x.splat::<S>().shl::<32>();
    ///     unsafe { macerator::vstore_unaligned(out.as_mut_ptr(), v) };
    ///     out[0] == 1 << 31
    /// }
    /// assert!(top_bit(1));
    /// ```
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        T::vshl::<S, N>(self)
    }

    /// Elementwise shift right by `N` bits: zeros shift in for unsigned
    /// lanes, copies of the sign bit for signed lanes.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        T::vshr::<S, N>(self)
    }
}

macro_rules! impl_shift {
    ($($ty: ty),*) => {
        $(paste! {
            impl VShift for $ty {
                #[inline(always)]
                fn vshl<S: Simd, const N: u32>(input: Vector<S, Self>) -> Vector<S, Self> {
                    const { assert!(N < <$ty>::BITS, "shift must be less than the lane width") };
                    S::typed(S::[<shl_ $ty>](*input, N))
                }
                #[inline(always)]
                fn vshr<S: Simd, const N: u32>(input: Vector<S, Self>) -> Vector<S, Self> {
                    const { assert!(N < <$ty>::BITS, "shift must be less than the lane width") };
                    S::typed(S::[<shr_ $ty>](*input, N))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<shl_ $ty _supported>]() && S::[<shr_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_shift!(u16, i16, u32, i32, u64, i64);
