use half::{bf16, f16};
use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait VWiden: Scalar {
    /// The lane type twice as wide.
    type Wide: Scalar;
    fn vwiden<S: Simd>(x: Vector<S, Self>) -> [Vector<S, Self::Wide>; 2];
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VWiden> Vector<S, T> {
    /// Lanes `0..n/2` and `n/2..n`, each widened to a lane twice as wide:
    /// `[lo, hi]`. Unsigned integers are zero-extended and signed ones
    /// sign-extended. Floats widen exactly, and NaN stays NaN with an
    /// unspecified payload.
    #[inline(always)]
    pub fn widen(self) -> [Vector<S, T::Wide>; 2] {
        T::vwiden(self)
    }
}

pub trait VNarrow: Scalar {
    /// The lane type twice as wide, which narrows to this one.
    type Wide: Scalar;
    fn vnarrow<S: Simd>(lo: Vector<S, Self::Wide>, hi: Vector<S, Self::Wide>) -> Vector<S, Self>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VNarrow> Vector<S, T> {
    /// The lanes of `lo`, then those of `hi`, each narrowed to a lane half as
    /// wide, the inverse of [`widen`](Self::widen). Integers keep their low
    /// bits, as `x as u8` does. Floats round to nearest, ties to even, as
    /// `x as f32` and [`half::f16::from_f32`] do: values too large become
    /// infinity, and NaN stays NaN with an unspecified payload.
    #[inline(always)]
    pub fn narrow(lo: Vector<S, T::Wide>, hi: Vector<S, T::Wide>) -> Self {
        T::vnarrow(lo, hi)
    }
}

macro_rules! impl_widen {
    ($($ty: ty => $wide: ty),*) => {
        $(paste! {
            impl VWiden for $ty {
                type Wide = $wide;
                #[inline(always)]
                fn vwiden<S: Simd>(x: Vector<S, Self>) -> [Vector<S, $wide>; 2] {
                    let [lo, hi] = S::[<widen_ $ty>](*x);
                    [S::typed(lo), S::typed(hi)]
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<widen_ $ty _supported>]()
                }
            }
            impl VNarrow for $ty {
                type Wide = $wide;
                #[inline(always)]
                fn vnarrow<S: Simd>(lo: Vector<S, $wide>, hi: Vector<S, $wide>) -> Vector<S, Self> {
                    S::typed(S::[<narrow_ $ty>](*lo, *hi))
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<narrow_ $ty _supported>]()
                }
            }
        })*
    };
}

impl_widen!(u8 => u16, i8 => i16, u16 => u32, i16 => i32, u32 => u64, i32 => i64);
impl_widen!(f16 => f32, f32 => f64);

// A bf16 is the upper half of an f32, so it widens by moving its bits up and
// narrows by rounding off the lower half, on any backend with 16-bit lane
// widening and narrowing.
impl VWiden for bf16 {
    type Wide = f32;
    #[inline(always)]
    fn vwiden<S: Simd>(x: Vector<S, Self>) -> [Vector<S, f32>; 2] {
        let [lo, hi] = S::widen_u16(*x);
        [S::typed(S::shl_u32(lo, 16)), S::typed(S::shl_u32(hi, 16))]
    }
    #[inline(always)]
    fn is_accelerated<S: Simd>() -> bool {
        S::widen_u16_supported() && S::shl_u32_supported()
    }
}

impl VNarrow for bf16 {
    type Wide = f32;
    #[inline(always)]
    fn vnarrow<S: Simd>(lo: Vector<S, f32>, hi: Vector<S, f32>) -> Vector<S, Self> {
        S::typed(S::narrow_u16(
            round_to_bf16::<S>(*lo),
            round_to_bf16::<S>(*hi),
        ))
    }
    #[inline(always)]
    fn is_accelerated<S: Simd>() -> bool {
        S::narrow_u16_supported() && S::shr_u32_supported() && S::select_32_supported()
    }
}

/// The bits of the bf16 nearest each f32 lane, ties to even, in the low half
/// of the lane. NaN lanes keep their upper bits with the quiet bit set, so a
/// payload in the lower half alone cannot round them to infinity.
#[inline(always)]
fn round_to_bf16<S: Simd>(x: S::Register) -> S::Register {
    let odd = S::bitand(S::shr_u32(x, 16), S::splat_u32(1));
    let rounded = S::add_u32(S::add_u32(x, S::splat_u32(0x7fff)), odd);
    let quiet = S::bitor(x, S::splat_u32(0x0040_0000));
    S::shr_u32(S::select_32(S::equals_f32(x, x), rounded, quiet), 16)
}
