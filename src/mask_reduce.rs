use half::{bf16, f16};
use paste::paste;

use crate::{Mask, Scalar, Simd};

pub trait VMaskReduce: Scalar {
    fn vany<S: Simd>(mask: Mask<S, Self>) -> bool;
    fn vall<S: Simd>(mask: Mask<S, Self>) -> bool;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VMaskReduce> Mask<S, T> {
    /// Whether any lane is set: a loop over vectors can stop early on it
    /// without storing the mask as booleans.
    #[inline(always)]
    pub fn any(&self) -> bool {
        T::vany(Mask(self.0))
    }

    /// Whether every lane is set.
    #[inline(always)]
    pub fn all(&self) -> bool {
        T::vall(Mask(self.0))
    }
}

macro_rules! impl_mask_reduce {
    ($bits: literal, $($ty: ty),*) => {
        $(paste! {
            impl VMaskReduce for $ty {
                #[inline(always)]
                fn vany<S: Simd>(mask: Mask<S, Self>) -> bool {
                    S::[<mask_any_ $bits>](mask.0)
                }
                #[inline(always)]
                fn vall<S: Simd>(mask: Mask<S, Self>) -> bool {
                    S::[<mask_all_ $bits>](mask.0)
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<mask_any_ $bits _supported>]()
                }
            }
        })*
    };
}

impl_mask_reduce!(8, u8, i8);
impl_mask_reduce!(16, u16, i16, f16, bf16);
impl_mask_reduce!(32, u32, i32, f32);
impl_mask_reduce!(64, u64, i64, f64);
