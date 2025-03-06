use core::{
    arch::aarch64::*,
    ops::{Add, Div, Mul, Sub},
};

use half::f16;
use num_traits::real::Real;
use paste::paste;

use crate::{Scalar, backend::arch::NullaryFnOnce, cast};

use super::{Simd, VRegister, Vector, WithSimd, arch::impl_simd};

impl VRegister for int8x16_t {}

pub struct NeonFma;

macro_rules! with_ty {
    ($func: ident, i8) => {
        paste!([<$func _s8>])
    };
    ($func: ident, i16) => {
        paste!([<$func _s16>])
    };
    ($func: ident, i32) => {
        paste!([<$func _s32>])
    };
    ($func: ident, i64) => {
        paste!([<$func _s64>])
    };
    ($func: ident, $ty: ident) => {
        paste!([<$func _ $ty>])
    }
}

macro_rules! impl_binop {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> Self::Register {
                cast!(with_ty!($intrinsic, $ty)(cast!(a), cast!(b)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}

macro_rules! impl_cmp {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> <$ty as Scalar>::Mask<Self> {
                cast!(with_ty!($intrinsic, $ty)(cast!(a), cast!(b)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}

macro_rules! impl_unop {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register) -> Self::Register {
                cast!(with_ty!($intrinsic, $ty)(cast!(a)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}

macro_rules! impl_binop_scalar {
    ($func: ident, $intrinsic: path, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> Self::Register {
                const LANES: usize = 16 / size_of::<$ty>();
                let a: [$ty; LANES] = cast!(a);
                let b: [$ty; LANES] = cast!(b);
                let mut out = [$ty::default(); LANES];

                for i in 0..LANES {
                    out[i] = $intrinsic(a[i], b[i]);
                }
                cast!(out)
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                false
            }
        })*
    };
}

macro_rules! lanes {
    ($($bits: literal),*) => {
        $(paste! {
            #[inline(always)]
            fn [<lanes $bits>]() -> usize {
                128 / $bits
            }
        })*
    };
}

impl Simd for NeonFma {
    type Register = int8x16_t;
    type Mask8 = Vector<Self, i8>;
    type Mask16 = Vector<Self, i16>;
    type Mask32 = Vector<Self, i32>;
    type Mask64 = Vector<Self, i64>;

    lanes!(8, 16, 32, 64);

    impl_binop!(add, vaddq, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
    impl_binop!(sub, vsubq, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
    impl_binop!(mul, vmulq, u8, i8, u16, i16, u32, i32, f32, f64);
    impl_binop!(div, vdivq, f32, f64);
    impl_binop!(min, vminq, u8, i8, u16, i16, u32, i32, f32, f64);
    impl_binop!(max, vmaxq, u8, i8, u16, i16, u32, i32, f32, f64);

    impl_unop!(recip, vrecpeq, f32, f64);
    impl_unop!(abs, vabsq, i8, i16, i32, i64, f32, f64);

    impl_cmp!(
        equals, vceqq, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
    );
    impl_cmp!(
        less_than, vcltq, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
    );
    impl_cmp!(
        less_than_or_equal,
        vcleq,
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        f32,
        u64,
        i64,
        f64
    );
    impl_cmp!(
        greater_than,
        vcgtq,
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        f32,
        u64,
        i64,
        f64
    );
    impl_cmp!(
        greater_than_or_equal,
        vcgeq,
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        f32,
        u64,
        i64,
        f64
    );

    impl_binop_scalar!(add, Add::add, f16);
    impl_binop_scalar!(sub, Sub::sub, f16);
    impl_binop_scalar!(mul, Mul::mul, f16, u64, i64);
    impl_binop_scalar!(div, Div::div, f16);
    impl_binop_scalar!(min, Ord::min, u64, i64);
    impl_binop_scalar!(min, f16::min, f16);
    impl_binop_scalar!(max, Ord::max, u64, i64);
    impl_binop_scalar!(max, f16::max, f16);

    fn vectorize<Op: WithSimd>(op: Op) -> Op::Output {
        struct Impl<Op> {
            op: Op,
        }
        impl<Op: WithSimd> NullaryFnOnce for Impl<Op> {
            type Output = Op::Output;

            #[inline(always)]
            fn call(self) -> Self::Output {
                self.op.with_simd::<NeonFma>()
            }
        }
        Self::run_vectorized(Impl { op })
    }

    #[inline(always)]
    unsafe fn load<T: Scalar>(ptr: *const T) -> super::Vector<Self, T> {
        cast!(vld1q_s8(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_unaligned<T: Scalar>(ptr: *const T) -> super::Vector<Self, T> {
        cast!(vld1q_s8(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_low<T: Scalar>(ptr: *const T) -> super::Vector<Self, T> {
        cast!(vld1q_lane_s64::<0>(ptr as _, cast!(Self::splat_i64(0))))
    }
    #[inline(always)]
    unsafe fn load_high<T: Scalar>(ptr: *const T) -> super::Vector<Self, T> {
        cast!(vld1q_lane_s64::<1>(
            (ptr as *const i64).add(i64::lanes::<Self>() / 2),
            cast!(Self::splat_i64(0))
        ))
    }
    #[inline(always)]
    unsafe fn store<T: Scalar>(ptr: *mut T, value: super::Vector<Self, T>) {
        unsafe { vst1q_s8(ptr as _, cast!(value)) };
    }
    #[inline(always)]
    unsafe fn store_unaligned<T: Scalar>(ptr: *mut T, value: super::Vector<Self, T>) {
        unsafe { vst1q_s8(ptr as _, cast!(value)) };
    }
    #[inline(always)]
    unsafe fn store_low<T: Scalar>(ptr: *mut T, value: super::Vector<Self, T>) {
        unsafe { vst1q_lane_s64::<0>(ptr as _, cast!(value)) };
    }
    #[inline(always)]
    unsafe fn store_high<T: Scalar>(ptr: *mut T, value: super::Vector<Self, T>) {
        unsafe {
            vst1q_lane_s64::<1>(
                (ptr as *mut i64).add(i64::lanes::<Self>() / 2),
                cast!(value),
            )
        };
    }
    #[inline(always)]
    fn splat_i8(value: i8) -> Self::Register {
        cast!(vdupq_n_s8(value))
    }
    #[inline(always)]
    fn splat_i16(value: i16) -> Self::Register {
        cast!(vdupq_n_s16(value))
    }
    #[inline(always)]
    fn splat_i32(value: i32) -> Self::Register {
        cast!(vdupq_n_s32(value))
    }
    #[inline(always)]
    fn splat_i64(value: i64) -> Self::Register {
        cast!(vdupq_n_s64(value))
    }
    #[inline(always)]
    fn bitand(a: Self::Register, b: Self::Register) -> Self::Register {
        cast!(vandq_s8(a, b))
    }
    #[inline(always)]
    fn bitand_supported() -> bool {
        true
    }
    #[inline(always)]
    fn bitor(a: Self::Register, b: Self::Register) -> Self::Register {
        cast!(vorrq_s8(a, b))
    }
    #[inline(always)]
    fn bitor_supported() -> bool {
        true
    }
    #[inline(always)]
    fn bitxor(a: Self::Register, b: Self::Register) -> Self::Register {
        cast!(veorq_s8(a, b))
    }
    #[inline(always)]
    fn bitxor_supported() -> bool {
        true
    }
    #[inline(always)]
    fn bitnot(a: Self::Register) -> Self::Register {
        Self::bitxor(a, Self::splat_i64(-1))
    }
    #[inline(always)]
    fn bitnot_supported() -> bool {
        true
    }
    #[inline(always)]
    fn mul_add_f16(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        let a: [f16; 8] = cast!(a);
        let b: [f16; 8] = cast!(b);
        let c: [f16; 8] = cast!(c);
        let mut out = [f16::default(); 8];

        for i in 0..8 {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
        cast!(out)
    }
    #[inline(always)]
    fn mul_add_f16_supported() -> bool {
        false
    }
    #[inline(always)]
    fn mul_add_f32(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        cast!(vfmaq_f32(cast!(a), cast!(b), cast!(c)))
    }
    #[inline(always)]
    fn mul_add_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn mul_add_f64(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        cast!(vfmaq_f64(cast!(a), cast!(b), cast!(c)))
    }
    #[inline(always)]
    fn mul_add_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn recip_f16(a: Self::Register) -> Self::Register {
        let a: [f16; 8] = cast!(a);
        let mut out = [f16::default(); 8];

        for i in 0..8 {
            out[i] = a[i].recip();
        }
        cast!(out)
    }
    #[inline(always)]
    fn recip_f16_supported() -> bool {
        false
    }
    #[inline(always)]
    fn abs_f16(a: Self::Register) -> Self::Register {
        let a: [f16; 8] = cast!(a);
        let mut out = [f16::default(); 8];

        for i in 0..8 {
            out[i] = a[i].abs();
        }
        cast!(out)
    }
    #[inline(always)]
    fn abs_f16_supported() -> bool {
        false
    }
}

impl NeonFma {
    impl_simd!("neon");
}
