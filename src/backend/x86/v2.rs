use arch::*;
#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;
use core::ops::{Add, Div, Mul, Sub};

use half::f16;
use num_traits::Float;
use paste::paste;

use crate::{backend::arch::NullaryFnOnce, Scalar, WithSimd};

use crate::backend::{arch::impl_simd, cast, seal::Sealed, Simd, VRegister, Vector};

use super::*;

impl VRegister for __m128 {}

pub struct V2;

impl Sealed for V2 {}

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

macro_rules! impl_unop_scalar {
    ($func: ident, $intrinsic: path, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register) -> Self::Register {
                const LANES: usize = 16 / size_of::<$ty>();
                let a: [$ty; LANES] = cast!(a);
                let mut out = [$ty::default(); LANES];

                for i in 0..LANES {
                    out[i] = a[i].$intrinsic();
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

macro_rules! cmp_int {
    ($($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<less_than_or_equal_ $ty>](
                a: Self::Register,
                b: Self::Register,
            ) -> <$ty as Scalar>::Mask<Self> {
                let gt = Self::[<greater_than_ $ty>](a, b);
                cast!(Self::bitnot(cast!(gt)))
            }
            #[inline(always)]
            fn [<less_than_or_equal_ $ty _supported>]() -> bool {
                Self::[<greater_than_ $ty _supported>]()
            }
            #[inline(always)]
            fn [<greater_than_or_equal_ $ty>](
                a: Self::Register,
                b: Self::Register,
            ) -> <$ty as Scalar>::Mask<Self> {
                let gt = Self::[<less_than_ $ty>](a, b);
                cast!(Self::bitnot(cast!(gt)))
            }
            #[inline(always)]
            fn [<greater_than_or_equal_ $ty _supported>]() -> bool {
                Self::[<less_than_ $ty _supported>]()
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

impl Simd for V2 {
    type Register = __m128;
    type Mask8 = Vector<Self, i8>;
    type Mask16 = Vector<Self, i16>;
    type Mask32 = Vector<Self, i32>;
    type Mask64 = Vector<Self, i64>;

    lanes!(8, 16, 32, 64);

    impl_binop_signless!(add, _mm_add, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
    impl_binop_signless!(sub, _mm_sub, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
    impl_binop!(div, _mm_div, f32, f64);
    impl_binop_signless!(mul, _mm_mul, f32, f64);
    impl_binop_signless!(mul, _mm_mullo, u16, i16, u32, i32);
    impl_binop!(min, _mm_min, u8, i8, u16, i16, u32, i32, f32, f64);
    impl_binop!(max, _mm_max, u8, i8, u16, i16, u32, i32, f32, f64);
    impl_binop_scalar!(add, Add::add, f16);
    impl_binop_scalar!(sub, Sub::sub, f16);
    impl_binop_scalar!(div, Div::div, f16);
    impl_binop_scalar!(mul, Mul::mul, i8, u8, f16, u64, i64);
    impl_binop_scalar!(min, Ord::min, u64, i64);
    impl_binop_scalar!(min, f16::min, f16);
    impl_binop_scalar!(max, Ord::max, u64, i64);
    impl_binop_scalar!(max, f16::max, f16);

    impl_binop_untyped!(bitand, _mm_and_si128);
    impl_binop_untyped!(bitor, _mm_or_si128);
    impl_binop_untyped!(bitxor, _mm_xor_si128);

    impl_unop!(recip, _mm_rcp, f32);
    impl_unop!(abs, _mm_abs, i8, i16, i32);
    impl_unop_scalar!(recip, recip, f16, f64);

    impl_cmp!(equals, _mm_cmpeq, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
    impl_cmp!(greater_than, _mm_cmpgt, i8, i16, i32, f32, i64, f64);
    impl_cmp!(greater_than_or_equal, _mm_cmpge, f32, f64);
    impl_cmp!(less_than, _mm_cmplt, i8, i16, i32, f32, f64);
    impl_cmp!(less_than_or_equal, _mm_cmple, f32, f64);

    cmp_int!(u8, i8, u16, i16, u32, i32, u64, i64);

    fn vectorize<Op: WithSimd>(op: Op) -> Op::Output {
        struct Impl<Op> {
            op: Op,
        }
        impl<Op: WithSimd> NullaryFnOnce for Impl<Op> {
            type Output = Op::Output;

            #[inline(always)]
            fn call(self) -> Self::Output {
                self.op.with_simd::<V2>()
            }
        }
        Self::run_vectorized(Impl { op })
    }

    #[inline(always)]
    unsafe fn mask_store_as_bool_8(out: *mut bool, mask: Self::Mask8) {
        let bools = Self::bitand(cast!(mask), Self::splat_i8(1));
        Self::store_unaligned(out as *mut u8, cast!(bools));
    }
    #[inline(always)]
    unsafe fn mask_store_as_bool_16(out: *mut bool, mask: Self::Mask16) {
        const LANES: usize = 128 / 16;
        let mask: [i16; LANES] = cast!(mask);
        for i in 0..LANES {
            *out.add(i) = mask[i] != 0;
        }
    }
    #[inline(always)]
    unsafe fn mask_store_as_bool_32(out: *mut bool, mask: Self::Mask32) {
        const LANES: usize = 128 / 32;
        let mask: [i32; LANES] = cast!(mask);
        for i in 0..LANES {
            *out.add(i) = mask[i] != 0;
        }
    }
    #[inline(always)]
    unsafe fn mask_store_as_bool_64(out: *mut bool, mask: Self::Mask64) {
        const LANES: usize = 128 / 64;
        let mask: [i64; LANES] = cast!(mask);
        for i in 0..LANES {
            *out.add(i) = mask[i] != 0;
        }
    }
    #[inline(always)]
    fn mask_from_bools_8(bools: &[bool]) -> Self::Mask8 {
        debug_assert_eq!(bools.len(), Self::lanes8());
        const LANES: usize = 128 / 8;
        let mut out = [0i8; LANES];
        for i in 0..LANES {
            out[i] = if bools[i] { -1 } else { 0 };
        }
        cast!(out)
    }
    #[inline(always)]
    fn mask_from_bools_16(bools: &[bool]) -> Self::Mask16 {
        debug_assert_eq!(bools.len(), Self::lanes16());
        const LANES: usize = 128 / 16;
        let mut out = [0i16; LANES];
        for i in 0..LANES {
            out[i] = if bools[i] { -1 } else { 0 };
        }
        cast!(out)
    }
    #[inline(always)]
    fn mask_from_bools_32(bools: &[bool]) -> Self::Mask32 {
        debug_assert_eq!(bools.len(), Self::lanes32());
        const LANES: usize = 128 / 32;
        let mut out = [0i32; LANES];
        for i in 0..LANES {
            out[i] = if bools[i] { -1 } else { 0 };
        }
        cast!(out)
    }
    #[inline(always)]
    fn mask_from_bools_64(bools: &[bool]) -> Self::Mask64 {
        debug_assert_eq!(bools.len(), Self::lanes64());
        const LANES: usize = 128 / 64;
        let mut out = [0i64; LANES];
        for i in 0..LANES {
            out[i] = if bools[i] { -1 } else { 0 };
        }
        cast!(out)
    }

    #[inline(always)]
    unsafe fn load<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        cast!(_mm_load_si128(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_unaligned<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        cast!(_mm_lddqu_si128(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_low<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        cast!(_mm_loadl_epi64(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_high<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        cast!(_mm_loadh_pd(cast!(Self::splat_f64(0.0)), ptr as _))
    }
    #[inline(always)]
    unsafe fn store<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        unsafe { _mm_store_si128(ptr as _, cast!(value)) };
    }
    #[inline(always)]
    unsafe fn store_unaligned<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        unsafe { _mm_storeu_si128(ptr as _, cast!(value)) };
    }
    #[inline(always)]
    unsafe fn store_low<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        unsafe {
            _mm_storel_epi64(ptr as _, cast!(value));
        };
    }
    #[inline(always)]
    unsafe fn store_high<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        unsafe { _mm_storeh_pd(ptr as _, cast!(value)) };
    }

    #[inline(always)]
    fn splat_i8(value: i8) -> Self::Register {
        cast!(_mm_set1_epi8(value))
    }

    #[inline(always)]
    fn splat_i16(value: i16) -> Self::Register {
        cast!(_mm_set1_epi16(value))
    }

    #[inline(always)]
    fn splat_i32(value: i32) -> Self::Register {
        cast!(_mm_set1_epi32(value))
    }

    #[inline(always)]
    fn splat_i64(value: i64) -> Self::Register {
        cast!(_mm_set1_pd(cast!(value)))
    }

    #[inline(always)]
    fn less_than_u8(a: Self::Register, b: Self::Register) -> <u8 as Scalar>::Mask<Self> {
        let bias = Self::splat_i8(i8::MIN);
        let a = Self::sub_u8(a, bias);
        let b = Self::sub_u8(b, bias);
        Self::less_than_i8(a, b)
    }
    #[inline(always)]
    fn less_than_u8_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_u16(a: Self::Register, b: Self::Register) -> <u16 as Scalar>::Mask<Self> {
        let bias = Self::splat_i16(i16::MIN);
        let a = Self::sub_u16(a, bias);
        let b = Self::sub_u16(b, bias);
        Self::less_than_i16(a, b)
    }
    #[inline(always)]
    fn less_than_u16_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_u32(a: Self::Register, b: Self::Register) -> <u32 as Scalar>::Mask<Self> {
        let bias = Self::splat_i32(i32::MIN);
        let a = Self::sub_u32(a, bias);
        let b = Self::sub_u32(b, bias);
        Self::less_than_i32(a, b)
    }
    #[inline(always)]
    fn less_than_u32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_u64(a: Self::Register, b: Self::Register) -> <u64 as Scalar>::Mask<Self> {
        let bias = Self::splat_i64(i64::MIN);
        let a = Self::sub_u64(a, bias);
        let b = Self::sub_u64(b, bias);
        Self::less_than_i64(a, b)
    }
    #[inline(always)]
    fn less_than_u64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_u8(a: Self::Register, b: Self::Register) -> <u8 as Scalar>::Mask<Self> {
        let bias = Self::splat_i8(i8::MIN);
        let a = Self::sub_u8(a, bias);
        let b = Self::sub_u8(b, bias);
        Self::greater_than_i8(a, b)
    }
    #[inline(always)]
    fn greater_than_u8_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_u16(a: Self::Register, b: Self::Register) -> <u16 as Scalar>::Mask<Self> {
        let bias = Self::splat_i16(i16::MIN);
        let a = Self::sub_u16(a, bias);
        let b = Self::sub_u16(b, bias);
        Self::greater_than_i16(a, b)
    }
    #[inline(always)]
    fn greater_than_u16_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_u32(a: Self::Register, b: Self::Register) -> <u32 as Scalar>::Mask<Self> {
        let bias = Self::splat_i32(i32::MIN);
        let a = Self::sub_u32(a, bias);
        let b = Self::sub_u32(b, bias);
        Self::greater_than_i32(a, b)
    }
    #[inline(always)]
    fn greater_than_u32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_u64(a: Self::Register, b: Self::Register) -> <u64 as Scalar>::Mask<Self> {
        let bias = Self::splat_i64(i64::MIN);
        let a = Self::sub_u64(a, bias);
        let b = Self::sub_u64(b, bias);
        Self::greater_than_i64(a, b)
    }
    #[inline(always)]
    fn greater_than_u64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_i64(a: Self::Register, b: Self::Register) -> <i64 as Scalar>::Mask<Self> {
        Self::greater_than_i64(b, a)
    }
    #[inline(always)]
    fn less_than_i64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn mul_add_f16(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        let a: [f16; 8] = cast!(a);
        let b: [f16; 8] = cast!(b);
        let c: [f16; 8] = cast!(c);
        let mut out = [f16::default(); 8];

        for i in 0..8 {
            out[i] = a[i] * b[i] + c[i];
        }
        cast!(out)
    }
    #[inline(always)]
    fn mul_add_f16_supported() -> bool {
        false
    }
    #[inline(always)]
    fn mul_add_f32(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        unsafe { _mm_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    fn mul_add_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn mul_add_f64(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        cast!(_mm_fmadd_pd(cast!(a), cast!(b), cast!(c)))
    }
    #[inline(always)]
    fn mul_add_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn bitnot(a: Self::Register) -> Self::Register {
        cast!(_mm_xor_si128(cast!(a), cast!(Self::splat_i64(-1))))
    }
    #[inline(always)]
    fn bitnot_supported() -> bool {
        true
    }
    #[inline(always)]
    fn abs_f16(a: Self::Register) -> Self::Register {
        let mask = Self::splat_i16(i16::MAX);
        Self::bitand(a, mask)
    }
    #[inline(always)]
    fn abs_f16_supported() -> bool {
        true
    }
    #[inline(always)]
    fn abs_f32(a: Self::Register) -> Self::Register {
        let mask = Self::splat_i32(i32::MAX);
        Self::bitand(a, mask)
    }
    #[inline(always)]
    fn abs_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn abs_f64(a: Self::Register) -> Self::Register {
        let mask = Self::splat_i64(i64::MAX);
        Self::bitand(a, mask)
    }
    #[inline(always)]
    fn abs_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn abs_i64(a: Self::Register) -> Self::Register {
        let mask = Self::splat_i64(i64::MAX);
        Self::bitand(a, mask)
    }
    #[inline(always)]
    fn abs_i64_supported() -> bool {
        true
    }
}

impl V2 {
    impl_simd!("sse", "sse2", "fxsr", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt");
}
