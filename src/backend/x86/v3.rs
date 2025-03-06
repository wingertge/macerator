use arch::*;
use bytemuck::Zeroable;
#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;
use core::ops::{Add, Div, Mul, Sub};

use half::f16;
use num_traits::Float;
use paste::paste;

use crate::{Scalar, WithSimd, backend::arch::NullaryFnOnce};

use crate::backend::{Simd, VRegister, Vector, arch::impl_simd, cast};

use super::*;

impl VRegister for __m256 {}

macro_rules! lanes {
    ($($bits: literal),*) => {
        $(paste! {
            #[inline(always)]
            fn [<lanes $bits>]() -> usize {
                256 / $bits
            }
        })*
    };
}

pub struct V3;

macro_rules! impl_binop_scalar {
    ($func: ident, $intrinsic: path, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> Self::Register {
                const LANES: usize = 32 / size_of::<$ty>();
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
                const LANES: usize = 32 / size_of::<$ty>();
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
            fn [<less_than_ $ty>](
                a: Self::Register,
                b: Self::Register,
            ) -> <$ty as Scalar>::Mask<Self> {
                Self::[<greater_than_ $ty>](a, b)
            }
            #[inline(always)]
            fn [<less_than_ $ty _supported>]() -> bool {
                Self::[<greater_than_ $ty _supported>]()
            }
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

impl Simd for V3 {
    type Register = __m256;
    type Mask8 = Vector<Self, i8>;
    type Mask16 = Vector<Self, i16>;
    type Mask32 = Vector<Self, i32>;
    type Mask64 = Vector<Self, i64>;

    lanes!(8, 16, 32, 64);

    impl_binop_signless!(
        add, _mm256_add, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
    );
    impl_binop_signless!(
        sub, _mm256_sub, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
    );
    impl_binop!(div, _mm256_div, f32, f64);
    impl_binop!(mul, _mm256_mul, f32, f64);
    impl_binop_signless!(mul, _mm256_mullo, u16, i16, u32, i32);
    impl_binop!(min, _mm256_min, u8, i8, u16, i16, u32, i32, f32, f64);
    impl_binop!(max, _mm256_max, u8, i8, u16, i16, u32, i32, f32, f64);
    impl_binop_scalar!(add, Add::add, f16);
    impl_binop_scalar!(sub, Sub::sub, f16);
    impl_binop_scalar!(div, Div::div, f16);
    impl_binop_scalar!(mul, Mul::mul, i8, u8, f16, u64, i64);
    impl_binop_scalar!(min, Ord::min, u64, i64);
    impl_binop_scalar!(min, f16::min, f16);
    impl_binop_scalar!(max, Ord::max, u64, i64);
    impl_binop_scalar!(max, f16::max, f16);

    impl_binop_untyped!(bitand, _mm256_and_si256);
    impl_binop_untyped!(bitor, _mm256_or_si256);
    impl_binop_untyped!(bitxor, _mm256_xor_si256);

    impl_unop!(recip, _mm256_rcp, f32);
    impl_unop!(abs, _mm256_abs, i8, i16, i32);
    impl_unop_scalar!(recip, recip, f16, f64);

    impl_cmp!(equals, _mm256_cmpeq, u8, i8, u16, i16, u32, i32, u64, i64);
    impl_cmp!(greater_than, _mm256_cmpgt, i8, i16, i32, i64);
    cmp_int!(u8, i8, u16, i16, u32, i32, u64, i64);

    fn vectorize<Op: WithSimd>(op: Op) -> Op::Output {
        struct Impl<Op> {
            op: Op,
        }
        impl<Op: WithSimd> NullaryFnOnce for Impl<Op> {
            type Output = Op::Output;

            #[inline(always)]
            fn call(self) -> Self::Output {
                self.op.with_simd::<V3>()
            }
        }
        Self::run_vectorized(Impl { op })
    }

    #[inline(always)]
    unsafe fn load<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        cast!(_mm256_load_si256(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_unaligned<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        cast!(_mm256_lddqu_si256(ptr as _))
    }
    #[inline(always)]
    unsafe fn load_low<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        let low = unsafe { _mm_lddqu_si128(ptr as _) };
        cast!(_mm256_castsi128_si256(low))
    }
    #[inline(always)]
    unsafe fn load_high<T: Scalar>(ptr: *const T) -> Vector<Self, T> {
        let high = unsafe { _mm_lddqu_si128((ptr as *const __m128i).add(1)) };
        cast!(_mm256_inserti128_si256::<1>(Zeroable::zeroed(), high))
    }
    #[inline(always)]
    unsafe fn store<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        unsafe { _mm256_store_si256(ptr as _, cast!(*value)) };
    }
    #[inline(always)]
    unsafe fn store_unaligned<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        unsafe { _mm256_storeu_si256(ptr as _, cast!(*value)) };
    }
    #[inline(always)]
    unsafe fn store_low<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        let low = unsafe { _mm256_castsi256_si128(cast!(*value)) };
        unsafe { _mm_storeu_si128(ptr as _, low) };
    }
    #[inline(always)]
    unsafe fn store_high<T: Scalar>(ptr: *mut T, value: Vector<Self, T>) {
        let high = unsafe { _mm256_extracti128_si256::<1>(cast!(value)) };
        unsafe { _mm_storeu_si128((ptr as *mut __m128i).add(1), high) };
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
    fn splat_i8(value: i8) -> Self::Register {
        cast!(_mm256_set1_epi8(value))
    }

    #[inline(always)]
    fn splat_i16(value: i16) -> Self::Register {
        cast!(_mm256_set1_epi16(value))
    }

    #[inline(always)]
    fn splat_i32(value: i32) -> Self::Register {
        cast!(_mm256_set1_epi32(value))
    }

    #[inline(always)]
    fn splat_i64(value: i64) -> Self::Register {
        cast!(_mm256_set1_pd(cast!(value)))
    }
    #[inline(always)]
    fn equals_f32(a: Self::Register, b: Self::Register) -> <f32 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_ps::<_CMP_EQ_UQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn equals_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn equals_f64(a: Self::Register, b: Self::Register) -> <f64 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_pd::<_CMP_EQ_UQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn equals_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_f32(a: Self::Register, b: Self::Register) -> <f32 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_ps::<_CMP_LT_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn less_than_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_f64(a: Self::Register, b: Self::Register) -> <f64 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_pd::<_CMP_LT_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn less_than_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_or_equal_f32(a: Self::Register, b: Self::Register) -> <f32 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_ps::<_CMP_LE_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn less_than_or_equal_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn less_than_or_equal_f64(a: Self::Register, b: Self::Register) -> <f64 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_pd::<_CMP_LE_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn less_than_or_equal_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_f32(a: Self::Register, b: Self::Register) -> <f32 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_ps::<_CMP_GT_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn greater_than_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_f64(a: Self::Register, b: Self::Register) -> <f64 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_pd::<_CMP_GT_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn greater_than_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_or_equal_f32(
        a: Self::Register,
        b: Self::Register,
    ) -> <f32 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_ps::<_CMP_GE_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn greater_than_or_equal_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn greater_than_or_equal_f64(
        a: Self::Register,
        b: Self::Register,
    ) -> <f64 as Scalar>::Mask<Self> {
        cast!(_mm256_cmp_pd::<_CMP_GE_OQ>(cast!(a), cast!(b)))
    }
    #[inline(always)]
    fn greater_than_or_equal_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn mul_add_f16(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        let a: [f16; 16] = cast!(a);
        let b: [f16; 16] = cast!(b);
        let c: [f16; 16] = cast!(c);
        let mut out = [f16::default(); 16];

        for i in 0..16 {
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
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    fn mul_add_f32_supported() -> bool {
        true
    }
    #[inline(always)]
    fn mul_add_f64(a: Self::Register, b: Self::Register, c: Self::Register) -> Self::Register {
        cast!(_mm256_fmadd_pd(cast!(a), cast!(b), cast!(c)))
    }
    #[inline(always)]
    fn mul_add_f64_supported() -> bool {
        true
    }
    #[inline(always)]
    fn bitnot(a: Self::Register) -> Self::Register {
        cast!(_mm256_xor_si256(cast!(a), cast!(Self::splat_i64(-1))))
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

impl V3 {
    impl_simd!(
        "sse", "sse2", "fxsr", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2",
        "bmi1", "bmi2", "fma", "lzcnt"
    );
}
