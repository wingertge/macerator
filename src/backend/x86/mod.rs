#![allow(
    clippy::missing_transmute_annotations,
    clippy::useless_transmute,
    clippy::transmute_int_to_float,
    unused_unsafe
)]

pub mod v2;
pub mod v3;
#[cfg(avx512)]
pub mod v4;

pub use v2::V2;
pub use v3::V3;
#[cfg(avx512)]
pub use v4::V4;
#[cfg(avx512_fp16)]
pub use v4::V4FP16;

macro_rules! lanes {
    ($($bits: literal),*) => {
        $(paste! {
            #[inline(always)]
            fn [<lanes $bits>]() -> usize {
                WIDTH / $bits
            }
        })*
    };
}
pub(crate) use lanes;

macro_rules! impl_binop_scalar {
    ($func: ident, $intrinsic: path, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> Self::Register {
                const LANES: usize = WIDTH / (8 * size_of::<$ty>());
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
pub(crate) use impl_binop_scalar;

macro_rules! impl_unop_scalar {
    ($func: ident, $intrinsic: path, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register) -> Self::Register {
                const LANES: usize = WIDTH / (8 * size_of::<$ty>());
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
pub(crate) use impl_unop_scalar;

macro_rules! impl_reduce_scalar {
    ($func: ident, $intrinsic: path, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register) -> $ty {
                const LANES: usize = WIDTH / (8 * size_of::<$ty>());
                let a: [$ty; LANES] = cast!(a);
                let mut out: $ty = a[0];

                for i in 1..LANES {
                    out = out.$intrinsic(a[i]);
                }
                out
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                false
            }
        })*
    };
}
pub(crate) use impl_reduce_scalar;

macro_rules! with_ty {
    ($func: ident, f16) => {
        paste!([<$func _ph>])
    };
    ($func: ident, f32) => {
        paste!([<$func _ps>])
    };
    ($func: ident, f64) => {
        paste!([<$func _pd>])
    };
    ($func: ident, $ty: ident) => {
        paste!([<$func _ep $ty>])
    }
}
pub(crate) use with_ty;

macro_rules! with_ty_signless {
    ($func: ident, u8) => {
        with_ty!($func, i8)
    };
    ($func: ident, u16) => {
        with_ty!($func, i16)
    };
    ($func: ident, u32) => {
        with_ty!($func, i32)
    };
    ($func: ident, u64) => {
        with_ty!($func, i64)
    };
    ($func: ident, $ty: ident) => {
        with_ty!($func, $ty)
    };
}
pub(crate) use with_ty_signless;

macro_rules! impl_binop_signless {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> Self::Register {
                cast!(with_ty_signless!($intrinsic, $ty)(cast!(a), cast!(b)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}
pub(crate) use impl_binop_signless;

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
pub(crate) use impl_binop;

macro_rules! impl_binop_untyped {
    ($func: ident, $intrinsic: ident) => {
        paste! {
            #[inline(always)]
            fn $func(a: Self::Register, b: Self::Register) -> Self::Register {
                cast!($intrinsic(cast!(a), cast!(b)))
            }
            #[inline(always)]
            fn [<$func _supported>]() -> bool {
                true
            }
        }
    };
}
pub(crate) use impl_binop_untyped;

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
pub(crate) use impl_unop;

#[allow(unused)]
macro_rules! impl_reduce {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register) -> $ty {
                unsafe { cast!(with_ty!($intrinsic, $ty)(cast!(a))) }
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}
#[allow(unused)]
pub(crate) use impl_reduce;

#[allow(unused)]
macro_rules! impl_reduce_signless {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register) -> $ty {
                cast!(with_ty_signless!($intrinsic, $ty)(cast!(a)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}
#[allow(unused)]
pub(crate) use impl_reduce_signless;

macro_rules! impl_cmp {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, b: Self::Register) -> <$ty as Scalar>::Mask<Self> {
                cast!(with_ty_signless!($intrinsic, $ty)(cast!(a), cast!(b)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}
pub(crate) use impl_cmp;

/// Rounding with an immediate mode. SSE4.1 `round` and AVX-512 `roundscale`
/// share the encoding: bits 1:0 select the mode, bit 3 suppresses the inexact
/// exception, and the `roundscale` scale in bits 7:4 is zero.
macro_rules! impl_round {
    ($intrinsic: ident, $($ty: ident: $suffix: ident),*) => {
        $(paste! {
            #[inline(always)]
            fn [<floor_ $ty>](a: Self::Register) -> Self::Register {
                cast!([<$intrinsic _ $suffix>]::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(cast!(a)))
            }
            #[inline(always)]
            fn [<floor_ $ty _supported>]() -> bool {
                true
            }
            #[inline(always)]
            fn [<ceil_ $ty>](a: Self::Register) -> Self::Register {
                cast!([<$intrinsic _ $suffix>]::<{ _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC }>(cast!(a)))
            }
            #[inline(always)]
            fn [<ceil_ $ty _supported>]() -> bool {
                true
            }
            #[inline(always)]
            fn [<trunc_ $ty>](a: Self::Register) -> Self::Register {
                cast!([<$intrinsic _ $suffix>]::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(cast!(a)))
            }
            #[inline(always)]
            fn [<trunc_ $ty _supported>]() -> bool {
                true
            }
            #[inline(always)]
            fn [<round_ties_even_ $ty>](a: Self::Register) -> Self::Register {
                cast!([<$intrinsic _ $suffix>]::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(cast!(a)))
            }
            #[inline(always)]
            fn [<round_ties_even_ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}

/// Shifts every lane by a count taken from the low 64 bits of an `__m128i`,
/// which is how `sll`, `srl`, and `sra` take it at every width.
macro_rules! impl_shift {
    ($func: ident, $intrinsic: ident, $($ty: ty),*) => {
        $(paste! {
            #[inline(always)]
            fn [<$func _ $ty>](a: Self::Register, n: u32) -> Self::Register {
                cast!(with_ty_signless!($intrinsic, $ty)(cast!(a), _mm_cvtsi32_si128(n as i32)))
            }
            #[inline(always)]
            fn [<$func _ $ty _supported>]() -> bool {
                true
            }
        })*
    };
}
pub(crate) use impl_round;

/// Select for masks held as vectors of all-ones or all-zero lanes: a byte
/// blend gives the same result for every lane width.
macro_rules! impl_select_blendv {
    ($intrinsic: ident, $($bits: literal),*) => {
        $(paste! {
            #[inline(always)]
            fn [<select_ $bits>](
                mask: Self::[<Mask $bits>],
                if_true: Self::Register,
                if_false: Self::Register,
            ) -> Self::Register {
                cast!($intrinsic(cast!(if_false), cast!(if_true), cast!(mask)))
            }
            #[inline(always)]
            fn [<select_ $bits _supported>]() -> bool {
                true
            }
        })*
    };
}
pub(crate) use impl_select_blendv;

/// Any and all for masks held as vectors of all-ones or all-zero lanes: every
/// byte of a set lane is set, so the byte mask answers for every lane width.
macro_rules! impl_mask_reduce_movemask {
    ($movemask: ident, $all: expr, $($bits: literal),*) => {
        $(paste! {
            #[inline(always)]
            fn [<mask_any_ $bits>](mask: Self::[<Mask $bits>]) -> bool {
                unsafe { $movemask(cast!(mask)) != 0 }
            }
            #[inline(always)]
            fn [<mask_any_ $bits _supported>]() -> bool {
                true
            }
            #[inline(always)]
            fn [<mask_all_ $bits>](mask: Self::[<Mask $bits>]) -> bool {
                unsafe { $movemask(cast!(mask)) == $all }
            }
            #[inline(always)]
            fn [<mask_all_ $bits _supported>]() -> bool {
                true
            }
        })*
    };
}
pub(crate) use impl_mask_reduce_movemask;

pub(crate) use impl_shift;
