#![allow(
    clippy::missing_transmute_annotations,
    clippy::useless_transmute,
    clippy::transmute_int_to_float,
    unused_unsafe
)]

pub mod v2;
pub mod v3;

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
