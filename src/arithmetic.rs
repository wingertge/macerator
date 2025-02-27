use core::ops::{Add, Div, Mul, Sub};

use paste::paste;
use pulp::Simd;

use crate::Vectorizable;

pub trait VAdd: Vectorizable + Add<Output = Self> {
    fn vadd<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}

pub trait VSub: Vectorizable + Sub<Output = Self> {
    fn vsub<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}

pub trait VMul: Vectorizable + Mul<Output = Self> {
    fn vmul<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}

pub trait VDiv: Vectorizable + Div<Output = Self> {
    fn vdiv<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S>;
}

pub trait VMulAdd: Vectorizable + Div<Output = Self> + Add<Output = Self> {
    fn vmuladd<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
}

macro_rules! impl_arith {
    ($trait: ident, $name: ident, $ty: ty) => {
        paste! {
            impl $trait for $ty {
                fn [<$trait:lower>]<S: Simd>(simd: S, lhs: Self::Vector<S>, rhs: Self::Vector<S>) -> Self::Vector<S> {
                    simd.[<$name _ $ty s>](lhs, rhs)
                }
            }
        }
    };
    ($trait: ident, $name: ident, $($ty: ty),*) => {
        $(impl_arith!($trait, $name, $ty);)*
    }
}

impl_arith!(VAdd, add, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_arith!(VSub, sub, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_arith!(VMul, mul, u16, i16, u32, i32, f32, f64);
impl_arith!(VDiv, div, f32, f64);

macro_rules! vmuladd_impl {
    (intrinsic $($ty: ty),*) => {
        $(paste! {
            impl VMulAdd for $ty {
                fn vmuladd<S: Simd>(
                    simd: S,
                    a: Self::Vector<S>,
                    b: Self::Vector<S>,
                    c: Self::Vector<S>,
                ) -> Self::Vector<S> {
                    simd.[<mul_add_ $ty s>](a, b, c)
                }
            }
        })*
    };
    (fallback $($ty: ty),*) => {
        $(paste! {
            impl VMulAdd for $ty {
                fn vmuladd<S: Simd>(
                    simd: S,
                    a: Self::Vector<S>,
                    b: Self::Vector<S>,
                    c: Self::Vector<S>,
                ) -> Self::Vector<S> {
                    let mul = simd.[<mul_ $ty s>](a, b);
                    simd.[<add_ $ty s>](mul, c)
                }
            }
        })*
    };
}

vmuladd_impl!(intrinsic f32, f64);
vmuladd_impl!(fallback u16, i16, u32, i32, u64, i64);
