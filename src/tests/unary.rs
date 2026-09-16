use core::fmt::Debug;
use half::f16;
use std::vec::Vec;

use num_traits::{Float, NumCast};

use crate::{
    assert_relative_eq,
    tests::{approx::RelativeEq, assert_eq, test_unop, unop},
    Simd, VAbs, VRecip, Vector,
};

use super::{testgen_unop, Unop};

#[inline(always)]
fn test_recip_impl<S: Simd, T: VRecip>(a: &[T]) -> Vec<T> {
    unop!(VRecip, |a: Vector<S, T>| a.recip());
    test_unop::<S, T, VRecipOp<T>>(a)
}

#[inline(always)]
fn test_abs_impl<S: Simd, T: VAbs>(a: &[T]) -> Vec<T> {
    unop!(VAbs, |a: Vector<S, T>| a.abs());
    test_unop::<S, T, VAbsOp<T>>(a)
}

fn assert_approx_eq_recip<T: RelativeEq<Epsilon = T> + Debug + NumCast + Copy>(
    lhs: &[T],
    rhs: &[T],
) {
    let epsilon = match core::mem::size_of::<T>() {
        // `f16`: unrefined hardware estimate on some backends, only ~8 bits accurate.
        ..=2 => T::from(2.0.powf(-8.0)).unwrap(),
        // `f32`: couple ULP, i.e. a small multiple of `f32::EPSILON` (2^-23).
        4 => T::from(4.0 * f32::EPSILON as f64).unwrap(),
        // `f64`: couple ULP, i.e. a small multiple of `f64::EPSILON` (2^-52).
        _ => T::from(4.0 * f64::EPSILON).unwrap(),
    };
    for (a, b) in lhs.iter().zip(rhs) {
        assert_relative_eq!(*a, *b, epsilon = epsilon);
    }
}

testgen_unop!(
    test_recip,
    recip,
    1,
    100,
    assert_approx_eq_recip,
    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
    f16,
    #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
    f32,
    #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
    f64
);
testgen_unop!(
    test_abs,
    abs,
    -100,
    100,
    assert_eq,
    i8,
    i16,
    i32,
    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
    f16,
    f32,
    f64
);
