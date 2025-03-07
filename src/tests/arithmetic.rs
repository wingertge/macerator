use crate::scalar::Fallback;

use core::{fmt::Debug, marker::PhantomData};

use approx::{assert_relative_eq, RelativeEq};
use bytemuck::Zeroable;
use half::f16;
use num_traits::NumCast;
use paste::paste;
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    Rng,
};

use crate::{
    vload_unaligned, vstore_unaligned, Scalar, Simd, VAdd, VDiv, VMul, VMulAdd, VSub, Vector,
};

trait Binop<T: Scalar> {
    fn call<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T>;
}

#[inline(always)]
fn test_binop<S: Simd, T: Scalar, Op: Binop<T>>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut output = vec![Zeroable::zeroed(); lhs.len()];
    let lhs = lhs.chunks_exact(lanes);
    let rhs = rhs.chunks_exact(lanes);
    let out = output.chunks_exact_mut(lanes);
    for ((lhs, rhs), out) in lhs.zip(rhs).zip(out) {
        let lhs = unsafe { vload_unaligned(lhs.as_ptr()) };
        let rhs = unsafe { vload_unaligned(rhs.as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), Op::call::<S>(lhs, rhs)) };
    }
    output
}

macro_rules! binop {
    ($trait: ident, $impl: expr) => {
        paste! {
            struct [<$trait Op>]<T>(PhantomData<T>);
            impl<T: $trait> Binop<T> for [<$trait Op>]<T> {
                #[inline(always)]
                fn call<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
                    $impl(lhs, rhs)
                }
            }
        }
    };
}

#[inline(always)]
fn test_add_impl<S: Simd, T: VAdd>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    binop!(VAdd, |a, b| a + b);
    test_binop::<S, T, VAddOp<T>>(lhs, rhs)
}
#[inline(always)]
fn test_sub_impl<S: Simd, T: VSub>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    binop!(VSub, |a, b| a - b);
    test_binop::<S, T, VSubOp<T>>(lhs, rhs)
}
#[inline(always)]
fn test_mul_impl<S: Simd, T: VMul>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    binop!(VMul, |a, b| a * b);
    test_binop::<S, T, VMulOp<T>>(lhs, rhs)
}
#[inline(always)]
fn test_div_impl<S: Simd, T: VDiv>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    binop!(VDiv, |a, b| a / b);
    test_binop::<S, T, VDivOp<T>>(lhs, rhs)
}
#[inline(always)]
fn test_fma_impl<S: Simd, T: VMulAdd>(a: &[T], b: &[T], c: &[T]) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let a = a.chunks_exact(lanes);
    let b = b.chunks_exact(lanes);
    let c = c.chunks_exact(lanes);
    let mut output = vec![Zeroable::zeroed(); a.len()];
    let out = output.chunks_exact_mut(lanes);
    for (((a, b), c), out) in a.zip(b).zip(c).zip(out) {
        let a = unsafe { vload_unaligned::<S, _>(a.as_ptr()) };
        let b = unsafe { vload_unaligned(b.as_ptr()) };
        let c = unsafe { vload_unaligned(c.as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), a.mul_add(b, c)) };
    }
    output
}

const SIZE: usize = 1024;
fn random<T: SampleUniform>(lo: T, hi: T) -> Vec<T> {
    let distribution = Uniform::new(lo, hi);
    rand::thread_rng()
        .sample_iter(&distribution)
        .take(SIZE)
        .collect()
}

macro_rules! testgen_binop {
    ($test_fn: ident, $reference: expr, $($ty: ty),*) => {
        $(paste! {
            #[test]
            fn [<$test_fn _ $ty>]() {
                let lhs = random(NumCast::from(8).unwrap(), NumCast::from(16).unwrap());
                let rhs = random(NumCast::from(0).unwrap(), NumCast::from(8).unwrap());
                let out_ref = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(a, b)| $reference(a, b))
                    .collect::<Vec<_>>();
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    use crate::backend::x86::{v2::V2, v3::V3};
                    if V3::is_available() {
                        let out = V3::run_vectorized(|| [<$test_fn _impl>]::<V3, $ty>(&lhs, &rhs));
                        assert_eq!(out_ref, out);
                    }
                    if V2::is_available() {
                        let out = V2::run_vectorized(|| [<$test_fn _impl>]::<V2, $ty>(&lhs, &rhs));
                        assert_eq!(out_ref, out);
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    use crate::backend::aarch64::NeonFma;
                    if NeonFma::is_available() {
                        let out = NeonFma::run_vectorized(|| [<$test_fn _impl>]::<NeonFma, $ty>(&lhs, &rhs));
                        assert_eq!(out_ref, out);
                    }
                }
                let out = [<$test_fn _impl>]::<Fallback, $ty>(&lhs, &rhs);
                assert_eq!(out_ref, out);
            }
        })*
    };
}

testgen_binop!(
    test_add,
    |a, b| a + b,
    u8,
    i8,
    u16,
    i16,
    f16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64
);

testgen_binop!(
    test_sub,
    |a, b| a - b,
    u8,
    i8,
    u16,
    i16,
    f16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64
);
testgen_binop!(test_div, |a, b| a / b, f16, f32, f64);
testgen_binop!(
    test_mul,
    |a, b| a * b,
    u8,
    i8,
    u16,
    i16,
    f16,
    u32,
    i32,
    f32,
    f64
);

fn assert_approx_eq<T: RelativeEq + Debug>(lhs: &[T], rhs: &[T]) {
    for (a, b) in lhs.iter().zip(rhs) {
        assert_relative_eq!(*a, *b);
    }
}

macro_rules! testgen_fma {
    ($test_fn: ident, $reference: expr, $($ty: ty),*) => {
        $(paste! {
            #[test]
            fn [<$test_fn _ $ty>]() {
                let a = random(NumCast::from(0).unwrap(), NumCast::from(8).unwrap());
                let b = random(NumCast::from(0).unwrap(), NumCast::from(8).unwrap());
                let c = random(NumCast::from(0).unwrap(), NumCast::from(64).unwrap());
                let out_ref = a
                    .iter()
                    .zip(b.iter()).zip(c.iter())
                    .map(|((a, b), c)| a * b + c)
                    .collect::<Vec<_>>();
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    use crate::backend::x86::{v2::V2, v3::V3};
                    if V3::is_available() {
                        let out = V3::run_vectorized(|| [<$test_fn _impl>]::<V3, $ty>(&a, &b, &c));
                        assert_approx_eq(&out_ref, &out);
                    }
                    if V2::is_available() {
                        let out = V2::run_vectorized(|| [<$test_fn _impl>]::<V2, $ty>(&a, &b, &c));
                        assert_approx_eq(&out_ref, &out);
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    use crate::backend::aarch64::NeonFma;
                    if NeonFma::is_available() {
                        let out = NeonFma::run_vectorized(|| [<$test_fn _impl>]::<NeonFma, $ty>(&a, &b, &c));
                        assert_approx_eq(&out_ref, &out);
                    }
                }
                let out = [<$test_fn _impl>]::<Fallback, $ty>(&a, &b, &c);
                assert_approx_eq(&out_ref, &out);
            }
        })*
    };
}

testgen_fma!(test_fma, f16, f32, f64);
