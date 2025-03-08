use bytemuck::Zeroable;
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    Rng,
};

use crate::{vload_unaligned, vstore_unaligned, Scalar, Simd, Vector};

mod arithmetic;
mod bitwise;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

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
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<$test_fn _ $ty>]() {
                use num_traits::NumCast;

                let lhs = $crate::tests::random(NumCast::from(8).unwrap(), NumCast::from(16).unwrap());
                let rhs = $crate::tests::random(NumCast::from(0).unwrap(), NumCast::from(8).unwrap());
                let out_ref = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(a, b)| $reference(a, b))
                    .collect::<Vec<_>>();
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    use $crate::backend::x86::{v2::V2, v3::V3};
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
                    use $crate::backend::aarch64::NeonFma;
                    if NeonFma::is_available() {
                        let out = NeonFma::run_vectorized(|| [<$test_fn _impl>]::<NeonFma, $ty>(&lhs, &rhs));
                        assert_eq!(out_ref, out);
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    use crate::backend::wasm32::Simd128;
                    let out = Simd128::run_vectorized(|| [<$test_fn _impl>]::<Simd128, $ty>(&lhs, &rhs));
                    assert_eq!(&out_ref, &out);
                }
                let out = [<$test_fn _impl>]::<$crate::backend::scalar::Fallback, $ty>(&lhs, &rhs);
                assert_eq!(out_ref, out);
            }
        })*
    };
}
pub(crate) use testgen_binop;

pub(crate) trait Binop<T: Scalar> {
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
        ::paste::paste! {
            struct [<$trait Op>]<T>(::core::marker::PhantomData<T>);
            impl<T: $trait> Binop<T> for [<$trait Op>]<T> {
                #[inline(always)]
                fn call<S: Simd>(lhs: Vector<S, T>, rhs: Vector<S, T>) -> Vector<S, T> {
                    $impl(lhs, rhs)
                }
            }
        }
    };
}
pub(crate) use binop;
