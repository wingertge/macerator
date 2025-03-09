use crate::{
    tests::{binop, test_binop},
    Simd, VOrd, Vector,
};

macro_rules! testgen_min_max {
    ($test_fn: ident, $reference: expr, $($ty: ty),*) => {
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<$test_fn _ $ty>]() {
                use num_traits::NumCast;

                let lhs = $crate::tests::random(NumCast::from(0).unwrap(), NumCast::from(127).unwrap());
                let rhs = $crate::tests::random(NumCast::from(0).unwrap(), NumCast::from(127).unwrap());
                let out_ref = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(a, b)| $ty::$reference(*a, *b))
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

#[inline(always)]
fn test_min_impl<S: Simd, T: VOrd>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    binop!(VOrd, |a: Vector<S, T>, b| a.min(b));
    test_binop::<S, T, VOrdOp<T>>(lhs, rhs)
}

#[inline(always)]
fn test_max_impl<S: Simd, T: VOrd>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    binop!(VOrd, |a: Vector<S, T>, b| a.max(b));
    test_binop::<S, T, VOrdOp<T>>(lhs, rhs)
}

testgen_min_max!(test_min, min, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
testgen_min_max!(test_max, max, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
