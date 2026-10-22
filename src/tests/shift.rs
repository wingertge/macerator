use std::{vec, vec::Vec};

use rand::Rng;

use crate::{tests::for_each_backend, vload_unaligned, vstore_unaligned, Simd, VShift, Vector};

#[inline(always)]
fn apply<S: Simd, T: VShift>(a: &[T], op: impl Fn(Vector<S, T>) -> Vector<S, T>) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::default(); a.len()];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let a = unsafe { vload_unaligned::<S, T>(a.as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), op(a)) };
    }
    out
}

/// Each count shifts left and then right, results concatenated in that order.
macro_rules! testgen_shift {
    ($($ty: ty: [$($n: literal),*]),*) => {
        $(::paste::paste! {
            #[inline(always)]
            fn [<test_shift_ $ty _impl>]<S: Simd, T: VShift>(a: &[T]) -> Vec<T> {
                let mut out = Vec::new();
                $(
                    out.extend(apply::<S, T>(a, |v| v.shl::<$n>()));
                    out.extend(apply::<S, T>(a, |v| v.shr::<$n>()));
                )*
                out
            }

            // Miri implements neither the NEON register-count shifts (`sshl`, `ushl`)
            // nor the AVX-512 ones.
            #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_shift_ $ty>]() {
                let mut rng = rand::rng();
                // Edges for the sign fill, then random bits; whole vectors only.
                let mut a: Vec<$ty> = vec![0, 1, <$ty>::MAX, <$ty>::MIN, <$ty>::MAX / 2 + 1];
                #[allow(unused_comparisons)]
                if <$ty>::MIN < 0 {
                    a.push((0 as $ty).wrapping_sub(1));
                }
                while a.len() < crate::tests::SIZE {
                    a.push(rng.random());
                }
                let mut want = Vec::new();
                $(
                    want.extend(a.iter().map(|x| x << $n));
                    want.extend(a.iter().map(|x| x >> $n));
                )*
                for_each_backend!([<test_shift_ $ty _impl>], $ty, &a[..], |out: &[$ty]| {
                    assert_eq!(&want[..], out)
                });
            }
        })*
    };
}

testgen_shift!(
    u16: [0, 1, 5, 8, 15],
    i16: [0, 1, 5, 8, 15],
    u32: [0, 1, 7, 23, 31],
    i32: [0, 1, 7, 23, 31],
    u64: [0, 1, 13, 32, 52, 63],
    i64: [0, 1, 13, 32, 52, 63]
);
