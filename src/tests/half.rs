use core::fmt::Debug;
use std::{vec, vec::Vec};

use crate::{
    vload_high, vload_low, vload_unaligned, vstore_high, vstore_low, vstore_unaligned, Scalar, Simd,
};

use super::for_each_backend;

/// Runs `vload_low`, `vload_high`, `vstore_low` and `vstore_high` on `src`.
/// Stores go to `buf[1..]`, so like `src[1..]` from the caller they are only
/// aligned to one element, which the half operations allow.
#[inline(always)]
fn test_half_ops_impl<S: Simd, T: Scalar>(src: &[T]) -> Vec<Vec<T>> {
    let lanes = T::lanes::<S>();
    let mut loaded = [vec![T::default(); lanes], vec![T::default(); lanes]];
    let mut stored = [vec![T::default(); lanes + 1], vec![T::default(); lanes + 1]];
    unsafe {
        vstore_unaligned(loaded[0].as_mut_ptr(), vload_low::<S, T>(src.as_ptr()));
        vstore_unaligned(loaded[1].as_mut_ptr(), vload_high::<S, T>(src.as_ptr()));
        let full = vload_unaligned::<S, T>(src.as_ptr());
        vstore_low(stored[0].as_mut_ptr().add(1), full);
        vstore_high(stored[1].as_mut_ptr().add(1), full);
    }
    let [low, high] = loaded;
    let [store_low, store_high] = stored.map(|s| s[1..].to_vec());
    vec![low, high, store_low, store_high]
}

/// A half lives at the same position in memory as in the register, with `ptr`
/// addressing the whole vector.
fn check_half_ops<T: Scalar + PartialEq + Debug>(src: &[T], out: &[Vec<T>]) {
    let [low, high, store_low, store_high] = out else {
        unreachable!()
    };
    let (lanes, zero) = (low.len(), T::default());
    let half = lanes / 2;
    assert_eq!(low[..half], src[..half], "vload_low");
    assert!(
        low[half..].iter().all(|x| *x == zero),
        "vload_low upper half"
    );
    assert_eq!(high[half..], src[half..lanes], "vload_high");
    assert!(
        high[..half].iter().all(|x| *x == zero),
        "vload_high lower half"
    );
    assert_eq!(store_low[..half], src[..half], "vstore_low");
    assert!(
        store_low[half..].iter().all(|x| *x == zero),
        "vstore_low wrote past its half"
    );
    assert_eq!(store_high[half..], src[half..lanes], "vstore_high");
    assert!(
        store_high[..half].iter().all(|x| *x == zero),
        "vstore_high wrote outside its half"
    );
}

macro_rules! testgen_half_ops {
    ($($ty: ty),*) => {
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_half_ops_ $ty>]() {
                let src: Vec<$ty> = (1..=65).map(|i| i as $ty).collect();
                // One element in, so 8-byte halves are not 8-byte aligned.
                let src = &src[1..];
                for_each_backend!(test_half_ops_impl, $ty, src, |out: &[Vec<$ty>]| {
                    check_half_ops(src, out)
                });
            }
        })*
    };
}

testgen_half_ops!(u8, u16, u32);
