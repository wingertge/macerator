use std::{vec, vec::Vec};

use crate::{backend::scalar::Fallback, Scalar, Simd, Vector};

use super::for_each_backend;

/// Loads a vector from `src` and stores it into a slice one element longer.
#[inline(always)]
fn test_load_store_impl<S: Simd, T: Scalar>(src: &[T]) -> Vec<T> {
    let mut out = vec![T::default(); T::lanes::<S>() + 1];
    Vector::<S, T>::load(src).store(&mut out);
    out
}

macro_rules! testgen_load_store {
    ($($ty: ty),*) => {
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_load_store_ $ty>]() {
                // Longer than any vector, and offset so it is not aligned.
                let src: Vec<$ty> = (1..=66).map(|i| i as $ty).collect();
                let src = &src[1..];
                for_each_backend!(test_load_store_impl, $ty, src, |out: &[$ty]| {
                    let lanes = out.len() - 1;
                    assert_eq!(out[..lanes], src[..lanes]);
                    assert_eq!(out[lanes], 0 as $ty, "stored past the vector");
                });
            }
        })*
    };
}

testgen_load_store!(u8, i16, u32, f32, u64, f64);

#[test]
#[should_panic(expected = "slice shorter than a vector")]
fn test_load_short_slice_panics() {
    let src = vec![0u32; u32::lanes::<Fallback>() - 1];
    Vector::<Fallback, u32>::load(&src);
}

#[test]
#[should_panic(expected = "slice shorter than a vector")]
fn test_store_short_slice_panics() {
    let mut dst = vec![0u32; u32::lanes::<Fallback>() - 1];
    1u32.splat::<Fallback>().store(&mut dst);
}
