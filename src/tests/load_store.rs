use std::{vec, vec::Vec};

use half::{bf16, f16};

use crate::{backend::scalar::Fallback, vload_partial, vstore_partial, Scalar, Simd, Vector};

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

/// A value whose bytes are all `b`. For `b` up to 0x7f this is never NaN, so
/// float lanes compare like the integers.
fn repeat_byte<T: Scalar>(b: u8) -> T {
    bytemuck::pod_read_unaligned(&[b; 8][..size_of::<T>()])
}

/// For each `n` from 0 to one past a whole vector: the lanes `vload_partial`
/// gives from a slice of exactly `n` elements, then what `vstore_partial` of a
/// whole vector writes into a slice of sentinels. Both pointers are one element
/// past the start of their allocation, so not aligned for the vector.
#[inline(always)]
fn test_partial_impl<S: Simd, T: Scalar>((src, sentinel): (&[T], T)) -> Vec<Vec<T>> {
    let lanes = T::lanes::<S>();
    let full = Vector::<S, T>::load(src);
    let mut out = Vec::new();
    for n in 0..=lanes + 1 {
        // Only the `min(n, lanes)` elements the contract asks for, so Miri
        // catches a read past them, including a missing clamp when `n > lanes`.
        let part = src[..=n.min(lanes)].to_vec();
        let mut loaded = vec![T::default(); lanes];
        unsafe { vload_partial::<S, T>(part[1..].as_ptr(), n) }.store(&mut loaded);
        let mut stored = vec![sentinel; lanes + 3];
        unsafe { vstore_partial(stored[1..].as_mut_ptr(), full, n) };
        out.push(loaded);
        out.push(stored);
    }
    out
}

macro_rules! testgen_partial {
    ($($ty: ty),*) => {
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_partial_ $ty>]() {
                let src: Vec<$ty> = (1..=66).map(repeat_byte).collect();
                let sentinel = repeat_byte::<$ty>(99);
                for_each_backend!(test_partial_impl, $ty, (&src[..], sentinel), |out: &[Vec<$ty>]| {
                    let lanes = out[0].len();
                    for (n, pair) in out.chunks_exact(2).enumerate() {
                        let k = n.min(lanes);
                        let (loaded, stored) = (&pair[0], &pair[1]);
                        assert_eq!(loaded[..k], src[1..=k], "vload_partial n={n}");
                        assert!(loaded[k..].iter().all(|&x| x == <$ty>::default()), "vload_partial n={n} upper lanes");
                        assert_eq!(stored[0], sentinel, "vstore_partial n={n} wrote before the pointer");
                        assert_eq!(stored[1..=k], src[..k], "vstore_partial n={n}");
                        assert!(stored[k + 1..].iter().all(|&x| x == sentinel), "vstore_partial n={n} wrote past {k}");
                    }
                });
            }
        })*
    };
}

testgen_partial!(u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64);

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
