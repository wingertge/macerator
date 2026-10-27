use std::{vec, vec::Vec};

use half::{bf16, f16};
use rand::Rng;

use crate::{
    tests::for_each_backend, vload_unaligned, vstore_unaligned, Mask, Scalar, Simd, VOrd, VSelect,
};

/// Random bit patterns, so float lanes include NaN payloads, infinities,
/// subnormals, and both zeros.
fn random_bits<T: Scalar>(len: usize) -> Vec<T> {
    let mut out = vec![T::zeroed(); len];
    rand::rng().fill(bytemuck::cast_slice_mut::<T, u8>(&mut out));
    out
}

fn random_bools(len: usize) -> Vec<bool> {
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random()).collect()
}

#[inline(always)]
fn test_select_impl<S: Simd, T: VSelect>((a, b, bools): (&[T], &[T], &[bool])) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::zeroed(); a.len()];
    for (i, out) in out.chunks_exact_mut(lanes).enumerate() {
        let range = i * lanes..(i + 1) * lanes;
        let a = unsafe { vload_unaligned::<S, T>(a[range.clone()].as_ptr()) };
        let b = unsafe { vload_unaligned::<S, T>(b[range.clone()].as_ptr()) };
        let mask = Mask::<S, T>::from_bools(&bools[range]);
        unsafe { vstore_unaligned(out.as_mut_ptr(), mask.select(a, b)) };
    }
    out
}

macro_rules! testgen_select {
    ($($(#[$meta:meta])* $ty: ty),*) => {
        $(::paste::paste! {
            $(#[$meta])*
            // Miri does not implement LoongArch `vbitsel.v`.
            #[cfg_attr(all(miri, loong64), ignore)]
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_select_ $ty>]() {
                let a = random_bits::<$ty>(crate::tests::SIZE);
                let b = random_bits::<$ty>(crate::tests::SIZE);
                let bools = random_bools(crate::tests::SIZE);
                let want: Vec<$ty> = (0..a.len())
                    .map(|i| if bools[i] { a[i] } else { b[i] })
                    .collect();
                for_each_backend!(
                    test_select_impl,
                    $ty,
                    (&a[..], &b[..], &bools[..]),
                    |out: &[$ty]| assert_eq!(
                        bytemuck::cast_slice::<$ty, u8>(&want),
                        bytemuck::cast_slice::<$ty, u8>(out)
                    )
                );
            }
        })*
    };
}

testgen_select!(u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64);

/// Masks built from comparisons and combined with `!`, `&`, and `|`, which
/// some backends store differently from `from_bools`.
#[inline(always)]
fn test_select_combined_impl<S: Simd, T: VSelect + VOrd>((x, y): (&[T], &[T])) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::zeroed(); x.len()];
    for (i, out) in out.chunks_exact_mut(lanes).enumerate() {
        let range = i * lanes..(i + 1) * lanes;
        let x = unsafe { vload_unaligned::<S, T>(x[range.clone()].as_ptr()) };
        let y = unsafe { vload_unaligned::<S, T>(y[range].as_ptr()) };
        let mask = !x.lt(y) & x.ne(y) | x.eq(T::default().splat());
        unsafe { vstore_unaligned(out.as_mut_ptr(), mask.select(x, y)) };
    }
    out
}

macro_rules! testgen_select_combined {
    ($($(#[$meta:meta])* $ty: ty),*) => {
        $(::paste::paste! {
            $(#[$meta])*
            // Miri does not implement LoongArch `vbitsel.v`.
            #[cfg_attr(all(miri, loong64), ignore)]
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_select_combined_ $ty>]() {
                use num_traits::NumCast;

                let mut x = crate::tests::random::<$ty>(
                    NumCast::from(0).unwrap(),
                    NumCast::from(8).unwrap(),
                );
                let y = crate::tests::random::<$ty>(
                    NumCast::from(0).unwrap(),
                    NumCast::from(8).unwrap(),
                );
                // Some zeros, for the `|` branch.
                for v in x.iter_mut().step_by(5) {
                    *v = <$ty>::default();
                }
                let want: Vec<$ty> = x
                    .iter()
                    .zip(&y)
                    .map(|(&x, &y)| {
                        // Without NaN, `!(x < y) & (x != y)` is `x > y`.
                        let set = x > y || x == <$ty>::default();
                        if set { x } else { y }
                    })
                    .collect();
                for_each_backend!(
                    test_select_combined_impl,
                    $ty,
                    (&x[..], &y[..]),
                    |out: &[$ty]| assert_eq!(&want[..], out)
                );
            }
        })*
    };
}

testgen_select_combined!(
    u8,
    i8,
    u16,
    i16,
    // Generating f16 inputs uses F16C or FP16 conversions Miri lacks.
    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
    f16,
    u32,
    i32,
    // Miri does not implement AVX-512 float compares.
    #[cfg_attr(all(miri, x86_v4), ignore)]
    f32,
    u64,
    i64,
    #[cfg_attr(all(miri, x86_v4), ignore)]
    f64
);

// `for_each_backend` passes a lane type these two do not need.
#[allow(clippy::extra_unused_type_parameters)]
#[inline(always)]
fn test_reinterpret_impl<S: Simd, T: Scalar>(a: &[f32]) -> Vec<u32> {
    let lanes = f32::lanes::<S>();
    let mut out = vec![0u32; a.len()];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let a = unsafe { vload_unaligned::<S, f32>(a.as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), a.reinterpret::<u32>()) };
    }
    out
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_reinterpret_f32_u32() {
    let a = random_bits::<f32>(crate::tests::SIZE);
    let want: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    for_each_backend!(test_reinterpret_impl, f32, &a[..], |out: &[u32]| {
        assert_eq!(&want[..], out)
    });
}

/// A reinterpret to a narrower lane keeps the bytes and multiplies the lanes.
#[allow(clippy::extra_unused_type_parameters)]
#[inline(always)]
fn test_reinterpret_bytes_impl<S: Simd, T: Scalar>(a: &[u64]) -> Vec<u8> {
    let lanes = u64::lanes::<S>();
    let mut out = vec![0u8; a.len() * 8];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes * 8)) {
        let a = unsafe { vload_unaligned::<S, u64>(a.as_ptr()) };
        let bytes = a.reinterpret::<u8>();
        assert_eq!(u8::lanes::<S>(), lanes * 8);
        unsafe { vstore_unaligned(out.as_mut_ptr(), bytes) };
    }
    out
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_reinterpret_u64_u8() {
    let a = random_bits::<u64>(crate::tests::SIZE);
    let want: Vec<u8> = bytemuck::cast_slice(&a).to_vec();
    for_each_backend!(test_reinterpret_bytes_impl, u64, &a[..], |out: &[u8]| {
        assert_eq!(&want[..], out)
    });
}
