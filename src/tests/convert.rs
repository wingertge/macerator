use std::{vec, vec::Vec};

use rand::Rng;

use crate::{
    tests::for_each_backend, vload_unaligned, vstore_unaligned, Simd, VRoundToInt, VToFloat, Vector,
};

#[inline(always)]
fn apply<S: Simd, A: crate::Scalar, B: crate::Scalar>(
    a: &[A],
    op: impl Fn(Vector<S, A>) -> Vector<S, B>,
) -> Vec<B> {
    let lanes = A::lanes::<S>();
    let mut out = vec![B::default(); a.len()];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let a = unsafe { vload_unaligned::<S, A>(a.as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), op(a)) };
    }
    out
}

/// Truncated results, then rounded ones.
#[inline(always)]
fn test_to_int_impl<S: Simd, T: VRoundToInt>(a: &[T]) -> Vec<T::Int> {
    let mut out = apply::<S, T, T::Int>(a, |v| v.trunc_to_int());
    out.extend(apply::<S, T, T::Int>(a, |v| v.round_to_int()));
    out
}

#[inline(always)]
fn test_from_int_impl<S: Simd, T: VToFloat>(a: &[T]) -> Vec<T::Float> {
    apply::<S, T, T::Float>(a, |v| v.to_float())
}

macro_rules! testgen_convert {
    // Attributes before a type apply to its `from_int` test.
    ($($(#[$meta:meta])* $ty: ty: $int: ty: $bits: ty: [$($edge: expr),*]),*) => {
        $(::paste::paste! {
            // Miri implements neither the AVX-512 float compares the fixups use nor
            // the NEON and LoongArch conversions.
            #[cfg_attr(all(miri, any(x86_v4, aarch64, loong64)), ignore)]
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_to_int_ $ty>]() {
                let mut rng = rand::rng();
                let mut a: Vec<$ty> = vec![
                    0.0, -0.0, 0.4, 0.5, 0.6, 1.5, 2.5, -0.5, -1.5, -2.5, 1e10, -1e10,
                    <$ty>::MAX, <$ty>::MIN, <$ty>::MIN_POSITIVE, <$ty>::from_bits(1),
                    <$ty>::INFINITY, <$ty>::NEG_INFINITY, <$ty>::NAN, -<$ty>::NAN,
                    $($edge),*
                ];
                // Integer bounds as floats: 2^(N-1) and -2^(N-1).
                let bound = -(<$int>::MIN as $ty);
                a.extend([bound, -bound]);
                for _ in 0..64 {
                    a.push(rng.random_range(-3e9..3e9) as $ty);
                    a.push(<$ty>::from_bits(rng.random::<$bits>()));
                }
                while a.len() % 64 != 0 {
                    a.push(0.0);
                }
                // Whole vectors within 2^51, some ties, for backends that
                // convert small lanes differently from large ones.
                let small = bound.min(2251799813685248.0);
                for i in 0..64 {
                    let x = rng.random_range(-small..small);
                    a.push(if i % 4 == 0 { x.trunc() + 0.5 } else { x });
                }
                // Whole vectors just past 2^51, where adding 1.5 * 2^52 no
                // longer rounds to an integer, to pin where the small path ends.
                let two_51 = 2251799813685248.0f64;
                let past = [two_51, two_51 + 0.5, two_51 + 1.0, two_51 + 3.0, 2.0 * two_51 - 0.5];
                for i in 0..64 {
                    let x = past[i % past.len()];
                    a.push((if i % 2 == 0 { x } else { -x }) as $ty);
                }
                let mut want: Vec<$int> = a.iter().map(|&x| x as $int).collect();
                want.extend(a.iter().map(|&x| x.round_ties_even() as $int));
                for_each_backend!(test_to_int_impl, $ty, &a[..], |out: &[$int]| {
                    for (i, (w, g)) in want.iter().zip(out).enumerate() {
                        let x = a[i % a.len()];
                        let op = if i < a.len() { "trunc" } else { "round" };
                        assert_eq!(w, g, "{op}_to_int({x:?})");
                    }
                });
            }

            // Miri does not implement the LoongArch conversions.
            #[cfg_attr(all(miri, loong64), ignore)]
            $(#[$meta])*
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_from_int_ $ty>]() {
                let mut rng = rand::rng();
                // Integers that need rounding, including ties.
                let precise = 1 << (<$ty>::MANTISSA_DIGITS);
                let mut a: Vec<$int> = vec![
                    0, 1, -1, <$int>::MIN, <$int>::MAX, precise, precise + 1, precise + 2,
                    precise + 3, -(precise + 1), -(precise + 3),
                ];
                while a.len() < crate::tests::SIZE {
                    a.push(rng.random());
                }
                let want: Vec<$ty> = a.iter().map(|&x| x as $ty).collect();
                for_each_backend!(test_from_int_impl, $int, &a[..], |out: &[$ty]| {
                    for ((x, w), g) in a.iter().zip(&want).zip(out) {
                        assert_eq!(w.to_bits(), g.to_bits(), "to_float({x})");
                    }
                });
            }
        })*
    };
}

testgen_convert!(
    f32: i32: u32: [2147483520.0, -2147483904.0, 4294967296.0, 8388607.5],
    // Miri does not implement AVX-512 i64 to f64 conversion.
    #[cfg_attr(all(miri, x86_v4), ignore)]
    f64: i64: u64: [
        9223372036854774784.0,
        -9223372036854777856.0,
        18446744073709551616.0,
        4503599627370495.5
    ]
);
