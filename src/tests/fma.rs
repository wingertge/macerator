use std::{vec, vec::Vec};

use rand::Rng;

use crate::{tests::for_each_backend, vload_unaligned, vstore_unaligned, Simd, VFma};

#[inline(always)]
fn test_fma_impl<S: Simd, T: VFma>((a, b, c): (&[T], &[T], &[T])) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::default(); a.len()];
    for (i, out) in out.chunks_exact_mut(lanes).enumerate() {
        let at = i * lanes;
        let a = unsafe { vload_unaligned::<S, T>(a[at..].as_ptr()) };
        let b = unsafe { vload_unaligned::<S, T>(b[at..].as_ptr()) };
        let c = unsafe { vload_unaligned::<S, T>(c[at..].as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), a.fma(b, c)) };
    }
    out
}

macro_rules! testgen_fma {
    ($($ty: ty),*) => {
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_fma_ $ty>]() {
                let eps = <$ty>::EPSILON;
                let inf = <$ty>::INFINITY;
                // (1 + eps)(1 - eps) - 1 is -eps^2 fused and 0 unfused.
                let mut cases: Vec<($ty, $ty, $ty)> = vec![
                    (1.0 + eps, 1.0 - eps, -1.0),
                    (inf, 0.0, 1.0),
                    (inf, 1.0, -inf),
                    (-0.0, 1.0, 0.0),
                    (-0.0, 1.0, -0.0),
                    (<$ty>::NAN, 1.0, 1.0),
                    (<$ty>::MAX, 2.0, -<$ty>::MAX),
                    (<$ty>::MIN_POSITIVE, 0.5, 0.0),
                ];
                let mut rng = rand::rng();
                while cases.len() < crate::tests::SIZE {
                    let a: $ty = rng.random_range(-100.0..100.0);
                    let b: $ty = rng.random_range(-100.0..100.0);
                    // With c = -(a * b), fused gives the rounding error of the
                    // product and unfused gives exactly zero.
                    cases.push((a, b, -(a * b)));
                    cases.push((a, b, rng.random_range(-100.0..100.0)));
                }
                let a: Vec<$ty> = cases.iter().map(|c| c.0).collect();
                let b: Vec<$ty> = cases.iter().map(|c| c.1).collect();
                let c: Vec<$ty> = cases.iter().map(|c| c.2).collect();
                let want: Vec<$ty> = cases.iter().map(|&(a, b, c)| a.mul_add(b, c)).collect();
                for_each_backend!(test_fma_impl, $ty, (&a[..], &b[..], &c[..]), |out: &[$ty]| {
                    for (i, (w, g)) in want.iter().zip(out).enumerate() {
                        let same = (w.is_nan() && g.is_nan()) || w.to_bits() == g.to_bits();
                        assert!(same, "fma{:?}: expected {w:?}, got {g:?}", cases[i]);
                    }
                });
            }
        })*
    };
}

testgen_fma!(f32, f64);
