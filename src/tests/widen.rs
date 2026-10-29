use std::{vec, vec::Vec};

use half::{bf16, f16};
use rand::Rng;

use crate::{tests::for_each_backend, Scalar, Simd, VNarrow, VWiden, Vector};

/// Each chunk of `T::lanes` elements widened, low half then high half.
#[inline(always)]
fn test_widen_impl<S: Simd, T: VWiden>(a: &[T]) -> Vec<T::Wide> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::Wide::default(); a.len()];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let [lo, hi] = Vector::<S, T>::load(a).widen();
        lo.store(&mut out[..lanes / 2]);
        hi.store(&mut out[lanes / 2..]);
    }
    out
}

/// Each chunk of `T::lanes` wide elements narrowed into one vector.
#[inline(always)]
fn test_narrow_impl<S: Simd, T: VNarrow>(a: &[T::Wide]) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::default(); a.len()];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let lo = Vector::<S, T::Wide>::load(&a[..lanes / 2]);
        let hi = Vector::<S, T::Wide>::load(&a[lanes / 2..]);
        Vector::<S, T>::narrow(lo, hi).store(out);
    }
    out
}

fn random_bits<T: Scalar>(len: usize) -> Vec<T> {
    let mut out = vec![T::zeroed(); len];
    rand::rng().fill(bytemuck::cast_slice_mut::<T, u8>(&mut out));
    out
}

macro_rules! testgen_widen_int {
    ($($ty: ty => $wide: ty),*) => {
        $(::paste::paste! {
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_widen_ $ty>]() {
                let mut a = random_bits::<$ty>(crate::tests::SIZE);
                a[..4].copy_from_slice(&[0, 1, <$ty>::MIN, <$ty>::MAX]);
                let want: Vec<$wide> = a.iter().map(|&x| x as $wide).collect();
                for_each_backend!(test_widen_impl, $ty, &a[..], |out: &[$wide]| {
                    assert_eq!(&want[..], out)
                });
            }

            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_narrow_ $ty>]() {
                let a = random_bits::<$wide>(crate::tests::SIZE);
                let want: Vec<$ty> = a.iter().map(|&x| x as $ty).collect();
                for_each_backend!(test_narrow_impl, $ty, &a[..], |out: &[$ty]| {
                    assert_eq!(&want[..], out)
                });
            }
        })*
    };
}

testgen_widen_int!(u8 => u16, i8 => i16, u16 => u32, i16 => i32, u32 => u64, i32 => i64);

/// Equal bits, or both NaN.
macro_rules! assert_float_eq {
    ($inputs: expr, $want: expr, $got: expr, $op: literal) => {
        for ((x, w), g) in $inputs.iter().zip($want).zip($got) {
            assert!(
                w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()),
                "{}({x:?}): want {w:?}, got {g:?}",
                $op
            );
        }
    };
}

macro_rules! testgen_widen_float {
    ($($(#[$meta:meta])* $ty: ty => $wide: ty: $bits: ty, $wide_bits: ty, $widen: expr, $narrow: expr),*) => {
        $(::paste::paste! {
            $(#[$meta])*
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_widen_ $ty>]() {
                // Random bits: NaN payloads, infinities, subnormals, both zeros.
                let a = random_bits::<$ty>(crate::tests::SIZE);
                let want: Vec<$wide> = a.iter().map(|&x| ($widen)(x)).collect();
                for_each_backend!(test_widen_impl, $ty, &a[..], |out: &[$wide]| {
                    assert_float_eq!(a, &want, out, "widen");
                });
            }

            $(#[$meta])*
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_narrow_ $ty>]() {
                let mut rng = rand::rng();
                // The bits a narrow value drops from its wide form.
                let dropped = <$wide>::MANTISSA_DIGITS as usize - narrow_mantissa_digits::<$ty>();
                let drop_mask: $wide_bits = (1 << dropped) - 1;
                let tie: $wide_bits = 1 << (dropped - 1);
                // Random wide bits, mostly out of the narrow range, then
                // narrow values with random dropped bits and with exact ties.
                let mut a = random_bits::<$wide>(crate::tests::SIZE);
                for x in random_bits::<$ty>(crate::tests::SIZE) {
                    let w = ($widen)(x).to_bits();
                    a.push(<$wide>::from_bits(w | (rng.random::<$wide_bits>() & drop_mask)));
                    a.push(<$wide>::from_bits(w | tie));
                }
                a.extend([<$wide>::MAX, <$wide>::MIN, <$wide>::INFINITY, <$wide>::NAN, 0.0, -0.0]);
                // NaNs with a payload only in the dropped bits, which round to
                // infinity unless kept quiet, and with every mantissa bit set,
                // which carries into the sign.
                let inf = <$wide>::INFINITY.to_bits();
                let sign = (-0.0 as $wide).to_bits();
                for bits in [inf | 1, inf | drop_mask, inf | tie, !sign, !0] {
                    a.extend([<$wide>::from_bits(bits), <$wide>::from_bits(bits | sign)]);
                }
                while a.len() % 64 != 0 {
                    a.push(1.0);
                }
                let want: Vec<$ty> = a.iter().map(|&x| ($narrow)(x)).collect();
                for_each_backend!(test_narrow_impl, $ty, &a[..], |out: &[$ty]| {
                    assert_float_eq!(a, &want, out, "narrow");
                });
            }
        })*
    };
}

fn narrow_mantissa_digits<T: 'static>() -> usize {
    use core::any::TypeId;
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f16>() {
        f16::MANTISSA_DIGITS as usize
    } else if t == TypeId::of::<bf16>() {
        bf16::MANTISSA_DIGITS as usize
    } else {
        f32::MANTISSA_DIGITS as usize
    }
}

testgen_widen_float!(
    // Generating the reference uses F16C or FP16 conversions Miri lacks.
    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
    f16 => f32: u16, u32, f16::to_f32, f16::from_f32,
    bf16 => f32: u16, u32, bf16::to_f32, bf16::from_f32,
    f32 => f64: u32, u64, |x: f32| x as f64, |x: f64| x as f32
);
