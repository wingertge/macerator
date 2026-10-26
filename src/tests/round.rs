use core::fmt::Debug;
use std::{vec, vec::Vec};

use half::f16;
use num_traits::Float;
use rand::Rng;

use crate::{tests::for_each_backend, vload_unaligned, vstore_unaligned, Scalar, Simd, Vector};

/// Values where rounding is easy to get wrong, then random ones.
fn inputs<T: Float + Scalar + Bits>() -> Vec<T> {
    // Above this every value is an integer.
    let integral = (T::epsilon().recip()).to_f64().expect("finite epsilon");
    let mut values: Vec<f64> = vec![
        0.0,
        0.4,
        0.5,
        0.6,
        1.0,
        1.5,
        2.5,
        3.5,
        4.5,
        -123.75,
        integral / 2.0 - 0.5,
        integral - 1.0,
        integral,
        integral + 2.0,
        f64::INFINITY,
        f64::NAN,
    ];
    let negated: Vec<f64> = values.iter().map(|v| -v).collect();
    values.extend(negated);
    let mut out: Vec<T> = values
        .into_iter()
        .map(|v| <T as num_traits::NumCast>::from(v).expect("float"))
        .collect();
    // The largest value below one half, the smallest subnormal, and the extremes.
    let half = <T as num_traits::NumCast>::from(0.5).expect("float");
    let below_half = T::from_bits_below(half);
    out.extend([
        below_half,
        -below_half,
        T::min_positive_value(),
        T::max_value(),
        T::min_value(),
    ]);
    out.extend([T::from_bits_low(1), -T::from_bits_low(1)]);
    let mut rng = rand::rng();
    for _ in 0..64 {
        out.push(<T as num_traits::NumCast>::from(rng.random_range(-64.0..64.0)).expect("float"));
    }
    let mut bits = vec![T::zeroed(); 64];
    rng.fill(bytemuck::cast_slice_mut::<T, u8>(&mut bits));
    out.extend(bits);
    // Whole vectors only.
    while !out.len().is_multiple_of(64) {
        out.push(T::zero());
    }
    out
}

/// Bit access for building edge values without `std` helpers per type.
trait Bits: Sized {
    /// The largest value below `x`, for positive finite `x`.
    fn from_bits_below(x: Self) -> Self;
    /// The value with these low bits and everything else zero.
    fn from_bits_low(bits: u16) -> Self;
}

impl Bits for f16 {
    fn from_bits_below(x: Self) -> Self {
        f16::from_bits(x.to_bits() - 1)
    }
    fn from_bits_low(bits: u16) -> Self {
        f16::from_bits(bits)
    }
}

impl Bits for f32 {
    fn from_bits_below(x: Self) -> Self {
        f32::from_bits(x.to_bits() - 1)
    }
    fn from_bits_low(bits: u16) -> Self {
        f32::from_bits(bits.into())
    }
}

impl Bits for f64 {
    fn from_bits_below(x: Self) -> Self {
        f64::from_bits(x.to_bits() - 1)
    }
    fn from_bits_low(bits: u16) -> Self {
        f64::from_bits(bits.into())
    }
}

/// Equal bits, or both NaN: payloads may differ between instructions.
fn assert_same<T: Float + Debug>(input: &[T], want: &[T], got: &[T], op: &str) {
    for ((x, w), g) in input.iter().zip(want).zip(got) {
        let same =
            (w.is_nan() && g.is_nan()) || (w == g && w.is_sign_negative() == g.is_sign_negative());
        assert!(same, "{op}({x:?}): expected {w:?}, got {g:?}");
    }
}

#[inline(always)]
fn apply<S: Simd, T: Scalar>(a: &[T], op: impl Fn(Vector<S, T>) -> Vector<S, T>) -> Vec<T> {
    let lanes = T::lanes::<S>();
    let mut out = vec![T::zeroed(); a.len()];
    for (a, out) in a.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let a = unsafe { vload_unaligned::<S, T>(a.as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), op(a)) };
    }
    out
}

macro_rules! testgen_round {
    ($op: ident, $reference: expr, $($(#[$meta:meta])* $ty: ty),*) => {
        $(::paste::paste! {
            #[inline(always)]
            fn [<test_ $op _ $ty _impl>]<S: Simd, T: crate::[<V $op:camel>]>(a: &[T]) -> Vec<T> {
                apply::<S, T>(a, |v| v.$op())
            }

            $(#[$meta])*
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_ $op _ $ty>]() {
                let a = inputs::<$ty>();
                let reference: fn($ty) -> $ty = $reference;
                let want: Vec<$ty> = a.iter().map(|&x| reference(x)).collect();
                for_each_backend!([<test_ $op _ $ty _impl>], $ty, &a[..], |out: &[$ty]| {
                    assert_same(&a, &want, out, stringify!($op))
                });
            }
        })*
    };
}

/// `std` has no f16 rounding; f32 holds every f16 value and every rounded
/// result exactly.
fn via_f32(x: f16, op: fn(f32) -> f32) -> f16 {
    f16::from_f32(op(x.to_f32()))
}

macro_rules! testgen_round_all {
    ($op: ident $(, #[$meta:meta])*) => {
        ::paste::paste! {
            mod [<$op _tests>] {
                use super::*;

                testgen_round!(
                    $op,
                    |x| via_f32(x, f32::$op),
                    // Generating f16 inputs uses F16C or FP16 conversions Miri lacks.
                    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
                    f16
                );
                testgen_round!($op, f32::$op, $(#[$meta])* f32);
                testgen_round!($op, f64::$op, $(#[$meta])* f64);
            }
        }
    };
}

// Miri does not implement AVX-512 `roundscale`.
testgen_round_all!(floor, #[cfg_attr(all(miri, x86_v4), ignore)]);
testgen_round_all!(ceil, #[cfg_attr(all(miri, x86_v4), ignore)]);
testgen_round_all!(trunc, #[cfg_attr(all(miri, x86_v4), ignore)]);
// Miri implements none of AVX-512 `roundscale`, NEON `frintn` (`llvm.roundeven`),
// and LoongArch `vfrintrne`.
testgen_round_all!(round_ties_even, #[cfg_attr(all(miri, any(x86_v4, aarch64, loong64)), ignore)]);
