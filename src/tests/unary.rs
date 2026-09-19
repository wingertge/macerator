use core::{fmt::Debug, num::FpCategory};
use half::f16;
use std::{vec, vec::Vec};

use num_traits::{Float, NumCast};

use crate::{
    assert_relative_eq,
    tests::{approx::RelativeEq, assert_eq, test_unop, unop},
    Simd, VAbs, VRecip, Vector,
};

use super::{testgen_unop, testgen_unop_values, Unop};

#[inline(always)]
fn test_recip_impl<S: Simd, T: VRecip>(a: &[T]) -> Vec<T> {
    unop!(VRecip, |a: Vector<S, T>| a.recip());
    test_unop::<S, T, VRecipOp<T>>(a)
}

#[inline(always)]
fn test_abs_impl<S: Simd, T: VAbs>(a: &[T]) -> Vec<T> {
    unop!(VAbs, |a: Vector<S, T>| a.abs());
    test_unop::<S, T, VAbsOp<T>>(a)
}

fn assert_approx_eq_recip<T: RelativeEq<Epsilon = T> + Debug + NumCast + Copy>(
    lhs: &[T],
    rhs: &[T],
) {
    let epsilon = match core::mem::size_of::<T>() {
        // `f16`: unrefined hardware estimate on some backends, only ~8 bits accurate.
        ..=2 => T::from(2.0.powf(-8.0)).unwrap(),
        // `f32`: couple ULP, i.e. a small multiple of `f32::EPSILON` (2^-23).
        4 => T::from(4.0 * f32::EPSILON as f64).unwrap(),
        // `f64`: couple ULP, i.e. a small multiple of `f64::EPSILON` (2^-52).
        _ => T::from(4.0 * f64::EPSILON).unwrap(),
    };
    for (a, b) in lhs.iter().zip(rhs) {
        // `epsilon` is only the absolute floor, which does nothing once a
        // reciprocal is large; `max_relative` is what a ULP budget needs.
        assert_relative_eq!(*a, *b, epsilon = epsilon, max_relative = epsilon);
    }
}

/// `assert_relative_eq!` treats inf and NaN as never equal, so special values
/// are compared bitwise instead: any NaN matches any NaN, and zeroes have to
/// agree on sign.
fn recip_bits_eq<T: Float>(a: T, b: T) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b && a.is_sign_negative() == b.is_sign_negative())
}

/// The contract documented on [`Vector::recip`], against scalar `T::recip()`.
///
/// `±0`, `±inf` and `NaN` have to map exactly; everything else owes the ULP
/// budget. The exception is a subnormal input, or an input whose reciprocal is
/// subnormal: a backend refining an estimate instruction with no subnormal
/// support (`rcp_ps`, i.e. sse/avx2) may saturate to `±inf`/`±0` there. Even
/// then it has to saturate towards the true value and keep the sign of the
/// input, and never land on NaN.
fn assert_recip_specials<T: Float + Debug + RelativeEq<Epsilon = T>>(
    input: &[T],
    expected: &[T],
    actual: &[T],
) {
    let epsilon = T::epsilon() * NumCast::from(4.0).unwrap();
    for ((x, want), got) in input.iter().zip(expected).zip(actual) {
        let saturation = if x.classify() == FpCategory::Subnormal {
            // 1/subnormal is enormous, so saturating overshoots to infinity.
            Some(T::infinity())
        } else if want.classify() == FpCategory::Subnormal {
            // 1/x is subnormal, so saturating undershoots to zero.
            Some(T::zero())
        } else {
            // Normal in, normal out: nothing to excuse.
            None
        };
        let saturation = saturation.map(|s| if x.is_sign_negative() { -s } else { s });
        assert!(
            recip_bits_eq(*got, *want)
                || crate::relative_eq!(*got, *want, epsilon = epsilon, max_relative = epsilon)
                || saturation.is_some_and(|s| recip_bits_eq(*got, s)),
            "recip({x:?}): expected {want:?} (saturation permitted: {saturation:?}), got {got:?}"
        );
    }
}

testgen_unop!(
    test_recip,
    recip,
    1,
    100,
    assert_approx_eq_recip,
    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
    f16
);
testgen_unop!(
    test_recip,
    recip,
    -100,
    100,
    assert_approx_eq_recip,
    #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
    f32,
    #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
    f64
);
testgen_unop_values!(
    test_recip_specials,
    test_recip_impl,
    recip,
    vec![
        // Where refinement used to corrupt the estimate: 2 - a*y0 is 0*inf.
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        // Ordinary lanes, to keep the fallback from being taken vector-wide.
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        // Subnormal, 1/x overflows: ±inf is the correctly rounded answer.
        1e-40,
        -1e-40,
        // Subnormal, 1/x is finite: ±inf here is saturation, not the answer.
        3e-39,
        -3e-39,
        1e-38,
        -1e-38,
        // Normal, 1/x is subnormal: ±0 here is saturation, not the answer.
        f32::MAX,
        -f32::MAX,
        1e38,
        -1e38,
    ],
    assert_recip_specials,
    #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
    f32
);
testgen_unop_values!(
    test_recip_specials,
    test_recip_impl,
    recip,
    vec![
        // Where refinement used to corrupt the estimate: 2 - a*y0 is 0*inf.
        0.0f64,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        // Ordinary lanes, to keep the fallback from being taken vector-wide.
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        // Subnormal, 1/x overflows: ±inf is the correctly rounded answer.
        1e-320,
        -1e-320,
        // Subnormal, 1/x is finite: ±inf here is saturation, not the answer.
        1e-308,
        -1e-308,
        // Normal, 1/x is subnormal: ±0 here is saturation, not the answer.
        f64::MAX,
        -f64::MAX,
        1e308,
        -1e308,
    ],
    assert_recip_specials,
    #[cfg_attr(all(miri, any(aarch64, x86_v4)), ignore)]
    f64
);
testgen_unop!(
    test_abs,
    abs,
    -100,
    100,
    assert_eq,
    i8,
    i16,
    i32,
    #[cfg_attr(all(miri, any(x86_v3, x86_v4, aarch64)), ignore)]
    f16,
    f32,
    f64
);

// TEMPORARY exhaustive verification harness.
std::thread_local! {
    static STATS: core::cell::RefCell<(u64, f32, f32)> =
        const { core::cell::RefCell::new((0, 0.0, f32::INFINITY)) };
}

fn check_exhaustive_recip(input: &[f32], want: &[f32], got: &[f32]) {
    for ((x, w), g) in input.iter().zip(want).zip(got) {
        if w.is_nan() {
            assert!(g.is_nan(), "recip({x:e}): want NaN, got {g:e}");
            continue;
        }
        assert!(!g.is_nan(), "recip({x:e}): want {w:e}, got NaN");
        assert!(
            g.is_sign_negative() == w.is_sign_negative(),
            "recip({x:e}): want {w:e}, got {g:e} (wrong sign)"
        );
        if g.to_bits() == w.to_bits() {
            continue;
        }
        // Saturation: must go towards the true value, never away from it.
        if *g == 0.0 {
            assert!(w.abs() < 1.0, "recip({x:e}): flushed {w:e} to zero");
            STATS.with_borrow_mut(|s| s.1 = s.1.max(w.abs()));
            continue;
        }
        if g.is_infinite() {
            assert!(w.abs() > 1.0, "recip({x:e}): want {w:e}, got {g:e}");
            STATS.with_borrow_mut(|s| s.2 = s.2.min(w.abs()));
            continue;
        }
        assert!(w.is_finite(), "recip({x:e}): want {w:e}, got {g:e}");
        let m = |u: u32| if u & 0x8000_0000 != 0 { !u } else { u | 0x8000_0000 };
        let ulp = (m(g.to_bits()) as i64 - m(w.to_bits()) as i64).unsigned_abs();
        assert!(ulp <= 4, "recip({x:e}): want {w:e}, got {g:e} ({ulp} ULP)");
        STATS.with_borrow_mut(|s| s.0 = s.0.max(ulp));
    }
}

#[test]
#[ignore]
fn exhaustive_recip_f32() {
    const CHUNK: usize = 4096;
    let mut a = vec![0f32; CHUNK];
    let mut want = vec![0f32; CHUNK];
    let mut base: u64 = 0;
    while base <= u32::MAX as u64 {
        for i in 0..CHUNK {
            a[i] = f32::from_bits((base + i as u64) as u32);
            want[i] = 1.0 / a[i];
        }
        crate::tests::for_each_backend!(test_recip_impl, f32, &a, |out: &[f32]| {
            check_exhaustive_recip(&a, &want, out)
        });
        base += CHUNK as u64;
    }
    STATS.with_borrow(|s| {
        std::println!("worst ULP: {}", s.0);
        std::println!("largest true value flushed to zero: {:e} (MIN_POSITIVE {:e})", s.1, f32::MIN_POSITIVE);
        std::println!("smallest true value saturated to inf: {:e} (MAX {:e})", s.2, f32::MAX);
    });
}
