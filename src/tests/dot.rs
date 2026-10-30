use std::{vec, vec::Vec};

use rand::Rng;

use crate::{tests::for_each_backend, vload_unaligned, vstore_unaligned, Scalar, Simd, VDotAdd};

#[inline(always)]
fn test_dot_impl<S: Simd, T: VDotAdd>((a, b, acc): (&[T], &[T], &[T::Acc])) -> Vec<T::Acc> {
    let lanes = T::lanes::<S>();
    let acc_lanes = T::Acc::lanes::<S>();
    let mut out = vec![T::Acc::default(); acc.len()];
    for (i, out) in out.chunks_exact_mut(acc_lanes).enumerate() {
        let a = unsafe { vload_unaligned::<S, T>(a[i * lanes..].as_ptr()) };
        let b = unsafe { vload_unaligned::<S, T>(b[i * lanes..].as_ptr()) };
        let acc = unsafe { vload_unaligned::<S, T::Acc>(acc[i * acc_lanes..].as_ptr()) };
        unsafe { vstore_unaligned(out.as_mut_ptr(), a.dot_add(b, acc)) };
    }
    out
}

// Miri does not implement the NEON pairwise widening add.
#[cfg_attr(all(miri, aarch64), ignore)]
#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_dot_i8() {
    let mut rng = rand::rng();
    // Groups of four: the largest sum, the smallest, mixed signs, and an
    // accumulator that wraps.
    let mut groups: Vec<([i8; 4], [i8; 4], i32)> = vec![
        ([-128; 4], [-128; 4], 0),
        ([-128; 4], [127; 4], 0),
        ([127; 4], [127; 4], i32::MAX),
        ([-128; 4], [-128; 4], i32::MAX),
        ([-128; 4], [127; 4], i32::MIN),
        ([1, 2, 3, 4], [5, -6, 7, -8], 100),
        ([1, 0, 0, 0], [1, 0, 0, 0], 0),
        ([0, 0, 0, 1], [0, 0, 0, 1], 0),
    ];
    // Enough for four 512-bit vectors.
    while groups.len() < 64 {
        groups.push((rng.random(), rng.random(), rng.random()));
    }
    // The largest sum, with a wrapping accumulator, again in the last group of
    // each 128-bit block past the first two.
    for k in (11..64).step_by(4) {
        groups[k] = ([-128; 4], [-128; 4], i32::MAX);
    }
    let a: Vec<i8> = groups.iter().flat_map(|g| g.0).collect();
    let b: Vec<i8> = groups.iter().flat_map(|g| g.1).collect();
    let acc: Vec<i32> = groups.iter().map(|g| g.2).collect();
    let want: Vec<i32> = groups
        .iter()
        .map(|(a, b, acc)| {
            let sum: i32 = a.iter().zip(b).map(|(&a, &b)| a as i32 * b as i32).sum();
            acc.wrapping_add(sum)
        })
        .collect();
    for_each_backend!(
        test_dot_impl,
        i8,
        (&a[..], &b[..], &acc[..]),
        |out: &[i32]| {
            for (k, (w, g)) in want.iter().zip(out).enumerate() {
                assert_eq!(w, g, "dot_add of group {k}: {:?}", groups[k]);
            }
        }
    );
}
