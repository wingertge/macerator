use std::{vec, vec::Vec};

use half::{bf16, f16};
use rand::Rng;

use crate::{tests::for_each_backend, Mask, Simd, VMaskReduce};

/// For each vector: expected and actual `any`, expected and actual `all`, for
/// a mask from `from_bools` and for the same mask built as `!from_bools(!b)`,
/// which some backends store with different bytes.
#[inline(always)]
fn test_mask_reduce_impl<S: Simd, T: VMaskReduce>(bools: &[bool]) -> Vec<[bool; 2]> {
    let lanes = T::lanes::<S>();
    let mut out = Vec::new();
    for chunk in bools.chunks_exact(lanes) {
        let inverted: Vec<bool> = chunk.iter().map(|b| !b).collect();
        let direct = Mask::<S, T>::from_bools(chunk);
        let negated = !Mask::<S, T>::from_bools(&inverted);
        let any = chunk.iter().any(|&b| b);
        let all = chunk.iter().all(|&b| b);
        out.extend([[any, direct.any()], [all, direct.all()]]);
        out.extend([[any, negated.any()], [all, negated.all()]]);
    }
    out
}

/// All false, all true, one lane set and one lane clear at each position, then
/// random; blocks of 64 cover the widest vector of any type. Under Miri, only
/// the positions that are first, second, or last in some vector width.
fn patterns() -> Vec<bool> {
    let mut bools = vec![false; 64];
    bools.extend([true; 64]);
    let positions: Vec<usize> = if cfg!(miri) {
        vec![0, 1, 3, 7, 15, 31, 32, 63]
    } else {
        (0..64).collect()
    };
    for p in positions {
        bools.extend((0..64).map(|i| i == p));
        bools.extend((0..64).map(|i| i != p));
    }
    let mut rng = rand::rng();
    bools.extend((0..128).map(|_| rng.random_bool(0.5)));
    bools
}

macro_rules! testgen_mask_reduce {
    ($($ty: ty),*) => {
        $(::paste::paste! {
            // Miri does not implement LoongArch `bnz.v` and `bnz.b`.
            #[cfg_attr(all(miri, loong64), ignore)]
            #[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
            fn [<test_mask_reduce_ $ty>]() {
                let bools = patterns();
                for_each_backend!(test_mask_reduce_impl, $ty, &bools[..], |out: &[[bool; 2]]| {
                    for (i, [want, got]) in out.iter().enumerate() {
                        let op = if i % 2 == 0 { "any" } else { "all" };
                        assert_eq!(want, got, "{op} at result {i}");
                    }
                });
            }
        })*
    };
}

testgen_mask_reduce!(u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64);
