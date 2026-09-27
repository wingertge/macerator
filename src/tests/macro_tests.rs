#![allow(
    unused,
    clippy::extra_unused_type_parameters,
    clippy::needless_lifetimes
)]

use std::vec::Vec;

use crate as macerator;
use macerator_macros::with_simd;

use crate::Simd;

#[with_simd]
fn test_simple<S: Simd>(a: Vec<f32>) -> f32 {
    let _ = a;
    0.0
}

#[with_simd]
fn test_generic<S: Simd, F: Default>(a: Vec<F>) -> F {
    let _ = a;
    F::default()
}

#[with_simd]
fn test_ref_input<S: Simd, F: Default>(a: &[F]) -> F {
    let _ = a;
    F::default()
}

#[with_simd]
fn test_ref_input_explicit<'a, S: Simd, F: Default>(a: &'a [F]) -> F {
    let _ = a;
    F::default()
}

#[with_simd]
fn test_ref_output<S: Simd>(a: &[f32]) -> &f32 {
    &a[0]
}

#[with_simd]
#[inline(never)]
fn test_inline_never<S: Simd>(a: u32) -> u32 {
    a + 1
}

#[with_simd]
#[inline]
fn test_inline_hint<S: Simd>(a: u32) -> u32 {
    a + 1
}

#[with_simd]
fn test_wildcard_arg<S: Simd>(_: u32, b: u32) -> u32 {
    b
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_attributes_and_wildcards() {
    assert_eq!(test_inline_never(1), 2);
    assert_eq!(test_inline_hint(1), 2);
    assert_eq!(test_wildcard_arg(1, 2), 2);
}
