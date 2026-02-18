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
