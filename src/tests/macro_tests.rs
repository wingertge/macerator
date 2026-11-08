#![allow(
    unused,
    clippy::extra_unused_type_parameters,
    clippy::needless_lifetimes
)]

use std::{borrow::Cow, boxed::Box, vec::Vec};

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

#[with_simd]
fn test_where_simd<S>(a: u32) -> u32
where
    S: Simd,
{
    a + 1
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_where_clause() {
    assert_eq!(test_where_simd(1), 2);
}

#[with_simd]
fn test_const_first<const N: usize, S: Simd, T: Copy>(a: [T; N]) -> T {
    a[0]
}

#[with_simd]
fn test_const_last<S: Simd, T: Copy, const N: usize>(a: [T; N]) -> T {
    a[0]
}

/// Generic arguments follow the declared order, without the SIMD parameter.
#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_generic_order() {
    assert_eq!(test_const_first::<2, u32>([1, 2]), 1);
    assert_eq!(test_const_last::<u32, 2>([1, 2]), 1);
}

#[with_simd]
fn test_ref_in_tuple<S: Simd>(a: (&u32, &u32)) -> u32 {
    *a.0 + *a.1
}

#[with_simd]
fn test_ref_to_ref<S: Simd>(a: &&u32) -> u32 {
    **a
}

#[with_simd]
fn test_ref_in_generic_arg<S: Simd>(a: Option<&u32>) -> u32 {
    *a.unwrap()
}

#[with_simd]
fn test_placeholder_lifetime<S: Simd>(a: Cow<'_, str>, b: &'_ u32) -> usize {
    a.len() + *b as usize
}

/// The two references can have different lifetimes, and `&mut` can't shrink the
/// inner one to match the outer.
#[with_simd]
fn test_mut_ref_to_ref<S: Simd>(a: &mut &u32) -> u32 {
    **a
}

/// These are higher-ranked. Naming their lifetime in the struct would make them
/// plain `fn(&'s u32)`, which can't be passed on to the inner fn.
#[with_simd]
fn test_fn_args<S: Simd>(a: fn(&u32) -> u32, b: Box<dyn Fn(&u32) -> u32>) -> u32 {
    let x = 1;
    a(&x) + b(&x)
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_elided_inputs() {
    let (x, mut y) = (1, &2);
    assert_eq!(test_ref_in_tuple((&x, y)), 3);
    assert_eq!(test_ref_to_ref(&y), 2);
    assert_eq!(test_ref_in_generic_arg(Some(&x)), 1);
    assert_eq!(test_placeholder_lifetime(Cow::Borrowed("ab"), &x), 3);
    assert_eq!(test_mut_ref_to_ref(&mut y), 2);
    assert_eq!(test_fn_args(|x| *x, Box::new(|x| *x)), 2);
    assert_eq!(test_hrtb_with_elided(&()), 1);
}

trait Pair<'x, T> {}

impl<'x, T> Pair<'x, T> for () {}

/// Only `'x` is higher-ranked here. The `&u32` is an ordinary elided lifetime.
#[with_simd]
fn test_hrtb_with_elided<S: Simd>(a: &dyn for<'x> Pair<'x, &u32>) -> u32 {
    1
}

#[with_simd]
#[allow(mismatched_lifetime_syntaxes)]
fn test_output_from_named<'a, S: Simd>(a: &'a u32) -> &u32 {
    a
}

/// One argument with one distinct lifetime, as Rust counts it.
#[with_simd]
#[allow(mismatched_lifetime_syntaxes)]
fn test_output_from_ref_to_ref<'a, S: Simd>(a: &'a &'a u32) -> &u32 {
    a
}

#[with_simd]
#[allow(mismatched_lifetime_syntaxes)]
fn test_output_from_tuple<'a, S: Simd>(a: (&'a u32, &'a u32)) -> &u32 {
    a.1
}

/// Lifetimes bound by `fn` and `for<'x>` don't count.
#[with_simd]
fn test_output_past_higher_ranked<S: Simd>(
    a: &u32,
    b: fn(&u32) -> u32,
    c: Box<dyn for<'x> Pair<'x, u32>>,
) -> &u32 {
    a
}

#[with_simd]
fn test_output_in_generic_arg<S: Simd>(a: Option<&u32>) -> Option<&u32> {
    a
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_elided_output() {
    let x = 1;
    assert_eq!(*test_output_from_named(&x), 1);
    assert_eq!(test_output_in_generic_arg(Some(&x)), Some(&1));
    assert_eq!(*test_output_from_ref_to_ref(&&x), 1);
    assert_eq!(*test_output_from_tuple((&0, &x)), 1);
    assert_eq!(*test_output_past_higher_ranked(&x, |x| *x, Box::new(())), 1);
}

#[with_simd]
fn test_type_only_in_output<S: Simd, T: Default>() -> T {
    T::default()
}

#[with_simd]
fn test_lifetime_only_in_output<'a, S: Simd>() -> &'a u32 {
    &1
}

#[with_simd]
fn test_unsized_type_only_in_output<S: Simd, T: ?Sized>(_: u32) -> Option<Box<T>> {
    None
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_params_only_in_output() {
    assert_eq!(test_type_only_in_output::<u32>(), 0);
    assert_eq!(*test_lifetime_only_in_output(), 1);
    assert!(test_unsized_type_only_in_output::<str>(1).is_none());
}

#[with_simd(arch = crate::Arch::detect())]
fn test_arch_private<S: Simd>(a: u32) -> u32 {
    a + 1
}

#[with_simd(arch = crate::Arch::detect())]
pub fn test_arch_pub<S: Simd>(a: u32) -> u32 {
    a + 1
}

#[with_simd(arch = crate::Arch::detect())]
unsafe fn test_arch_unsafe<S: Simd>(a: u32) -> u32 {
    a + 1
}

#[with_simd(arch = crate::AutoArch::new())]
pub unsafe fn test_arch_pub_unsafe<S: Simd>(a: u32) -> u32 {
    a + 1
}

#[::wasm_bindgen_test::wasm_bindgen_test(unsupported = test)]
fn test_with_simd_arch() {
    unsafe {
        assert_eq!(test_arch_private(1), 2);
        assert_eq!(test_arch_pub(1), 2);
        assert_eq!(test_arch_unsafe(1), 2);
        assert_eq!(test_arch_pub_unsafe(1), 2);
    }
}
