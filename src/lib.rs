#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(avx512_fp16, feature(stdarch_x86_avx512_f16))]
#![cfg_attr(loong64, feature(stdarch_loongarch))]
#![cfg_attr(
    all(loong64, feature = "std"),
    feature(stdarch_loongarch_feature_detection)
)]

#[cfg(any(feature = "std", test))]
extern crate std;

mod arithmetic;
pub(crate) mod backend;
mod base;
mod bitwise;
mod convert;
mod dot;
mod mask_reduce;

mod fma;
mod ord;
mod reduce;
mod select;

mod shift;
mod unary;

#[cfg(test)]
mod tests;

/// The function `with_simd` generates for `arch` is `unsafe`, but an unsafe
/// call in `arch` on a safe function still needs an `unsafe` block, as it would
/// in the body.
///
/// ```compile_fail,E0133
/// unsafe fn pick() -> macerator::Arch {
///     macerator::Arch::detect()
/// }
///
/// #[macerator::with_simd(arch = pick())]
/// fn f<S: macerator::Simd>(x: u32) -> u32 {
///     x
/// }
/// ```
#[cfg(doctest)]
pub struct WithSimdArchUnsafeCall;

pub use arithmetic::*;
pub use backend::*;
pub use base::*;
pub use bitwise::*;
pub use convert::*;
pub use dot::*;
pub use mask_reduce::*;

pub use fma::*;
pub use ord::*;
pub use reduce::*;
pub use select::*;

pub use shift::*;
pub use unary::*;

pub use macerator_macros::*;
