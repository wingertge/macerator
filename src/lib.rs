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
mod mask_reduce;

mod fma;
mod ord;
mod reduce;
mod select;

mod shift;
mod unary;

#[cfg(test)]
mod tests;

pub use arithmetic::*;
pub use backend::*;
pub use base::*;
pub use bitwise::*;
pub use convert::*;
pub use mask_reduce::*;

pub use fma::*;
pub use ord::*;
pub use reduce::*;
pub use select::*;

pub use shift::*;
pub use unary::*;

pub use macerator_macros::*;
