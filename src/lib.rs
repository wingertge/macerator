#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    all(feature = "nightly", target_arch = "x86_64"),
    feature(avx512_target_feature, stdarch_x86_avx512, stdarch_x86_avx512_f16)
)]

mod arithmetic;
pub(crate) mod backend;
mod base;
mod bitwise;
mod ord;
mod unary;

#[cfg(test)]
mod tests;

pub use arithmetic::*;
pub use backend::*;
pub use base::*;
pub use bitwise::*;
pub use ord::*;
pub use unary::*;

pub use macerator_macros::*;
