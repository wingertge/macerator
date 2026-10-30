use crate::{Scalar, Simd, Vector};

pub trait VDotAdd: Scalar {
    /// The accumulator lane type; a vector holds a quarter as many of them.
    type Acc: Scalar;
    fn vdot_add<S: Simd>(
        a: Vector<S, Self>,
        b: Vector<S, Self>,
        acc: Vector<S, Self::Acc>,
    ) -> Vector<S, Self::Acc>;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: VDotAdd> Vector<S, T> {
    /// Dot products of each group of four lanes, added to the matching lane of
    /// `acc`: lane `k` of the result is `acc[k]` plus the sum of
    /// `self[4k + j] * b[4k + j]` for `j` in `0..4`, the products taken at the
    /// accumulator's width. A group's sum is exact; adding it to `acc` wraps.
    /// The inner loop of int8 matrix multiplication. SSE4.2, AVX2, AVX-512,
    /// NEON, and wasm do it in a few instructions; LoongArch and the scalar
    /// fallback compute it lane by lane.
    ///
    /// ```
    /// # use macerator::{Simd, Vector};
    /// fn dot_step<S: Simd>(a: Vector<S, i8>, b: Vector<S, i8>, acc: Vector<S, i32>) -> Vector<S, i32> {
    ///     a.dot_add(b, acc)
    /// }
    /// ```
    #[inline(always)]
    pub fn dot_add(self, b: Self, acc: Vector<S, T::Acc>) -> Vector<S, T::Acc> {
        T::vdot_add(self, b, acc)
    }
}

impl VDotAdd for i8 {
    type Acc = i32;
    #[inline(always)]
    fn vdot_add<S: Simd>(
        a: Vector<S, i8>,
        b: Vector<S, i8>,
        acc: Vector<S, i32>,
    ) -> Vector<S, i32> {
        S::typed(S::dot_i8(*a, *b, *acc))
    }
    #[inline(always)]
    fn is_accelerated<S: Simd>() -> bool {
        S::dot_i8_supported()
    }
}
