use half::f16;
use paste::paste;

use crate::{Scalar, Simd, Vector};

pub trait ReduceAdd: Scalar {
    fn reduce_add<S: Simd>(input: Vector<S, Self>) -> Self;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: ReduceAdd> Vector<S, T> {
    /// Sum of all lanes. Integers wrap on overflow.
    ///
    /// For floats the order of the additions depends on the backend, so sums
    /// can differ in the last bits between backends.
    #[inline(always)]
    pub fn reduce_add(self) -> T {
        T::reduce_add(self)
    }
}

pub trait ReduceMin: Scalar {
    fn reduce_min<S: Simd>(input: Vector<S, Self>) -> Self;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: ReduceMin> Vector<S, T> {
    /// Smallest lane.
    ///
    /// For f32 and f64, NaN lanes depend on the backend: NEON returns NaN
    /// when any lane is NaN, while x86, wasm and the scalar fallback can
    /// return a lane that is not NaN. On wasm this differs from
    /// [`min`](Self::min), which propagates NaN. Which zero comes back when
    /// lanes hold both `-0.0` and `0.0` is also unspecified. Do not rely on
    /// either case where results must agree across backends.
    #[inline(always)]
    pub fn reduce_min(self) -> T {
        T::reduce_min(self)
    }
}

pub trait ReduceMax: Scalar {
    fn reduce_max<S: Simd>(input: Vector<S, Self>) -> Self;
    fn is_accelerated<S: Simd>() -> bool;
}

impl<S: Simd, T: ReduceMax> Vector<S, T> {
    /// Largest lane.
    ///
    /// For floats, NaN lanes and signed zeros behave as described for
    /// [`reduce_min`](Self::reduce_min).
    #[inline(always)]
    pub fn reduce_max(self) -> T {
        T::reduce_max(self)
    }
}

macro_rules! impl_reduce {
    ($trait: ident, $name: ident, $($ty: ty),*) => {
        $(paste! {
            impl $trait for $ty {
                #[inline(always)]
                fn [<$trait:snake>]<S: Simd>(input: Vector<S, Self>) -> Self {
                    S::[<$name _ $ty>](*input)
                }
                #[inline(always)]
                fn is_accelerated<S: Simd>() -> bool {
                    S::[<$name _ $ty _supported>]()
                }
            }
        })*
    };
}

impl_reduce!(ReduceAdd, reduce_add, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64);
impl_reduce!(ReduceMin, reduce_min, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64);
impl_reduce!(ReduceMax, reduce_max, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64);
