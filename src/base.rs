use core::fmt::Debug;
use paste::paste;
use pulp::{
    bytemuck::{AnyBitPattern, CheckedBitPattern, NoUninit, Pod, Zeroable},
    MemMask, Simd,
};

pub trait Vectorizable: Sized + Copy + AnyBitPattern + NoUninit {
    type Vector<S: Simd>: Debug + Copy + Send + Sync + Pod + 'static;
    type Mask<S: Simd>: Debug
        + Copy
        + Send
        + Sync
        + Zeroable
        + NoUninit
        + CheckedBitPattern
        + 'static;

    fn lanes<S: Simd>() -> usize;

    /// Convert slice into a head slice containing as many vectorized values as
    /// possible, and a tail slice, containing the leftover elements.
    fn as_simd<S: Simd>(data: &[Self]) -> (&[Self::Vector<S>], &[Self]);
    /// Convert mutable slice into a head slice containing as many vectorized
    /// values as possible, and a tail slice, containing the leftover
    /// elements.
    fn as_mut_simd<S: Simd>(data: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]);
    /// Convert slice into a head slice containing leftover single elements,
    /// and a tail slice containing as many vectorized values as possible.
    fn as_rsimd<S: Simd>(data: &[Self]) -> (&[Self], &[Self::Vector<S>]);
    /// Convert mutable slice into a head slice containing leftover single
    /// elements, and a tail slice containing as many vectorized values as
    /// possible.
    fn as_mut_rsimd<S: Simd>(data: &mut [Self]) -> (&mut [Self], &mut [Self::Vector<S>]);
    /// Load a vector from an aligned element pointer. Must be aligned to the
    /// whole vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`read`](std::ptr::read), with the
    /// additional requirement that the entire vector must be aligned and
    /// valid, not just the element at `ptr`.
    unsafe fn vload<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S>;
    /// Load a vector from an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read_unaligned`](std::ptr::read_unaligned), with the additional
    /// requirement that the entire vector must be valid, not just the
    /// element at `ptr`.
    unsafe fn vload_unaligned<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S>;
    /// Load the lower half of a vector from an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read_unaligned`](std::ptr::read_unaligned), with the additional
    /// requirement that the lower half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vload_low<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S>;
    /// Load the upper half of a vector from an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read_unaligned`](std::ptr::read_unaligned), with the additional
    /// requirement that the upper half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vload_high<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S>;
    /// Store the lower half of a vector to an aligned element pointer. Must be
    /// aligned to the whole vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write`](std::ptr::write), with the additional
    /// requirement that the entire vector must be valid and aligned to the size
    /// of the full vectgor, not just the element at `ptr`.
    unsafe fn vstore<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>);
    /// Store the upper half of a vector to an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write_unaligned`](std::ptr::write_unaligned), with the additional
    /// requirement that the entire vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vstore_unaligned<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>);
    /// Store the upper half of a vector to an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write_unaligned`](std::ptr::write_unaligned), with the additional
    /// requirement that the lower half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vstore_low<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>);
    /// Store the upper half of a vector to an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write_unaligned`](std::ptr::write_unaligned), with the additional
    /// requirement that the upper half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vstore_high<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>);
    /// Loads the elements of the vector that are selected by the `mask` from an
    /// aligned element pointer. The pointer must be aligned to the full
    /// vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read`](std::ptr::read), with the additional
    /// requirement that all selected elements of the vector must be valid, not
    /// just the element at `ptr`.
    unsafe fn vmask_load<S: Simd>(
        simd: S,
        mask: MemMask<Self::Mask<S>>,
        ptr: *const Self,
    ) -> Self::Vector<S>;
    /// Stores the elements of the vector that are selected by the `mask` to an
    /// aligned element pointer. The pointer must be aligned to the full
    /// vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write`](std::ptr::write), with the additional
    /// requirement that all selected elements of the vector must be valid, not
    /// just the element at `ptr`.
    unsafe fn vmask_store<S: Simd>(
        simd: S,
        mask: MemMask<Self::Mask<S>>,
        ptr: *mut Self,
        value: Self::Vector<S>,
    );
    /// Create a vector with the scalar `value` in each element.
    fn splat<S: Simd>(simd: S, value: Self) -> Self::Vector<S>;
}

/// Inverted type method to allow using method syntax where possible
/// This can only be done for methods that take `T` directly, since Rust isn't
/// able to infer the type when using an associated type.
pub trait SimdExt: Simd {
    /// Load a vector from an aligned element pointer. Must be aligned to the
    /// whole vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`read`](std::ptr::read), with the
    /// additional requirement that the entire vector must be aligned and
    /// valid, not just the element at `ptr`.
    unsafe fn vload<T: Vectorizable>(self, ptr: *const T) -> T::Vector<Self> {
        T::vload(self, ptr)
    }
    /// Load a vector from an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read_unaligned`](std::ptr::read_unaligned), with the additional
    /// requirement that the entire vector must be valid, not just the
    /// element at `ptr`.
    unsafe fn vload_unaligned<T: Vectorizable>(self, ptr: *const T) -> T::Vector<Self> {
        T::vload_unaligned(self, ptr)
    }
    /// Load the lower half of a vector from an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read_unaligned`](std::ptr::read_unaligned), with the additional
    /// requirement that the lower half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vload_low<T: Vectorizable>(self, ptr: *const T) -> T::Vector<Self> {
        T::vload_low(self, ptr)
    }
    /// Load the upper half of a vector from an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read_unaligned`](std::ptr::read_unaligned), with the additional
    /// requirement that the upper half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vload_high<T: Vectorizable>(self, ptr: *const T) -> T::Vector<Self> {
        T::vload_high(self, ptr)
    }
    /// Store the lower half of a vector to an aligned element pointer. Must be
    /// aligned to the whole vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write`](std::ptr::write), with the additional
    /// requirement that the entire vector must be valid and aligned to the size
    /// of the full vectgor, not just the element at `ptr`.
    unsafe fn vstore<T: Vectorizable>(self, ptr: *mut T, value: T::Vector<Self>) {
        T::vstore(self, ptr, value);
    }
    /// Store the upper half of a vector to an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write_unaligned`](std::ptr::write_unaligned), with the additional
    /// requirement that the entire vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vstore_unaligned<T: Vectorizable>(self, ptr: *mut T, value: T::Vector<Self>) {
        T::vstore_unaligned(self, ptr, value);
    }
    /// Store the upper half of a vector to an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write_unaligned`](std::ptr::write_unaligned), with the additional
    /// requirement that the lower half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vstore_low<T: Vectorizable>(self, ptr: *mut T, value: T::Vector<Self>) {
        T::vstore_low(self, ptr, value);
    }
    /// Store the upper half of a vector to an unaligned element pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write_unaligned`](std::ptr::write_unaligned), with the additional
    /// requirement that the upper half of the vector must be valid, not just
    /// the element at `ptr`.
    unsafe fn vstore_high<T: Vectorizable>(self, ptr: *mut T, value: T::Vector<Self>) {
        T::vstore_high(self, ptr, value);
    }
    /// Loads the elements of the vector that are selected by the `mask` from an
    /// aligned element pointer. The pointer must be aligned to the full
    /// vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`read`](std::ptr::read), with the additional
    /// requirement that all selected elements of the vector must be valid, not
    /// just the element at `ptr`.
    unsafe fn vmask_load<T: Vectorizable>(
        self,
        mask: MemMask<T::Mask<Self>>,
        ptr: *const T,
    ) -> T::Vector<Self> {
        T::vmask_load(self, mask, ptr)
    }
    /// Stores the elements of the vector that are selected by the `mask` to an
    /// aligned element pointer. The pointer must be aligned to the full
    /// vector.
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`write`](std::ptr::write), with the additional
    /// requirement that all selected elements of the vector must be valid, not
    /// just the element at `ptr`.
    unsafe fn vmask_store<T: Vectorizable>(
        self,
        mask: MemMask<T::Mask<Self>>,
        ptr: *mut T,
        value: T::Vector<Self>,
    ) {
        T::vmask_store(self, mask, ptr, value);
    }
    /// Create a vector with the scalar `value` in each element.
    fn splat<T: Vectorizable>(self, value: T) -> T::Vector<Self> {
        T::splat(self, value)
    }
}

impl<S: Simd> SimdExt for S {}

macro_rules! impl_vectorizable {
    ($ty: ty, $mask: ident) => {
        paste! {
            impl Vectorizable for $ty {
                type Vector<S: Simd> = S::[<$ty s>];
                type Mask<S: Simd> = S::[<$mask s>];

                fn lanes<S: Simd>() -> usize {
                    S::[<$ty:upper _LANES>]
                }

                #[inline(always)]
                fn as_simd<S: Simd>(data: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
                    S::[<as_simd_ $ty s>](data)
                }
                #[inline(always)]
                fn as_mut_simd<S: Simd>(data: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
                    S::[<as_mut_simd_ $ty s>](data)
                }
                #[inline(always)]
                fn as_rsimd<S: Simd>(data: &[Self]) -> (&[Self], &[Self::Vector<S>]) {
                    S::[<as_rsimd_ $ty s>](data)
                }
                #[inline(always)]
                fn as_mut_rsimd<S: Simd>(data: &mut [Self]) -> (&mut [Self], &mut [Self::Vector<S>]) {
                    S::[<as_mut_rsimd_ $ty s>](data)
                }
                #[inline(always)]
                unsafe fn vload<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S> {
                    simd.[<load_ptr_ $ty s>](ptr)
                }
                #[inline(always)]
                unsafe fn vload_unaligned<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S> {
                    simd.[<load_unaligned_ptr_ $ty s>](ptr)
                }
                #[inline(always)]
                unsafe fn vload_low<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S> {
                    simd.[<load_unaligned_ptr_low_ $ty s>](ptr)
                }
                #[inline(always)]
                unsafe fn vload_high<S: Simd>(simd: S, ptr: *const Self) -> Self::Vector<S> {
                    simd.[<load_unaligned_ptr_high_ $ty s>](ptr)
                }
                #[inline(always)]
                unsafe fn vstore<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>) {
                    simd.[<store_ptr_ $ty s>](ptr, value)
                }
                #[inline(always)]
                unsafe fn vstore_unaligned<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>) {
                    simd.[<store_unaligned_ptr_ $ty s>](ptr, value)
                }
                #[inline(always)]
                unsafe fn vstore_low<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>) {
                    simd.[<store_unaligned_ptr_low_ $ty s>](ptr, value)
                }
                #[inline(always)]
                unsafe fn vstore_high<S: Simd>(simd: S, ptr: *mut Self, value: Self::Vector<S>) {
                    simd.[<store_unaligned_ptr_high_ $ty s>](ptr, value)
                }
                #[inline(always)]
                unsafe fn vmask_load<S: Simd>(simd: S, mask: MemMask<Self::Mask<S>>, ptr: *const Self) -> Self::Vector<S> {
                    simd.[<mask_load_ptr_ $ty s>](mask, ptr)
                }
                #[inline(always)]
                unsafe fn vmask_store<S: Simd>(simd: S, mask: MemMask<Self::Mask<S>>, ptr: *mut Self, value: Self::Vector<S>) {
                    simd.[<mask_store_ptr_ $ty s>](mask, ptr, value)
                }
                #[inline(always)]
                fn splat<S: Simd>(simd: S, value: Self) -> Self::Vector<S> {
                    simd.[<splat_ $ty s>](value)
                }
            }
        }
    };
}

impl_vectorizable!(u8, m8);
impl_vectorizable!(i8, m8);
impl_vectorizable!(u16, m16);
impl_vectorizable!(i16, m16);
impl_vectorizable!(u32, m32);
impl_vectorizable!(i32, m32);
impl_vectorizable!(f32, m32);
impl_vectorizable!(u64, m64);
impl_vectorizable!(i64, m64);
impl_vectorizable!(f64, m64);
