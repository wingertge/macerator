# Macerator

Originally a thin wrapper around [`pulp`](https://github.com/sarah-quinones/pulp) to provide type-generic SIMD
operations using similar traits to the standard library, but now uses its own backend for more ergonomic
type inference behaviour and wider backend support. As backends are stabilized in Rust, the MSRV will
be increased and nightly requirements will be removed. For crates with a lower MSRV, older versions
should be automatically used by cargo.

## AVX-512 on Stable

AVX-512 intrinsics are stable since Rust 1.89, and the MSRV is now 1.94, so the `avx512` feature
(on by default) works on stable.

## Backends

| Feature set          | Tested on | Requires Nightly |
| -------------------- | --------- | ---------------- |
| x86_64-v2 (sse4.2)   | Hardware  | ❌                |
| x86_64-v3 (avx2)     | Hardware  | ❌                |
| x86_64-v4 (avx512)   | Hardware  | ❌                |
| aarch64 (Neon)       | Hardware  | ❌                |
| aarch64 (Neon + fp16)| Hardware  | ❌                |
| loongarch64 (lsx)    | QEMU      | ✅                |
| loongarch64 (lasx)   | QEMU      | ✅                |
| wasm32 (simd128)[^1] | Chrome    | ❌                |
| Any other target     | None[^2]  | ❌                |

[^1]: `wasm32` doesn't support runtime feature detection, so binary must be built
with `target_feature=+simd128`.
[^2]: Manually tested to ensure it builds, CI using QEMU may be added in the future. The scalar
backend is tested on all supported platforms.

`f16` support for `x86_64-v4` is disabled by default, since only one
Intel arch currently supports it, and AMD has no support. This may change as support expands. Note
that this also requires nightly, since the AVX-512 FP16 intrinsics are still unstable.

## Features

| Feature   | Default | What it does |
| --------- | ------- | ------------ |
| `std`     | ✅      | Detects CPU features at runtime. Without it, backends are chosen from the target features set at compile time, and scalar math uses `libm`. |
| `avx512`  | ✅      | Enables the `x86_64-v4` backend. |
| `fp16`    | ✅      | Enables native `f16` on Neon, and on `x86_64-v4` together with `nightly`. |
| `nightly` | ❌      | Enables what needs a nightly compiler: `loongarch64`, and `f16` on `x86_64-v4`. Without it, `loongarch64` builds still compile but only use the scalar backend. |

## Example

`#[with_simd]` generates a function without the `S` parameter, here `clamp_slice::<T>`. It picks
the best backend for the CPU at runtime and runs the body compiled for that backend.

```rust
use macerator::{vload_partial, vstore_partial, with_simd, Simd, VOrd, Vector};

// Helpers need to inline into the `with_simd` function to get the backend's CPU features.
#[inline(always)]
fn clamp<S: Simd, T: VOrd>(value: Vector<S, T>, min: T, max: T) -> Vector<S, T> {
    let min = min.splat();
    let max = max.splat();
    value.min(max).max(min)
}

#[with_simd]
fn clamp_slice<S: Simd, T: VOrd>(values: &mut [T], min: T, max: T) {
    let mut chunks = values.chunks_exact_mut(T::lanes::<S>());
    for chunk in &mut chunks {
        clamp(Vector::<S, T>::load(chunk), min, max).store(chunk);
    }
    let tail = chunks.into_remainder();
    // SAFETY: both only touch `tail.len()` elements, which `tail` holds.
    unsafe {
        let value = vload_partial::<S, T>(tail.as_ptr(), tail.len());
        vstore_partial(tail.as_mut_ptr(), clamp(value, min, max), tail.len());
    }
}
```
