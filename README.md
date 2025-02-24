# Macerator

A thin wrapper around [`pulp`](https://github.com/sarah-quinones/pulp) to provide type-generic SIMD
operations using similar traits to the standard library. Currently based on a fork of pulp until
necessary changes are merged.

## Example

```rust
fn vclamp<S: Simd, T: VOrd>(simd: S, value: T::Vector<S>, min: T, max: T) -> T::Vector<S> {
    let min = simd.splat(min);
    let max = simd.splat(max);
    let v = T::vmin(value, max);
    T::vmax(value, min)
}
```
