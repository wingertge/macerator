# Macerator

Originally a thin wrapper around [`pulp`](https://github.com/sarah-quinones/pulp) to provide type-generic SIMD
operations using similar traits to the standard library, but now uses its own backend for more ergonomic
type inference behaviour. Currently in an MVP state, with only operations and backends needed for
[`burn`](https://github.com/tracel-ai/burn), which means no unstable features are currently used. This
may change in the future, but it will likely keep following `burn` quite closely.

## Backends

Currently supports backends for `x86_64-v2`, `x86_64-v3`, `x86_64-v4` (nightly only), `aarch64` and
`wasm32`. Note that `wasm32` doesn't support runtime feature detection, so binary must be built
with `target_feature=+simd128`. `f16` support for `x86_64-v4` is disabled by default, since only one
Intel arch currently supports it, and AMD has no support. This may change as support expands.

## Example

```rust
fn clamp<S: Simd, T: VOrd>(value: Vector<S, T>, min: T, max: T) -> Vector<S, T> {
    let min = min.splat();
    let max = max.splat();
    value.min(max).max(min)
}
```
