# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1](https://github.com/wingertge/macerator/compare/macerator-v0.5.0...macerator-v0.5.1) - 2026-09-27

### Fixed

- *(macros)* accept where-bounds on the SIMD parameter ([#67](https://github.com/wingertge/macerator/pull/67))
- select the LSX backend on CPUs without LASX ([#64](https://github.com/wingertge/macerator/pull/64))
- build without default features and on no_std targets ([#62](https://github.com/wingertge/macerator/pull/62))

### Other

- say when mul_add is fused, and update the README ([#66](https://github.com/wingertge/macerator/pull/66))
- cover the whole integer range in compare, min/max and reduce tests ([#63](https://github.com/wingertge/macerator/pull/63))

## [0.5.0](https://github.com/wingertge/macerator/compare/macerator-v0.4.0...macerator-v0.5.0) - 2026-09-27

### Added

- add VSqrt ([#60](https://github.com/wingertge/macerator/pull/60))
- use Newton-Raphson for recip() to get ULP-level precision  ([#47](https://github.com/wingertge/macerator/pull/47))

### Fixed

- [**breaking**] make half-vector loads and stores consistent across backends ([#53](https://github.com/wingertge/macerator/pull/53))
- *(macros)* keep dispatcher-only attributes off the with_simd body ([#55](https://github.com/wingertge/macerator/pull/55))
- wrap integer lanes emulated with scalar code ([#54](https://github.com/wingertge/macerator/pull/54))
- emit all-ones lanes from emulated f16 comparisons ([#51](https://github.com/wingertge/macerator/pull/51))
- compute abs_i64 correctly on SSE4.2, AVX2 and LoongArch ([#52](https://github.com/wingertge/macerator/pull/52))
- declare FP16 reduce_add asm scratch register as clobbered ([#50](https://github.com/wingertge/macerator/pull/50))
- use ordered predicate for AVX2 float equality ([#49](https://github.com/wingertge/macerator/pull/49))

### Other

- widen the f16 reduce_add tolerance ([#61](https://github.com/wingertge/macerator/pull/61))
- allow inlining arch detection into dispatchers ([#58](https://github.com/wingertge/macerator/pull/58))
- use plain unaligned loads on SSE4.2 and AVX2 ([#56](https://github.com/wingertge/macerator/pull/56))

## [0.4.0](https://github.com/wingertge/macerator/compare/macerator-v0.3.4...macerator-v0.4.0) - 2026-09-02

### Added

- Add FP16 variant for `aarch64` ([#40](https://github.com/wingertge/macerator/pull/40))

### Fixed

- [**breaking**] Make `run_vectorized` and `dispatch` unsafe to avoid unsoundness ([#41](https://github.com/wingertge/macerator/pull/41))

### Other

- Update dependencies of `macerator-macros` ([#43](https://github.com/wingertge/macerator/pull/43))

## [0.3.4](https://github.com/wingertge/macerator/compare/macerator-v0.3.3...macerator-v0.3.4) - 2026-08-03

### Other

- updated the following local packages: macerator-macros

## [0.3.3](https://github.com/wingertge/macerator/compare/macerator-v0.3.2...macerator-v0.3.3) - 2026-05-04

### Fixed

- Avoid stack clobbering in load_low/load_high on loong64 ([#36](https://github.com/wingertge/macerator/pull/36))

### Other

- Add Miri CI workflow ([#35](https://github.com/wingertge/macerator/pull/35))
- Fix miri UB in scalar backend ([#33](https://github.com/wingertge/macerator/pull/33))

## [0.3.2](https://github.com/wingertge/macerator/compare/macerator-v0.3.1...macerator-v0.3.2) - 2026-04-20

### Fixed

- Separate relaxed ops from Simd128 and provide fallback option ([#28](https://github.com/wingertge/macerator/pull/28))

## [0.3.1](https://github.com/wingertge/macerator/compare/macerator-v0.3.0...macerator-v0.3.1) - 2026-03-10

### Fixed

- Remove unsupported `__mm_fmadd` from x86_64-v2 ([#25](https://github.com/wingertge/macerator/pull/25))

### Other

- Unconditionally enable `no_std` ([#22](https://github.com/wingertge/macerator/pull/22))

## [0.3.0](https://github.com/wingertge/macerator/compare/macerator-v0.2.10...macerator-v0.3.0) - 2026-02-08

### Added

- Make masks more ergonomic and add mask logical ops ([#20](https://github.com/wingertge/macerator/pull/20))

### Fixed

- Seal traits that were meant to be sealed before ([#21](https://github.com/wingertge/macerator/pull/21))
- Fix weird lifetime handling in macro to allow for reference params ([#17](https://github.com/wingertge/macerator/pull/17))
- Add missing `#[inline(always)]` on unsigned int `reduce_add` ([#15](https://github.com/wingertge/macerator/pull/15))

### Other

- Revert seal on `Scalar`, manually impl Send/Sync
- Only trigger CI on main push ([#18](https://github.com/wingertge/macerator/pull/18))

## [0.2.10](https://github.com/wingertge/macerator/compare/macerator-v0.2.9...macerator-v0.2.10) - 2026-02-07

### Added

- Implement add, min and max reductions ([#13](https://github.com/wingertge/macerator/pull/13))

## [0.2.9](https://github.com/wingertge/macerator/compare/macerator-v0.2.8...macerator-v0.2.9) - 2025-08-07

### Added

- Allow using AVX-512 on stable if rustc is at version 1.89 or higher ([#12](https://github.com/wingertge/macerator/pull/12))

### Other

- Update README
- Add commitlint

## [0.2.8](https://github.com/wingertge/macerator/compare/macerator-v0.2.7...macerator-v0.2.8) - 2025-05-19

### Added

- Add support for `loongarch64` ([#9](https://github.com/wingertge/macerator/pull/9))

### Other

- Update README.md ([#10](https://github.com/wingertge/macerator/pull/10))
- Clean up conditional compilation ([#8](https://github.com/wingertge/macerator/pull/8))
- Merge branch 'main' of https://github.com/wingertge/macerator
- Explicitly specify MSRV

## [0.2.7](https://github.com/wingertge/macerator/compare/macerator-v0.2.6...macerator-v0.2.7) - 2025-05-14

### Other

- Fix compile issues on targets without SIMD support

## [0.2.6](https://github.com/wingertge/macerator/compare/macerator-v0.2.5...macerator-v0.2.6) - 2025-03-15

### Added

- Enable F16C for V3+ so the compiler can use native conversion

## [0.2.5](https://github.com/wingertge/macerator/compare/macerator-v0.2.4...macerator-v0.2.5) - 2025-03-14

### Other

- Add WASM rustflags to config.toml
- Add rustflags required for WASM compilation of rand
- Update `rand` and `half`
- Add release-plz
