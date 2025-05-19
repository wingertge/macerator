# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
