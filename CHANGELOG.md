# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note: This changelog tracks what has shipped in released versions. For future plans and milestones, see [ROADMAP.md](ROADMAP.md).*

## [Unreleased]

### Added
- `penumbra_fhe.version()`: A Python API returning the current crate version.
- `penumbra_fhe.keygen()`: Generate cryptographic client and server keys.
- `penumbra_fhe.set_server_key()`: Set the server evaluation key globally for the current thread context.
- `penumbra_fhe.encrypt()` / `penumbra_fhe.decrypt()`: Encrypt and decrypt scalar values.
- `penumbra_fhe.Ciphertext`: Python class wrapper for encrypted scalars supporting `+`, `-`, and `*` (scalar multiplication) operators.

### Changed

### Deprecated

### Removed

### Fixed

### Security

[Unreleased]: https://github.com/Harikeshav-R/penumbra-fhe/compare/HEAD...HEAD
