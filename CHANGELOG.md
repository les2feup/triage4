# Changelog

All notable changes to the `triage4` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] - 2026-05-27

### Changed
- **Documentation**: Enhanced `README.md` with the official LES2 project banner, version & environment badges, and a developer quick-start usage example.

## [1.0.0] - 2026-05-26

This is the stable release of the core `triage4` scheduling library, isolated from the research infrastructure and ready for production deployment.

### Added
- **Core Scheduler**: Implemented `TRIAGE4Scheduler` class for the four-band hierarchical scheduling system.
- **Adaptive Alarm Protection**: Integrated `AdaptiveTokenBucket` and `AlarmRateMonitor` to prevent network denial-of-service/floods.
- **Fair Queuing**: Integrated `DeviceFairQueue` and `SourceAwareQueue` to prevent single-device or single-zone starvation.
- **GitHub Actions Publishing Workflow**: Added `.github/workflows/publish.yml` using Node 20+ compatible steps and OIDC Trusted Publishing to release the package to PyPI on tag pushes.
- **Verification Reports**: Created package validation and CI/CD workflow documentation under `docs/chat-reports/`.

### Refactored
- **Core Package Isolation**: Relocated all research benchmarks, evaluation workloads, and non-core scripts into a top-level `assessment/` directory.
- **Package Scope**: Configured `pyproject.toml` and `MANIFEST.in` so that build artifacts (`.whl` and `.tar.gz`) contain only the production-safe `triage4` namespace and metadata.
- **Standardized Imports**: Added `__init__.py` files to subdirectories in `assessment/` to ensure absolute path import consistency across baseline and validation tests.
