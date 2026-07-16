# Changelog

All notable changes to the `triage4` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-07-16

Adaptive Alarm Protection now limits the source that causes an overload rather than the band as a whole.

### Added
- **Per-source rate limiting**: `SourceRateLimiter` gives every alarm source its own rate monitor and token bucket, so a source is judged only on the rate it alone emits. Admission now requires both the per-source decision and the band-global one, evaluated in that order, so shed traffic never spends the tokens legitimate alarms depend on.
- **Configuration**: `alarm_source_abnormal_threshold`, `alarm_source_deactivation_threshold`, `alarm_source_limit_budget`, `alarm_source_limit_period`, and `alarm_source_burst_capacity`, plus `disable_source_rate_limit` to reproduce the previous band-global-only behaviour.
- **Delivery tracking**: `SchedulerResult.delivered` marks which jobs were served. A dropped job is stored with a waiting time of `0.0`, which no timing array can distinguish from a job served instantly; consumers computing latency must filter on this mask rather than read the arrays directly.
- **Scheduler metadata**: per-source activation and deactivation counts, and the number of sources tracked and currently limited.

### Changed
- **Shedding is now per-source.** Previously one aggregate alarm rate was measured across the ALARM band and one shared budget was spent, so a legitimate alarm could be dropped because of another device's behaviour. A source that stays below its own threshold is no longer affected by what its neighbours do, and the first alarm from any source is always admitted, since its bucket is created full. Existing deployments should re-check their AAP tuning: a device that 1.1.0 left alone may now be rate-limited on its own.
- **Alarm-rate monitors are keyed by device** rather than by zone priority.

### Removed
- **Legacy `src/schedulers` tree**: thin adapter modules kept for the packaging transition, now unused by the repo, the tests, and the prototype. It was already excluded from the distribution, so package consumers are unaffected.

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
