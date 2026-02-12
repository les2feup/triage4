# triage4

Core TRIAGE/4 scheduling package for priority-aware IoT message serving.

## Package Scope

- Distributable package code lives in `src/triage4`.
- Research and benchmarking code lives in `assessment/` and is intentionally non-distribution.
- Compatibility modules under `src/schedulers` and `src/vanilla` support repository workflows and are excluded from wheel packaging.

## Installation

Install core package for runtime use:

```bash
pip install -e .
```

Install research dependencies for local assessment/benchmark execution:

```bash
pip install -e ".[research]"
```

Install development + research dependencies:

```bash
pip install -e ".[dev,research]"
```

## Packaging Guarantees

- Project distribution name: `triage4`
- Published wheel content: `triage4/*` modules and package metadata only
- `assessment/`, `tests/`, and benchmark artifacts are not shipped in distribution files
