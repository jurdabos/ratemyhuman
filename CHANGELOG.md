# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]
### Security
- `pyproject.toml`, `uv.lock`, `requirements.txt`: raised the Pillow floor to `>=12.2.0` and refreshed the generated lock / export so runtime installs use `pillow==12.2.0`. This clears the Dependabot-reported Pillow range `>=11.2.0,<11.3.0` (`CVE-2025-48379`, high) and `CVE-2026-25990` (high); the `>=12.2.0` floor also keeps the project aligned with the broader repository-wide Pillow bump.
### Changed
- `pyproject.toml`, `uv.lock`, `requirements.txt`: replaced brittle exact PyTorch nightly pins (`torch==2.12.0.dev20260324+cu128`, `torchvision==0.26.0.dev20260324+cu128`) with durable cu128 nightly ranges (`torch>=2.12.0.dev,<2.13`, `torchvision>=0.27.0.dev,<0.28`) while keeping `[tool.uv.sources]` routed to the `pytorch-cu128` index. `uv.lock` now pins currently available wheels (`torch==2.12.0.dev20260408+cu128`, `torchvision==0.27.0.dev20260407+cu128`), so clean checkouts can resolve and sync again even after PyTorch has rotated the older nightly wheels off the index.
### Notes / clarifications
- `uv sync` completed successfully after the dependency refresh.
- `uv run pytest -m "not slow" --tb=short` passed: 110 tests passed, 3 slow tests deselected.
