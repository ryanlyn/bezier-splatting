"""Shared pytest configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--save-outputs",
        action="store_true",
        default=False,
        help="Save diagnostic images and metrics to tests/outputs/",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run reconstruction tests in fast mode (fewer targets, fewer steps, tier-1 only)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (full optimization loop)")
