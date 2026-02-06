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
    parser.addoption(
        "--typecheck",
        action="store_true",
        default=False,
        help="Enable jaxtyping+beartype runtime shape checking on bezier_splatting",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (full optimization loop)")
    if config.getoption("--typecheck", default=False):
        from jaxtyping import install_import_hook
        install_import_hook("bezier_splatting", "beartype.beartype")
