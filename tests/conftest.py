"""Shared pytest configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--save-outputs",
        action="store_true",
        default=False,
        help="Save diagnostic images and metrics to tests/outputs/",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (full optimization loop)")
