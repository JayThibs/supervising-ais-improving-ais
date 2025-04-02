import pytest

def pytest_configure(config):
    """Register custom marks to avoid warnings."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow) 