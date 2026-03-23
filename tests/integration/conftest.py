"""Configuration for integration tests."""

import pytest


def pytest_addoption(parser):
    """Add command line option to run slow tests."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow integration tests (includes actual training)"
    )
    parser.addoption(
        "--run-mock",
        action="store_true",
        default=False,
        help="Run mock training tests (simulated, no MLX required)"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "mock: marks tests as mock tests (no MLX required)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    skip_mock = pytest.mark.skip(reason="Need --run-mock option to run")
    
    run_slow = config.getoption("--run-slow")
    run_mock = config.getoption("--run-mock")
    
    for item in items:
        # Skip slow tests unless --run-slow
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        
        # Skip mock tests unless --run-mock (but run them by default if no flags)
        if "mock" in item.keywords and not run_mock and run_slow:
            item.add_marker(skip_mock)


@pytest.fixture
def mock_training_data():
    """Create minimal mock training data."""
    import json
    import tempfile
    
    examples = [
        {"instruction": "Test 1", "output": "Answer 1"},
        {"instruction": "Test 2", "output": "Answer 2"},
        {"instruction": "Test 3", "output": "Answer 3"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
        return f.name
