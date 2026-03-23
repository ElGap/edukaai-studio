"""Pytest configuration for EdukaAI Studio tests.

This file provides:
- Common fixtures for all tests
- Test data generators
- Mock utilities
- Test configuration
"""

import sys
from pathlib import Path
import pytest
import json
import tempfile
import queue
from unittest.mock import Mock

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============ COMMON FIXTURES ============

@pytest.fixture
def test_data_dir():
    """Directory for test data files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_hf_token():
    """Mock HuggingFace token for testing."""
    return "hf_test_token_12345"


@pytest.fixture
def sample_training_config():
    """Sample training configuration for tests."""
    return {
        'model': 'mlx-community/Phi-3-mini-4k-instruct-4bit',
        'model_name': 'Phi-3 Mini',
        'iters': 10,
        'learning_rate': 1e-4,
        'batch_size': 1,
        'grad_accumulation': 4,
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.0,
        'max_seq_length': 2048,
        'early_stopping': 2,
        'validation_split': 10,
    }


@pytest.fixture
def sample_alpaca_data():
    """Sample Alpaca format training data."""
    return [
        {
            "instruction": "Explain the concept of machine learning.",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence..."
        },
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Write a haiku about nature.",
            "input": "",
            "output": "Wind through the trees\nLeaves dance in the autumn air\nNature's symphony"
        }
    ]


@pytest.fixture
def sample_mlx_format_data():
    """Sample MLX format training data."""
    return [
        {"text": "Instruction: Explain the concept of machine learning.\n\nOutput: Machine learning is a subset of artificial intelligence..."},
        {"text": "Instruction: What is the capital of France?\n\nOutput: The capital of France is Paris."},
        {"text": "Instruction: Write a haiku about nature.\n\nOutput: Wind through the trees\nLeaves dance in the autumn air\nNature's symphony"}
    ]


@pytest.fixture
def mock_gradio_components():
    """Mock Gradio components for UI testing."""
    class MockComponent:
        def __init__(self, name):
            self.name = name
            self.value = None
        
        def update(self, value):
            self.value = value
            return self
    
    return {
        'textbox': MockComponent('textbox'),
        'button': MockComponent('button'),
        'slider': MockComponent('slider'),
        'plot': MockComponent('plot'),
    }


@pytest.fixture
def mock_state():
    """Mock application state for testing."""
    return {
        'model_id': 'mlx-community/Phi-3-mini-4k-instruct-4bit',
        'model_name': 'Phi-3 Mini',
        'training_config': {},
        'training_active': False,
        'training_complete': False,
        'output_dir': None,
    }


# ============ TEST DATA GENERATORS ============

def generate_training_data(num_samples: int = 10, format: str = 'alpaca'):
    """Generate synthetic training data for testing.
    
    Args:
        num_samples: Number of samples to generate
        format: 'alpaca' or 'mlx'
        
    Returns:
        List of training samples
    """
    if format == 'alpaca':
        return [
            {
                "instruction": f"Sample instruction {i}",
                "input": f"Sample input {i}" if i % 2 == 0 else "",
                "output": f"Sample output {i}"
            }
            for i in range(num_samples)
        ]
    elif format == 'mlx':
        return [
            {"text": f"Instruction: Sample {i}\n\nOutput: Response {i}"}
            for i in range(num_samples)
        ]
    else:
        raise ValueError(f"Unknown format: {format}")


# ============ MOCK UTILITIES ============

@pytest.fixture
def mock_subprocess_factory():
    """Factory for creating mock subprocesses."""
    def create_mock(stdout_lines=None, returncode=0, delay=0):
        """Create a mock subprocess.
        
        Args:
            stdout_lines: Lines to yield as stdout
            returncode: Exit code
            delay: Delay between lines (seconds)
        """
        import time
        
        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.cmd = cmd
                self._returncode = returncode
                self._stdout = stdout_lines or []
                self._delay = delay
                self._poll = None
            
            @property
            def stdout(self):
                for line in self._stdout:
                    yield line
                    if self._delay:
                        time.sleep(self._delay)
            
            def poll(self):
                return self._poll
            
            def wait(self, timeout=None):
                self._poll = self._returncode
                return self._returncode
            
            def terminate(self):
                self._poll = -15
            
            def kill(self):
                self._poll = -9
        
        return MockPopen
    
    return create_mock


# ============ PYTEST CONFIGURATION ============

def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location
    for item in items:
        if "security" in item.nodeid.lower():
            item.add_marker(pytest.mark.security)
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)


# ============ TEST ENVIRONMENT ============

@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment."""
    import os
    
    # Ensure we're in test mode
    os.environ['EDUKAAI_TEST_MODE'] = '1'
    
    yield
    
    # Cleanup
    if 'EDUKAAI_TEST_MODE' in os.environ:
        del os.environ['EDUKAAI_TEST_MODE']


# Make fixtures available globally
pytest.fixture(generate_training_data)

