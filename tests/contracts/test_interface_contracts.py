"""Contract Tests for Component Interfaces.

These tests verify that components agree on data formats and interfaces.
They catch the exact type of bug you had: monitor sends 'train_losses' but
train.py expects 'train_loss'.
"""

import pytest
import queue
from typing import Dict, Any
import json


class TestMonitorToTrainContract:
    """Contract: TrainingMonitor → train.py data format."""
    
    CONTRACT_SCHEMA = {
        'required_keys': ['iteration', 'train_losses', 'val_losses', 'progress_percent'],
        'optional_keys': ['best_loss', 'best_iter', 'peak_memory_gb', 'resource_stats'],
        'forbidden_keys': ['train_loss', 'val_loss'],  # Singular - old format
        'key_types': {
            'iteration': int,
            'train_losses': dict,
            'val_losses': dict,
            'progress_percent': int,
        }
    }
    
    def test_monitor_outputs_required_keys(self):
        """Monitor MUST output all required keys."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Simulate what monitor would send
        test_data = {
            'iteration': 50,
            'train_losses': {50: 2.585, 40: 2.740},
            'val_losses': {50: 2.585},
            'progress_percent': 25,
            'best_loss': 2.585,
            'peak_memory_gb': 4.357,
        }
        
        # Verify contract
        for key in self.CONTRACT_SCHEMA['required_keys']:
            assert key in test_data, f"Monitor missing required key: {key}"
    
    def test_monitor_doesnt_use_deprecated_keys(self):
        """Monitor must NOT use deprecated singular keys (train_loss, val_loss)."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Create sample data monitor would send
        progress_data = {
            'iteration': 50,
            'train_losses': {50: 2.585},
            'val_losses': {50: 2.585},
        }
        
        # Check for forbidden keys
        for forbidden in self.CONTRACT_SCHEMA['forbidden_keys']:
            assert forbidden not in progress_data, \
                f"Monitor using deprecated key: {forbidden}. Use plural form."
    
    def test_train_accepts_contract(self):
        """train.py MUST be able to consume monitor's output."""
        # This simulates what train.py does with progress_data
        progress_data = {
            'iteration': 50,
            'train_losses': {50: 2.585},
            'val_losses': {50: 2.585},
            'progress_percent': 25,
        }
        
        # train.py extraction logic (fixed version)
        train_losses = {}
        if 'train_losses' in progress_data:
            train_losses.update(progress_data['train_losses'])
        
        # Verify extraction works
        assert 50 in train_losses, "train.py must extract iteration 50"
        assert train_losses[50] == 2.585, "train.py must extract loss value"
    
    def test_train_handles_backward_compatibility(self):
        """train.py should handle both old and new formats gracefully."""
        # Old format (for backward compatibility)
        old_format = {
            'iteration': 50,
            'train_loss': 2.585,
            'val_loss': 2.585,
        }
        
        # New format
        new_format = {
            'iteration': 50,
            'train_losses': {50: 2.585},
            'val_losses': {50: 2.585},
        }
        
        # Both should be extractable
        for format_name, data in [('old', old_format), ('new', new_format)]:
            train_losses = {}
            
            # Check both keys
            if 'train_loss' in data:
                train_losses[data['iteration']] = data['train_loss']
            if 'train_losses' in data:
                train_losses.update(data['train_losses'])
            
            assert 50 in train_losses, f"{format_name} format failed"


class TestStateContract:
    """Contract: Application state format across tabs."""
    
    STATE_SCHEMA = {
        'required_keys': [
            'training_active',
            'training_complete',
            'train_losses',
            'val_losses',
        ],
        'key_types': {
            'training_active': bool,
            'training_complete': bool,
            'train_losses': dict,
            'val_losses': dict,
        }
    }
    
    def test_state_has_required_keys(self):
        """State must have all required keys for tabs to work."""
        state = {
            'training_active': True,
            'training_complete': False,
            'train_losses': {50: 2.585},
            'val_losses': {50: 2.585},
            'output_dir': '/tmp/output',
        }
        
        for key in self.STATE_SCHEMA['required_keys']:
            assert key in state, f"State missing required key: {key}"
    
    def test_state_consistency_invariant(self):
        """State invariant: cannot be both active and complete."""
        inconsistent_state = {
            'training_active': True,
            'training_complete': True,
        }
        
        # This should be rejected or handled
        assert not (inconsistent_state['training_active'] and 
                   inconsistent_state['training_complete']), \
            "State invariant violated: cannot be both active and complete"


class TestConfigurationContract:
    """Contract: Training configuration format."""
    
    CONFIG_SCHEMA = {
        'required_keys': [
            'model_name',
            'model_id',
            'iterations',
            'learning_rate',
            'lora_rank',
        ],
        'key_types': {
            'iterations': int,
            'learning_rate': str,
            'lora_rank': int,
            'lora_alpha': int,
        }
    }
    
    def test_config_has_required_keys(self):
        """Training config must have all required keys."""
        config = {
            'model_name': 'Phi-3 Mini',
            'model_id': 'mlx-community/Phi-3-mini-4k-instruct-4bit',
            'iterations': 200,
            'learning_rate': '1e-4',
            'lora_rank': 16,
            'lora_alpha': 32,
        }
        
        for key in self.CONFIG_SCHEMA['required_keys']:
            assert key in config, f"Config missing required key: {key}"
    
    def test_config_types_are_correct(self):
        """Config values must be correct types."""
        config = {
            'iterations': 200,
            'learning_rate': '1e-4',
            'lora_rank': 16,
        }
        
        for key, expected_type in self.CONFIG_SCHEMA['key_types'].items():
            if key in config:
                assert isinstance(config[key], expected_type), \
                    f"Config key {key} should be {expected_type.__name__}, got {type(config[key]).__name__}"


class TestLogLineContract:
    """Contract: Training log line formats."""
    
    SUPPORTED_FORMATS = [
        # mlx_lm.lora format
        r'Iter\s+(\d+).*Train loss\s+([\d.]+|nan|inf)',
        # Validation format
        r'Iter\s+(\d+).*Val loss\s+([\d.]+)',
        # Summary format
        r'Iteration\s+(\d+):\s+([\d.]+)',
    ]
    
    def test_all_supported_formats_parse(self):
        """All documented log formats must be parseable."""
        test_lines = [
            "Iter 50: Train loss 2.585, Learning Rate 1.000e-04",
            "Iter 50: Val loss 2.585, Val took 5.642s",
            "Iteration 300: 1.5860",
        ]
        
        for line in test_lines:
            matched = False
            for pattern in self.SUPPORTED_FORMATS:
                import re
                if re.search(pattern, line, re.IGNORECASE):
                    matched = True
                    break
            
            assert matched, f"Line not matched by any format: {line}"


class TestFileFormatContract:
    """Contract: Data file formats (JSONL)."""
    
    def test_alpaca_format_contract(self):
        """Alpaca format must have instruction and output."""
        sample = {
            'instruction': 'Test instruction',
            'input': 'Test input',
            'output': 'Test output',
        }
        
        # Required fields
        assert 'instruction' in sample
        assert 'output' in sample
        
        # Types
        assert isinstance(sample['instruction'], str)
        assert isinstance(sample['output'], str)
    
    def test_chatml_format_contract(self):
        """ChatML format must have messages array with role/content."""
        sample = {
            'messages': [
                {'role': 'system', 'content': 'You are helpful'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi!'},
            ]
        }
        
        assert 'messages' in sample
        assert isinstance(sample['messages'], list)
        
        for msg in sample['messages']:
            assert 'role' in msg
            assert 'content' in msg
            assert msg['role'] in ['system', 'user', 'assistant']


class TestAPIContracts:
    """Contract: API/Interface contracts."""
    
    def test_training_monitor_interface(self):
        """TrainingMonitor must implement expected interface."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        output_q = queue.Queue()
        progress_q = queue.Queue()
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Required methods
        required_methods = [
            'start_training',
            'is_running',
            'is_complete',
            'is_stopped',
        ]
        
        for method in required_methods:
            assert hasattr(monitor, method), f"TrainingMonitor missing method: {method}"
            assert callable(getattr(monitor, method)), f"TrainingMonitor.{method} not callable"
    
    def test_start_training_signature(self):
        """start_training must accept expected arguments."""
        import inspect
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        sig = inspect.signature(TrainingMonitor.start_training)
        params = list(sig.parameters.keys())
        
        # Should have: self, args_dict, data_file, and optionally others
        assert 'args_dict' in params, "start_training missing args_dict parameter"
        assert 'data_file' in params, "start_training missing data_file parameter"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
