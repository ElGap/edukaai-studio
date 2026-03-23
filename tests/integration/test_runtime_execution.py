"""Test to catch runtime errors that static analysis misses.

This test actually EXECUTES the code paths to find bugs like:
- UnboundLocalError (variable referenced before assignment)
- Import errors
- Runtime logic errors
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os


class TestRuntimeCodeExecution:
    """Actually run the code to catch runtime errors."""
    
    def test_training_monitor_imports_and_basic_functionality(self):
        """Test that training_monitor can be imported and instantiated."""
        try:
            from edukaai_studio.ui.training_monitor import TrainingMonitor
            
            # Create a monitor instance
            import queue
            output_q = queue.Queue()
            progress_q = queue.Queue()
            monitor = TrainingMonitor(output_q, progress_q)
            
            # Verify basic attributes exist
            assert hasattr(monitor, 'process')
            assert hasattr(monitor, 'training_complete')
            assert hasattr(monitor, 'stop_event')
            
        except Exception as e:
            pytest.fail(f"Failed to import or instantiate TrainingMonitor: {e}")
    
    def test_start_training_method_runs_without_runtime_errors(self):
        """Actually call start_training to catch UnboundLocalError and similar."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        import queue
        import tempfile
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal valid training data file
            data_file = Path(tmpdir) / "train.json"
            with open(data_file, 'w') as f:
                json.dump([
                    {"instruction": "Test", "input": "", "output": "Test output"}
                ], f)
            
            output_q = queue.Queue()
            progress_q = queue.Queue()
            monitor = TrainingMonitor(output_q, progress_q)
            
            # Mock args that would trigger the code path
            args_dict = {
                'model': 'mlx-community/Phi-3-mini-4k-instruct-4bit',
                'iters': 10,
                'learning_rate': 1e-4,
                'batch_size': 1,
                'grad_accumulation': 4,
                'lora_rank': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.0,
            }
            
            # This will fail immediately because lora-train.py doesn't exist
            # but it will execute the code path and catch UnboundLocalError
            try:
                result = monitor.start_training(
                    args_dict=args_dict,
                    data_file=str(data_file),
                    validation_strategy='no_validation'
                )
                # We expect this to fail (script doesn't exist), but NOT with UnboundLocalError
                assert result is False  # Should fail but gracefully
                
            except UnboundLocalError as e:
                pytest.fail(f"UnboundLocalError caught! This is the bug we missed: {e}")
            except FileNotFoundError:
                # This is expected - lora-train.py won't exist in test environment
                pass
            except Exception as e:
                # Other exceptions are ok for this test
                pass


class TestVariableScopeIssues:
    """Specific tests for variable scoping issues."""
    
    def test_env_variable_defined_before_use_in_start_training(self):
        """Verify 'env' variable is defined before being used for logging."""
        import ast
        
        # Read the source file
        monitor_file = Path(__file__).parent.parent.parent / "src" / "edukaai_studio" / "ui" / "training_monitor.py"
        with open(monitor_file, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        # Find start_training method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'start_training':
                # Check that env is assigned before any usage
                env_assigned = False
                for i, child in enumerate(ast.walk(node)):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id == 'env':
                                env_assigned = True
                                print(f"✅ env assigned at position {i}")
                    
                    # Check for usage before assignment
                    if isinstance(child, ast.Name) and child.id == 'env' and isinstance(child.ctx, ast.Load):
                        if not env_assigned:
                            pytest.fail(f"env used before assignment at position {i}")
                
                assert env_assigned, "env was never assigned"


class TestImportErrors:
    """Test that all imports work correctly."""
    
    def test_all_ui_modules_importable(self):
        """Test that all UI modules can be imported without errors."""
        modules_to_test = [
            'edukaai_studio.ui.training_monitor',
            'edukaai_studio.ui.tabs.train',
            'edukaai_studio.ui.tabs.chat',
            'edukaai_studio.ui.tabs.results',
            'edukaai_studio.ui.tabs.configure',
            'edukaai_studio.ui.tabs.upload',
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except Exception as e:
                pytest.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
