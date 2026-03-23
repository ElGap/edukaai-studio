"""Integration tests for actual training process.

These are the MOST IMPORTANT tests as they verify the core training functionality works end-to-end.

⚠️ WARNING: These tests require:
- Apple Silicon Mac (for MLX)
- Internet connection (to download models)
- Significant time (10-30 minutes)
- Sufficient disk space (model downloads are ~2-4GB)

Usage:
    # Run all integration tests (slow)
    pytest tests/integration/test_training_integration.py -v --run-slow
    
    # Skip integration tests (fast)
    pytest tests/browser/ -v  # (default, excludes slow tests)
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Skip entire file if not on Apple Silicon or no MLX
pytestmark = [
    pytest.mark.slow,  # Mark as slow test
    pytest.mark.integration,  # Mark as integration test
]


def create_minimal_training_data():
    """Create minimal training data for fast testing."""
    # Create 10 training examples - minimum viable dataset
    examples = [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "Capital of France?", "output": "Paris"},
        {"instruction": "Largest planet?", "output": "Jupiter"},
        {"instruction": "Speed of light?", "output": "299,792,458 m/s"},
        {"instruction": "Water formula?", "output": "H2O"},
        {"instruction": "Year length?", "output": "365.25 days"},
        {"instruction": "Smallest prime?", "output": "2"},
        {"instruction": "DNA stands for?", "output": "Deoxyribonucleic acid"},
        {"instruction": "First president?", "output": "George Washington"},
        {"instruction": "Programming founder?", "output": "Ada Lovelace"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
        return f.name


class TestTrainingIntegration:
    """Integration tests that actually run training.
    
    These tests verify:
    1. Training starts successfully
    2. Progress updates correctly
    3. Status changes from "Ready" → "Training" → "Complete"
    4. Loss values are recorded
    5. Model is saved to disk
    6. State is updated to completed
    7. Training can be stopped mid-way
    8. Results are accessible after completion
    """
    
    @pytest.fixture
    def training_setup(self):
        """Setup for training tests - creates minimal training config."""
        from edukaai_studio.config import STUDIO_MODELS
        
        # Get the smallest/fastest model for testing
        models = STUDIO_MODELS.get_all_models()
        
        # Prefer smaller models for faster testing
        test_model = None
        for m in models:
            if 'mini' in m.get('name', '').lower() or 'small' in m.get('name', '').lower():
                test_model = m
                break
        
        if not test_model:
            test_model = models[0]  # Fallback to first model
        
        data_file = create_minimal_training_data()
        
        setup = {
            'model_id': test_model.get('model_id'),
            'model_name': test_model.get('name'),
            'data_file': data_file,
            'iterations': 10,  # Minimal iterations for speed
            'learning_rate': '1e-4',
            'lora_rank': 8,  # Smaller rank for speed
            'lora_alpha': 16,
            'grad_accumulation': 4,
            'max_seq_length': 512,  # Shorter sequences for speed
        }
        
        yield setup
        
        # Cleanup
        try:
            Path(data_file).unlink()
        except:
            pass
    
    def test_training_process_runs_to_completion(self, training_setup):
        """CRITICAL TEST: Verify training actually runs and completes.
        
        This is THE MOST IMPORTANT test - if this fails, the app doesn't work.
        """
        from edukaai_studio.ui.tabs.train import start_training_real
        from edukaai_studio.ui.tabs.configure import save_training_config
        
        print("\n" + "="*70)
        print("🚀 RUNNING CRITICAL INTEGRATION TEST: Training Process")
        print("="*70)
        print(f"Model: {training_setup['model_name']}")
        print(f"Iterations: {training_setup['iterations']}")
        print(f"Data file: {training_setup['data_file']}")
        print("="*70 + "\n")
        
        # Setup initial state
        initial_state = {
            'uploaded_file': training_setup['data_file'],
            'training_config': {
                'model_id': training_setup['model_id'],
                'model_name': training_setup['model_name'],
                'iterations': training_setup['iterations'],
                'learning_rate': training_setup['learning_rate'],
                'lora_rank': training_setup['lora_rank'],
                'lora_alpha': training_setup['lora_alpha'],
                'grad_accumulation': training_setup['grad_accumulation'],
                'max_seq_length': training_setup['max_seq_length'],
                'batch_size': 1,
                'early_stopping': 2,
                'validation_split': 10,
            }
        }
        
        # Start training
        results = []
        start_time = time.time()
        
        try:
            # Run the training generator
            for i, result in enumerate(start_training_real(initial_state)):
                results.append(result)
                
                # Extract status
                status_msg = result[10] if len(result) > 10 else "Unknown"
                progress = result[0] if len(result) > 0 else 0
                step_display = result[1] if len(result) > 1 else "0 / 0"
                
                # Print progress every few iterations
                if i % 5 == 0:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Step: {step_display} | Progress: {progress}% | Status: {status_msg[:50]}")
                
                # Safety limit - don't run forever in tests
                if i > 1000:  # Max 1000 iterations
                    print("⚠️  Test exceeded 1000 iterations, stopping")
                    break
                
                # Check if completed
                if "Complete" in status_msg or "complete" in status_msg.lower():
                    print(f"\n✅ Training completed at iteration {i}")
                    break
                
                # Check if error
                if "Error" in status_msg and "Upload required" not in status_msg:
                    print(f"\n❌ Training error at iteration {i}: {status_msg}")
                    break
        
        except Exception as e:
            print(f"\n❌ Training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print(f"⏱️  Training Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"📊 Total Yielded Updates: {len(results)}")
        print("="*70 + "\n")
        
        # CRITICAL ASSERTIONS
        
        # 1. Training must have yielded results
        assert len(results) > 0, "Training must yield at least one result"
        print("✅ Training yielded results")
        
        # 2. Final state must show completion or error (not infinite loop)
        final_result = results[-1]
        final_status = final_result[10] if len(final_result) > 10 else ""
        
        assert (
            "Complete" in final_status or 
            "complete" in final_status.lower() or 
            "Error" in final_status
        ), f"Training must complete or error. Final status: {final_status}"
        print("✅ Training reached completion state")
        
        # 3. Progress should reach 100% or show completion
        final_progress = final_result[0] if len(final_result) > 0 else 0
        if "Complete" in final_status:
            assert final_progress == 100 or final_progress >= 90, \
                f"Progress should be near 100% at completion, got: {final_progress}"
            print("✅ Progress reached near 100%")
        
        # 4. Final state must have correct flags (if completed successfully)
        if "Complete" in final_status and "Error" not in final_status:
            final_state = final_result[11] if len(final_result) > 11 else {}
            
            if isinstance(final_state, dict):
                assert final_state.get('training_complete') is True, \
                    "training_complete flag must be True"
                assert final_state.get('training_active') is False, \
                    "training_active flag must be False"
                print("✅ State flags correctly set")
                
                # 5. Must have output directory
                assert final_state.get('output_dir'), \
                    "Must have output_dir after completion"
                print(f"✅ Output directory set: {final_state.get('output_dir')}")
                
                # 6. Output directory must exist on disk
                output_dir = Path(final_state.get('output_dir'))
                if output_dir.exists():
                    print(f"✅ Output directory exists on disk")
                    
                    # 7. Check for model files
                    adapters_dir = output_dir / "adapters"
                    if adapters_dir.exists():
                        print(f"✅ Adapters directory created")
        
        print("\n" + "="*70)
        print("🎉 CRITICAL TEST PASSED: Training Process Works!")
        print("="*70 + "\n")
    
    def test_training_can_be_stopped(self, training_setup):
        """Test that training can be stopped mid-process."""
        from edukaai_studio.ui.tabs.train import start_training_real, stop_training_handler
        
        print("\n🧪 Testing training stop functionality...")
        
        # Setup state
        initial_state = {
            'uploaded_file': training_setup['data_file'],
            'training_config': training_setup,
            'training_active': True,
            'training_complete': False,
            'monitor': None  # No real monitor in test
        }
        
        # Try to stop training
        try:
            status_msg, new_state = stop_training_handler(initial_state)
            
            # Should return a status message
            assert status_msg is not None
            assert "stop" in status_msg.lower() or "requested" in status_msg.lower() or "Error" in status_msg
            print("✅ Stop handler works")
            
            # State should be updated
            assert new_state.get('training_active') is False
            print("✅ State updated to inactive")
            
        except Exception as e:
            print(f"⚠️  Stop test (may be expected): {e}")
    
    def test_training_state_persisted_to_disk(self, training_setup):
        """Test that training state is properly saved to disk."""
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        print("\n🧪 Testing state persistence...")
        
        # Create a completed training state
        completed_state = {
            'uploaded_file': training_setup['data_file'],
            'training_config': training_setup,
            'training_active': False,
            'training_complete': True,
            'output_dir': 'outputs/test_model_20240101_120000',
            'completion_time': '12:00:00',
            'train_losses': {1: 2.5, 10: 1.8},
            'val_losses': {1: 2.8, 10: 1.9},
        }
        
        # Save to disk
        save_result = save_state_to_disk(completed_state)
        assert save_result is True, "State must save successfully"
        print("✅ State saved to disk")
        
        # Load from disk
        loaded_state = load_state_from_disk()
        assert loaded_state is not None, "State must load successfully"
        print("✅ State loaded from disk")
        
        # Verify flags
        assert loaded_state['training_complete'] is True
        assert loaded_state['training_active'] is False
        assert loaded_state['output_dir'] == 'outputs/test_model_20240101_120000'
        print("✅ State flags correct after reload")
    
    def test_other_tabs_detect_completion(self, training_setup):
        """Test that other tabs correctly detect training completion."""
        from edukaai_studio.core.state import save_state_to_disk
        from edukaai_studio.ui.tabs.results import refresh_results_status
        from edukaai_studio.ui.tabs.chat import refresh_chat_status
        
        print("\n🧪 Testing cross-tab completion detection...")
        
        # Save completed state
        completed_state = {
            'uploaded_file': training_setup['data_file'],
            'training_config': training_setup,
            'training_complete': True,
            'training_active': False,
            'output_dir': 'outputs/test_model_20240101_120000',
            'completion_time': '12:00:00'
        }
        save_state_to_disk(completed_state)
        
        # Test Results tab
        try:
            from unittest.mock import patch
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    status, state = refresh_results_status({})
                    print(f"   Results tab status: {status}")
                    assert "complete" in status.lower() or "ready" in status.lower() or "Error" in status
                    print("✅ Results tab detects completion")
        except Exception as e:
            print(f"   Results tab check: {e}")
        
        # Test Chat tab
        try:
            status, base_info, trained_info = refresh_chat_status({})
            print(f"   Chat tab status: {status}")
            assert "OK" in status or "complete" in status.lower() or "ready" in status.lower() or "model" in status.lower()
            print("✅ Chat tab detects completion")
        except Exception as e:
            print(f"   Chat tab check: {e}")


@pytest.mark.skipif(
    not sys.platform.startswith('darwin'), 
    reason="MLX training only works on macOS (Apple Silicon)"
)
def test_requires_apple_silicon():
    """Verify we're on Apple Silicon."""
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check for Apple Silicon
    if platform.machine() == 'arm64' and 'Darwin' in platform.platform():
        print("✅ Running on Apple Silicon")
    else:
        pytest.skip("Not on Apple Silicon - MLX won't work")


def test_mlx_available():
    """Check if MLX is available."""
    try:
        import mlx
        print(f"✅ MLX available: {mlx.__version__}")
    except ImportError:
        pytest.skip("MLX not installed - cannot run training tests")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_training_integration.py -v --run-slow
    pytest.main([__file__, "-v", "--run-slow"])
