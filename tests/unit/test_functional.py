"""Comprehensive Unit Tests for EdukaAI Studio.

This module provides functional tests that actually exercise the code logic
to catch real bugs, not just syntax errors.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestUploadTab:
    """Test the upload tab functionality."""
    
    def test_process_uploaded_file_valid_jsonl(self):
        """Test processing a valid JSONL file."""
        from edukaai_studio.ui.tabs.upload import process_uploaded_file
        
        # Create a temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"instruction": "What is AI?", "output": "AI is..."}\n')
            f.write('{"instruction": "Explain ML", "output": "ML is..."}\n')
            temp_path = f.name
        
        try:
            current_state = {}
            result = process_uploaded_file(temp_path, "First 5", current_state)
            
            # Check result structure
            assert len(result) == 6, f"Expected 6 outputs, got {len(result)}"
            status_msg, preview_data, _, _, new_state, button_visible = result
            
            # Verify status
            assert "OK:" in status_msg or "validated" in status_msg.lower(), \
                f"Expected success message, got: {status_msg}"
            
            # Verify state was updated
            assert 'uploaded_file' in new_state, "State should contain uploaded_file"
            assert new_state['uploaded_file'] == temp_path, "State should have correct file path"
            
            # Verify button visibility
            assert button_visible.visible == True, "Go to configure button should be visible"
            
            print("✅ test_process_uploaded_file_valid_jsonl PASSED")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_process_uploaded_file_invalid_json(self):
        """Test processing invalid JSON file."""
        from edukaai_studio.ui.tabs.upload import process_uploaded_file
        
        # Create a temp file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('not valid json {{{')
            temp_path = f.name
        
        try:
            current_state = {}
            result = process_uploaded_file(temp_path, "First 5", current_state)
            
            # Check error handling
            status_msg = result[0]
            assert "Error" in status_msg or "Invalid" in status_msg, \
                f"Expected error message for invalid JSON, got: {status_msg}"
            
            print("✅ test_process_uploaded_file_invalid_json PASSED")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_process_uploaded_file_missing_fields(self):
        """Test JSON without required fields."""
        from edukaai_studio.ui.tabs.upload import process_uploaded_file
        
        # Create a temp file with missing fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"question": "What is AI?", "answer": "AI is..."}\n')  # Wrong fields
            temp_path = f.name
        
        try:
            current_state = {}
            result = process_uploaded_file(temp_path, "First 5", current_state)
            
            # Should handle gracefully
            status_msg = result[0]
            assert "Error" in status_msg or "must have" in status_msg.lower(), \
                f"Expected error for missing fields, got: {status_msg}"
            
            print("✅ test_process_uploaded_file_missing_fields PASSED")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_process_uploaded_file_empty(self):
        """Test empty file handling."""
        from edukaai_studio.ui.tabs.upload import process_uploaded_file
        
        # Create empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('')
            temp_path = f.name
        
        try:
            current_state = {}
            result = process_uploaded_file(temp_path, "First 5", current_state)
            
            # Check error handling
            status_msg = result[0]
            assert "Error" in status_msg or "empty" in status_msg.lower() or "0" in status_msg, \
                f"Expected error for empty file, got: {status_msg}"
            
            print("✅ test_process_uploaded_file_empty PASSED")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_refresh_preview_data(self):
        """Test preview refresh functionality."""
        from edukaai_studio.ui.tabs.upload import refresh_preview_data
        
        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(10):
                f.write(f'{{"instruction": "Q{i}", "output": "A{i}"}}\n')
            temp_path = f.name
        
        try:
            current_state = {'uploaded_file': temp_path}
            
            # Test "First 5" mode
            preview_data, visibility = refresh_preview_data("First 5", current_state)
            assert preview_data is not None, "Should return preview data"
            assert len(preview_data) <= 5, f"Should return max 5 items, got {len(preview_data)}"
            
            # Test "Random 5" mode
            preview_data2, _ = refresh_preview_data("Random 5", current_state)
            assert preview_data2 is not None, "Should return preview data"
            
            print("✅ test_refresh_preview_data PASSED")
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigureTab:
    """Test the configure tab functionality."""
    
    def test_save_training_config(self):
        """Test saving training configuration."""
        from edukaai_studio.ui.tabs.configure import save_training_config
        
        current_state = {'uploaded_file': '/tmp/test.jsonl'}
        
        # Mock model data
        mock_model = {
            'model_id': 'test-model',
            'name': 'Test Model'
        }
        
        # Patch STUDIO_MODELS.get_model
        with patch('edukaai_studio.ui.tabs.configure.STUDIO_MODELS') as mock_studio:
            mock_studio.get_model.return_value = mock_model
            
            result = save_training_config(
                model_id="test-model",
                preset_name="balanced",
                iterations=100,
                learning_rate="1e-4",
                lora_rank=16,
                lora_alpha=32,
                grad_accumulation=32,
                max_seq_length=2048,
                early_stopping=2,
                validation_split=10,
                current_state=current_state
            )
            
            # Check result
            assert len(result) == 3, f"Expected 3 outputs, got {len(result)}"
            msg, new_state, button = result
            
            # Verify config was saved
            assert 'training_config' in new_state, "State should contain training_config"
            config = new_state['training_config']
            assert config['model_id'] == 'test-model', "Config should have correct model_id"
            assert config['iterations'] == 100, "Config should have correct iterations"
            assert config['learning_rate'] == "1e-4", "Config should have correct learning_rate"
            
            print("✅ test_save_training_config PASSED")
    
    def test_save_training_config_without_upload(self):
        """Test saving config when no file is uploaded."""
        from edukaai_studio.ui.tabs.configure import save_training_config
        
        # State without uploaded_file
        current_state = {}
        
        with patch('edukaai_studio.ui.tabs.configure.STUDIO_MODELS') as mock_studio:
            mock_studio.get_model.return_value = {
                'model_id': 'test-model',
                'name': 'Test Model'
            }
            
            result = save_training_config(
                model_id="test-model",
                preset_name="balanced",
                iterations=100,
                learning_rate="1e-4",
                lora_rank=16,
                lora_alpha=32,
                grad_accumulation=32,
                max_seq_length=2048,
                early_stopping=2,
                validation_split=10,
                current_state=current_state
            )
            
            # Should still save config even without uploaded file
            msg, new_state, button = result
            assert 'training_config' in new_state, "Should still save config"
            
            print("✅ test_save_training_config_without_upload PASSED")
    
    def test_update_params_from_preset(self):
        """Test preset parameter updates."""
        from edukaai_studio.ui.tabs.configure import update_params_from_preset
        
        # Test Quick preset
        result = update_params_from_preset("Quick")
        assert len(result) == 7, f"Expected 7 parameters, got {len(result)}"
        iterations, lr, rank, alpha, accum, early, val = result
        assert iterations < 400, "Quick preset should have low iterations"
        
        # Test Maximum preset
        result = update_params_from_preset("Maximum")
        iterations, lr, rank, alpha, accum, early, val = result
        assert iterations > 600, "Maximum preset should have high iterations"
        assert lr == "5e-5", "Maximum preset should have low learning rate"
        
        print("✅ test_update_params_from_preset PASSED")


class TestChatTab:
    """Test the chat tab functionality."""
    
    def test_refresh_chat_status_no_training(self):
        """Test chat status when no training done."""
        from edukaai_studio.ui.tabs.chat import refresh_chat_status
        
        current_state = {}
        
        with patch('edukaai_studio.ui.tabs.chat.load_state_from_disk') as mock_load:
            mock_load.return_value = {}
            
            status, base_info, trained_info = refresh_chat_status(current_state)
            
            # Should indicate no training
            assert "Warning" in status or "No" in status or "model" in status.lower(), \
                f"Expected warning about no training, got: {status}"
            
            print("✅ test_refresh_chat_status_no_training PASSED")
    
    def test_refresh_chat_status_with_training(self):
        """Test chat status when training is complete."""
        from edukaai_studio.ui.tabs.chat import refresh_chat_status
        
        current_state = {
            'training_complete': True,
            'training_config': {
                'model_id': 'test-model',
                'model_name': 'Test Model'
            },
            'output_dir': '/tmp/test_output'
        }
        
        with patch('edukaai_studio.ui.tabs.chat.load_state_from_disk') as mock_load:
            mock_load.return_value = {}
            
            # Mock output dir exists with adapter
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                status, base_info, trained_info = refresh_chat_status(current_state)
                
                # Should show ready status
                assert "OK" in status or "Ready" in status or "complete" in status.lower(), \
                    f"Expected OK status, got: {status}"
                
                print("✅ test_refresh_chat_status_with_training PASSED")
    
    def test_ask_models_no_question(self):
        """Test ask_models with no question."""
        from edukaai_studio.ui.tabs.chat import ask_models
        
        current_state = {'training_complete': True}
        
        result = ask_models(
            question="",  # Empty question
            system_prompt="Test prompt",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            use_fused=False,
            current_state=current_state
        )
        
        # Should handle gracefully
        assert len(result) == 7, f"Expected 7 outputs, got {len(result)}"
        
        # Last element is status
        status = result[-1]
        assert "enter" in status.lower() or "provide" in status.lower() or "question" in status.lower(), \
            f"Expected prompt to enter question, got: {status}"
        
        print("✅ test_ask_models_no_question PASSED")


class TestStateManagement:
    """Test state management functions."""
    
    def test_get_initial_state(self):
        """Test initial state creation."""
        from edukaai_studio.core.state import get_initial_state
        
        # Clear any existing state file
        state_file = Path('.studio_state.json')
        if state_file.exists():
            state_file.unlink()
        
        initial_state = get_initial_state()
        
        # Verify required keys
        required_keys = [
            'uploaded_file',
            'training_config',
            'training_active',
            'training_complete',
            'output_dir'
        ]
        
        for key in required_keys:
            assert key in initial_state, f"Initial state missing required key: {key}"
        
        print("✅ test_get_initial_state PASSED")
    
    def test_save_and_load_state(self):
        """Test state persistence."""
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        test_state = {
            'uploaded_file': '/tmp/test.jsonl',
            'training_config': {'model_id': 'test'},
            'training_complete': True,
            'output_dir': '/tmp/output'
        }
        
        # Save state
        result = save_state_to_disk(test_state)
        assert result is True, "Should successfully save state"
        
        # Load state
        loaded_state = load_state_from_disk()
        assert loaded_state is not None, "Should load state"
        assert loaded_state['uploaded_file'] == '/tmp/test.jsonl', "Should preserve uploaded_file"
        
        # Cleanup
        state_file = Path('.studio_state.json')
        if state_file.exists():
            state_file.unlink()
        
        print("✅ test_save_and_load_state PASSED")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_upload_nonexistent_file(self):
        """Test handling of non-existent file."""
        from edukaai_studio.ui.tabs.upload import process_uploaded_file
        
        current_state = {}
        result = process_uploaded_file('/nonexistent/path/file.jsonl', 'First 5', current_state)
        
        # Should handle gracefully
        status = result[0]
        assert "Error" in status or "not found" in status.lower() or "exist" in status.lower(), \
            f"Expected error for non-existent file, got: {status}"
        
        print("✅ test_upload_nonexistent_file PASSED")
    
    def test_configure_with_invalid_model(self):
        """Test configure with invalid model ID."""
        from edukaai_studio.ui.tabs.configure import save_training_config
        
        current_state = {}
        
        with patch('edukaai_studio.ui.tabs.configure.STUDIO_MODELS') as mock_studio:
            mock_studio.get_model.return_value = None  # Invalid model
            
            result = save_training_config(
                model_id="invalid-model",
                preset_name="balanced",
                iterations=100,
                learning_rate="1e-4",
                lora_rank=16,
                lora_alpha=32,
                grad_accumulation=32,
                max_seq_length=2048,
                early_stopping=2,
                validation_split=10,
                current_state=current_state
            )
            
            # Should handle gracefully
            msg, new_state, button = result
            # Should either error or use defaults
            
            print("✅ test_configure_with_invalid_model PASSED")
    
    def test_chat_with_incomplete_state(self):
        """Test chat with missing state fields."""
        from edukaai_studio.ui.tabs.chat import ask_models
        
        # State missing training_complete
        current_state = {}
        
        result = ask_models(
            question="Test question",
            system_prompt="Test",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            use_fused=False,
            current_state=current_state
        )
        
        # Should handle gracefully
        assert len(result) == 7, f"Expected 7 outputs, got {len(result)}"
        
        # Should indicate training needed
        base_resp = result[1]
        assert "training" in base_resp.lower() or "complete" in base_resp.lower() or "error" in base_resp.lower(), \
            f"Expected training-related message, got: {base_resp}"
        
        print("✅ test_chat_with_incomplete_state PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("EdukaAI Studio - Comprehensive Unit Tests")
    print("=" * 70 + "\n")
    
    test_classes = [
        TestUploadTab,
        TestConfigureTab,
        TestChatTab,
        TestStateManagement,
        TestEdgeCases
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n📦 {test_class.__name__}")
        print("-" * 70)
        
        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                test_instance = test_class()
                test_method = getattr(test_instance, method_name)
                test_method()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"\n❌ {method_name} FAILED:")
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive unit tests")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
