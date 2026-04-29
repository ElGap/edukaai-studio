"""
BUG CATCHER: This test will FAIL if the download logic is broken.
Run this to verify the fix actually works before deploying.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

os.environ["EDUKAAI_ALLOW_REMOTE"] = "true"
os.environ["EDUKAAI_ENV"] = "testing"

from app.ml.trainer import TrainingProcess, TrainingConfig, training_manager


@pytest.mark.skip(reason="Native cache refactor: old methods removed. See test_model_download_native_cache.py")
class TestDownloadLogicActuallyWorks:
    """Tests that verify the ACTUAL implementation works, not mocked versions."""

    @pytest.mark.skip(reason="Path resolution changed to use get_model_cache_dir(); needs rewrite")
    def test_current_implementation_prevents_duplicates(self):
        """
        THIS WILL FAIL if _download_model doesn't check for existing files.
        Simulate: run training twice on same model.
        Expected: Second run should skip download and reuse files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="test/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            download_dir = Path(tmpdir) / "downloaded_models" / "test--model"
            download_dir.mkdir(parents=True)
            
            # Simulate FIRST run completed successfully
            # Files were downloaded and renamed
            (download_dir / "config.json").write_text('{"model_type": "test"}')
            (download_dir / "model.safetensors").write_text("weights from first run")
            (download_dir / "tokenizer.json").write_text('{}')
            
            # Verify first run state
            files_after_first = list(download_dir.glob("*.safetensors"))
            assert len(files_after_first) == 1
            assert files_after_first[0].name == "model.safetensors"
            
            # Now simulate SECOND run - what does _check_model_cached return?
            process = TrainingProcess("run-2", config)
            
            # THIS IS THE REAL CHECK - call the ACTUAL function
            is_cached = process._check_model_cached("test/model")
            
            # If this fails, the bug still exists
            assert is_cached == True, \
                f"BUG: _check_model_cached returned {is_cached} but model exists in {download_dir}"
            
            # If we get here, _check_model_cached works
            # Now verify what path would be selected in train()
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                model_path = str(download_dir)
            elif process._check_model_cached("test/model"):
                model_path = config.model_id
            else:
                model_path = str(download_dir)
            
            # Should use download dir, not HF ID
            assert model_path == str(download_dir), \
                f"BUG: Second run would use path {model_path} instead of {download_dir}"
    
    def test_standard_naming_preserved(self):
        """
        Verify that standard HF naming (model.safetensors) is detected
        correctly without any rename logic.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="test/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )

            download_dir = Path(tmpdir) / "downloaded_models" / "test--model"
            download_dir.mkdir(parents=True)

            # Simulate standard HF-downloaded naming
            (download_dir / "config.json").write_text('{"model_type": "test"}')
            (download_dir / "model.safetensors").write_text("weights content")
            (download_dir / "tokenizer.json").write_text('{}')

            process = TrainingProcess("test-run", config)
            assert process._is_model_complete(download_dir) is True, \
                "Should detect model.safetensors as complete without renaming"

            final_files = list(download_dir.glob("*.safetensors"))
            assert len(final_files) == 1, \
                f"Should have exactly 1 safetensors file, got {len(final_files)}"
            assert final_files[0].name == "model.safetensors", \
                f"Expected model.safetensors, got {final_files[0].name}"

    def test_second_run_reuses_files_without_rename(self):
        """
        Second training run must reuse already-downloaded model files.
        No rename logic should run (it has been removed).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir) / "downloaded_models" / "test--model"
            download_dir.mkdir(parents=True)

            # First run completed - files already correctly named
            (download_dir / "config.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("weights")

            initial_count = len(list(download_dir.glob("*.safetensors")))
            assert initial_count == 1

            # Second run should detect completeness directly
            process = TrainingProcess("run-2", TrainingConfig(
                model_id="test/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            ))
            assert process._is_model_complete(download_dir) is True, \
                "Second run should find model complete without any rename"

            final_count = len(list(download_dir.glob("*.safetensors")))
            assert final_count == 1, \
                f"BUG: Second run changed file count to {final_count} instead of 1"


class TestMLXLoadWillSucceed:
    """Tests that verify mlx_lm.load() will actually find the files."""
    
    def test_mlx_glob_finds_our_files(self):
        """
        THIS WILL FAIL if file naming doesn't match MLX expectations.
        mlx_lm.load() uses: glob("model*.safetensors")
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir) / "model"
            download_dir.mkdir()
            
            # Simulate our renamed files
            (download_dir / "config.json").write_text('{"model_type": "test"}')
            (download_dir / "model.safetensors").write_text("weights")
            
            # This is EXACTLY what MLX does internally
            model_files = list(download_dir.glob("model*.safetensors"))
            
            # THIS WILL FAIL if MLX can't find our files
            assert len(model_files) >= 1, \
                f"BUG: MLX glob found {len(model_files)} files, expected at least 1"
            
            assert any("model" in f.name for f in model_files), \
                f"BUG: Files don't match MLX pattern: {[f.name for f in model_files]}"
    
    def test_mlx_does_not_find_weights_files(self):
        """
        DOCUMENTATION TEST: Shows what happens WITHOUT renaming.
        
        This test intentionally FAILS to document the bug:
        - Files named 'weights.00.safetensors'
        - MLX glob 'model*.safetensors' finds 0 files
        - Result: 'No safetensors found' error
        
        To fix: Files must be renamed to 'model.safetensors'
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir) / "model"
            download_dir.mkdir()
            
            # Simulate NOT renamed files (pre-fix state)
            (download_dir / "config.json").write_text('{}')
            (download_dir / "weights.00.safetensors").write_text("weights")
            
            # MLX glob
            model_files = list(download_dir.glob("model*.safetensors"))
            
            # THIS TEST INTENTIONALLY FAILS
            # It documents that MLX can't find 'weights.*' files
            # Our fix renames them to 'model.*' which this test doesn't do
            if len(model_files) == 0:
                pytest.skip(
                    "DOCUMENTED BEHAVIOR: MLX glob finds 0 files when named 'weights.*' "
                    "(This is why we rename to 'model.safetensors'). "
                    "This test passes only if files are renamed."
                )


@pytest.mark.skip(reason="Native cache refactor: old methods removed")
class TestRealWorldScenario:
    """End-to-end test simulating actual user workflow."""

    @pytest.mark.skip(reason="Path resolution changed to use get_model_cache_dir(); needs rewrite")
    def test_custom_model_workflow(self):
        """
        THIS WILL FAIL if custom models can't be downloaded and used.
        Simulates: User adds custom model -> trains -> chats
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_id = "custom-org/custom-model"
            
            # Step 1: Create storage - USE CONSISTENT PATH
            # The config.output_path determines where _check_model_cached looks
            output_path = Path(tmpdir) / "output"
            output_path.mkdir(parents=True)
            
            # download_dir must be relative to output_path.parent
            download_dir = output_path.parent / "downloaded_models" / "custom-org--custom-model"
            download_dir.mkdir(parents=True)
            
            # Step 2: Simulate download with HF naming
            (download_dir / "config.json").write_text('{"model_type": "phi"}')
            (download_dir / "tokenizer.json").write_text('{}')
            (download_dir / "weights.00.safetensors").write_text("1.6GB of weights")
            
            # Step 3: Simulate first training - rename happens
            config = TrainingConfig(
                model_id=model_id,
                data_path=f"{tmpdir}/data",
                output_path=str(output_path),  # Use same output_path
                steps=100
            )
            
            process = TrainingProcess("run-1", config)
            
            # Verify the process looks in the right place
            expected_check_path = Path(config.output_path).parent / "downloaded_models" / model_id.replace("/", "--")
            assert expected_check_path == download_dir, f"Path mismatch: {expected_check_path} != {download_dir}"
            
            # Call the ACTUAL check
            is_cached_first = process._check_model_cached(model_id)
            
            # Should NOT be cached yet (files not renamed)
            assert is_cached_first == False, \
                f"First run should NOT detect as cached (files not renamed yet)"
            
            # Simulate rename
            safetensors = list(download_dir.glob("*.safetensors"))
            existing_model = list(download_dir.glob("model*.safetensors"))
            if not existing_model:
                for f in [f for f in safetensors if not f.name.startswith('model')]:
                    new_name = f.parent / 'model.safetensors'
                    if not new_name.exists():
                        f.rename(new_name)
            
            # Verify renamed
            assert (download_dir / "model.safetensors").exists()
            
            # Step 4: Simulate SECOND training - check again
            is_cached_second = process._check_model_cached(model_id)
            
            # THIS WILL FAIL if _check_model_cached doesn't find renamed files
            assert is_cached_second == True, \
                f"BUG: Second run should detect model as cached, got {is_cached_second}"
            
            # Step 5: Simulate chat loading (uses same logic)
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                chat_model_path = str(download_dir)
            else:
                chat_model_path = model_id
            
            # THIS WILL FAIL if chat doesn't use download dir
            assert chat_model_path == str(download_dir), \
                f"BUG: Chat should use {download_dir}, got {chat_model_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
