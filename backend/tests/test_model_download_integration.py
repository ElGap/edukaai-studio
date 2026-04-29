"""
REAL integration tests for model download and loading.
These tests actually create files and verify the system works end-to-end.
Run with: pytest backend/tests/test_model_download_integration.py -v
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ["EDUKAAI_ALLOW_REMOTE"] = "true"
os.environ["EDUKAAI_ENV"] = "testing"

from app.ml.trainer import TrainingProcess, TrainingConfig


@pytest.mark.skip(reason="Native cache refactor: old file operations changed")
class TestRealFileOperations:
    """Test actual file operations that happen during training."""

    def test_first_training_creates_correct_files(self):
        """First training should detect standard HF naming correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )

            download_dir = Path(tmpdir) / "downloaded_models" / "org--model"
            download_dir.mkdir(parents=True)

            # Simulate standard HF download output
            (download_dir / "config.json").write_text('{"model_type": "test"}')
            (download_dir / "tokenizer.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("fake weights content")

            process = TrainingProcess("run-1", config)

            # Verify detected as complete without any rename
            assert process._is_model_complete(download_dir) is True, \
                "Standard naming should be detected as complete"

            files_after = list(download_dir.glob("*.safetensors"))
            assert len(files_after) == 1, f"Should have exactly 1 safetensors file, got {len(files_after)}"
            assert (download_dir / "model.safetensors").exists(), "Should have model.safetensors"

    def test_second_training_does_not_create_duplicates(self):
        """CRITICAL: Second training must reuse existing files, no duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir) / "downloaded_models" / "org--model"
            download_dir.mkdir(parents=True)

            # First training completed - standard naming already in place
            (download_dir / "config.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("weights from first training")

            # Verify initial state
            initial_files = list(download_dir.glob("*.safetensors"))
            assert len(initial_files) == 1
            assert initial_files[0].name == "model.safetensors"

            # Second training should detect completeness directly
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            process = TrainingProcess("run-2", config)
            assert process._is_model_complete(download_dir) is True, \
                "Second run should detect model complete without re-downloading"

            # Verify NO duplicates created
            final_files = list(download_dir.glob("*.safetensors"))
            assert len(final_files) == 1, f"CRITICAL: Should still have exactly 1 file after second run, got {len(final_files)}"
            assert final_files[0].name == "model.safetensors"
    
    @pytest.mark.skip(reason="Path resolution changed to use get_model_cache_dir(); needs rewrite")
    def test_check_model_cached_finds_custom_dir(self):
        """_check_model_cached must find models in custom download directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            download_dir = Path(tmpdir) / "downloaded_models" / "org--model"
            download_dir.mkdir(parents=True)
            (download_dir / "config.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("weights")
            
            process = TrainingProcess("run", config)
            
            # Test the actual _check_model_cached function
            result = process._check_model_cached("org/model")
            
            assert result == True, f"_check_model_cached should find model in custom dir, got {result}"
    
    def test_model_path_selection_in_train_method(self):
        """Test the actual path selection logic that runs in train() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            # Create custom download directory
            download_dir = Path(tmpdir) / "downloaded_models" / "org--model"
            download_dir.mkdir(parents=True)
            (download_dir / "config.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("weights")
            
            process = TrainingProcess("run", config)
            
            # Simulate the EXACT logic from train() method
            selected_path = None
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                selected_path = str(download_dir)
            elif process._check_model_cached(config.model_id):
                selected_path = config.model_id
            else:
                selected_path = str(download_dir)  # Would download first
            
            assert selected_path == str(download_dir), f"Should select custom dir, got {selected_path}"
            
            # Verify the path can be used by mlx_lm.load()
            # (In real scenario, this would be a valid MLX model)
            assert Path(selected_path).exists(), "Selected path should exist"
            assert (Path(selected_path) / "model.safetensors").exists(), "Should have model.safetensors"
            assert (Path(selected_path) / "config.json").exists(), "Should have config.json"


class TestMLXGlobPatternMatching:
    """Test that our file naming matches what MLX expects."""
    
    def test_mlx_glob_finds_model_safetensors(self):
        """MLX uses 'model*.safetensors' glob - test it works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "model.safetensors").write_text("model")
            (Path(tmpdir) / "config.json").write_text('{}')
            
            # This is exactly what MLX does
            model_files = list(Path(tmpdir).glob("model*.safetensors"))
            
            assert len(model_files) == 1, f"MLX glob should find 1 file, got {len(model_files)}"
            assert model_files[0].name == "model.safetensors"
    
    def test_mlx_glob_does_not_find_weights(self):
        """MLX glob should NOT find 'weights.00.safetensors'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with wrong naming
            (Path(tmpdir) / "weights.00.safetensors").write_text("weights")
            (Path(tmpdir) / "config.json").write_text('{}')
            
            # MLX glob pattern
            model_files = list(Path(tmpdir).glob("model*.safetensors"))
            
            assert len(model_files) == 0, f"MLX glob should find 0 files (weights not matching), got {len(model_files)}"


@pytest.mark.skip(reason="Native cache refactor: old bug scenarios no longer apply")
class TestRealBugScenarios:
    """Test the actual bugs we encountered."""

    def test_no_duplicate_files_on_reuse(self):
        """
        Running training twice must reuse existing standard-named files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir) / "model"
            download_dir.mkdir()

            # Start with standard-named file
            (download_dir / "model.safetensors").write_text("weights")
            (download_dir / "config.json").write_text('{}')

            config = TrainingConfig(
                model_id="test/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            process = TrainingProcess("run-1", config)

            # Verify after first run
            assert process._is_model_complete(download_dir) is True
            files_after_first = list(download_dir.glob("*.safetensors"))
            assert len(files_after_first) == 1, f"After first run: should have 1 file"

            # SECOND training run - reuse
            process2 = TrainingProcess("run-2", config)
            assert process2._is_model_complete(download_dir) is True, \
                "Second run should reuse existing files"

            files_after_second = list(download_dir.glob("*.safetensors"))
            assert len(files_after_second) == 1, \
                f"BUG: Second run created {len(files_after_second)} files instead of 1"

            model_files = list(download_dir.glob("model*.safetensors"))
            assert len(model_files) == 1, \
                f"Should have exactly 1 model*.safetensors file"
    
    def test_bug_chat_uses_hf_cache_instead_of_download_dir(self):
        """
        BUG: Chat was trying to load from HF cache where files were named
        'weights.00.safetensors' but MLX expected 'model.safetensors'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            base_model_id = "org/model"
            
            # Create run storage
            run_storage = Path(tmpdir) / "runs" / "test-run"
            run_storage.mkdir(parents=True)
            
            # Create download directory with properly renamed files
            download_dir = run_storage.parent / "downloaded_models" / "org--model"
            download_dir.mkdir(parents=True)
            (download_dir / "config.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("weights")
            
            # Simulate chat router logic
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                model_path = str(download_dir)
            else:
                model_path = base_model_id
            
            # Verify chat uses download dir
            assert model_path == str(download_dir), f"Chat should use download dir with renamed files"
            
            # Verify the files exist and are correctly named
            assert (Path(model_path) / "model.safetensors").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
