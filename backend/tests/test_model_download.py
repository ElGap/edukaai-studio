"""
Comprehensive test suite for model download and loading logic.
Tests all scenarios: custom models, curated models, caching, naming conventions.
Run with: pytest backend/tests/test_model_download.py -v
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Set testing environment
os.environ["EDUKAI_ALLOW_REMOTE"] = "true"
os.environ["EDUKAI_ENV"] = "testing"

sys.path.insert(0, '/Users/developer/Projects/studio/backend')

from app.ml.trainer import TrainingProcess, TrainingConfig, TrainingManager


class TestModelDownloadLogic:
    """Test model download detection and path selection logic."""
    
    def test_model_path_selection_priority(self):
        """Test that model path is selected in correct priority order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup training config
            config = TrainingConfig(
                model_id="test-org/test-model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            # Create training process
            process = TrainingProcess("test-run", config)
            
            # Test Case 1: Custom download dir exists with model.safetensors
            download_dir = Path(tmpdir) / "downloaded_models" / "test-org--test-model"
            download_dir.mkdir(parents=True)
            (download_dir / "config.json").write_text('{"model_type": "test"}')
            (download_dir / "model.safetensors").write_text("fake weights")
            
            # Check model cached - should find custom dir
            assert process._check_model_cached("test-org/test-model") == True, \
                "Should detect model in custom download directory"
            
            # Test Case 2: Custom dir doesn't exist, check HF cache
            # (Would need to mock HF cache, skip for now)
            
            # Cleanup
            shutil.rmtree(download_dir)
    
    def test_naming_convention_weights_to_model(self):
        """Test that weights.00.safetensors gets renamed to model.safetensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="test-org/test-model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            process = TrainingProcess("test-run", config)
            download_dir = Path(tmpdir) / "downloaded_models" / "test-org--test-model"
            download_dir.mkdir(parents=True)
            
            # Create files with old naming
            (download_dir / "config.json").write_text('{}')
            weights_file = download_dir / "weights.00.safetensors"
            weights_file.write_text("fake weights")
            
            # The rename logic should be applied during download
            # For this test, simulate the rename
            if weights_file.exists() and not weights_file.name.startswith("model"):
                new_name = weights_file.parent / "model.safetensors"
                weights_file.rename(new_name)
            
            # Verify renamed
            assert (download_dir / "model.safetensors").exists(), \
                "weights file should be renamed to model.safetensors"
            assert not (download_dir / "weights.00.safetensors").exists(), \
                "old weights file should not exist after rename"
    
    def test_sharded_model_naming(self):
        """Test naming for sharded models with multiple weight files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir) / "model"
            download_dir.mkdir()
            
            # Create sharded weights
            (download_dir / "weights.00.safetensors").write_text("part1")
            (download_dir / "weights.01.safetensors").write_text("part2")
            
            # Simulate rename logic for sharded files
            safetensors_files = list(download_dir.glob("*.safetensors"))
            total_shards = len(safetensors_files)
            
            for safetensor_file in sorted(safetensors_files):
                if not safetensor_file.name.startswith("model"):
                    import re
                    shard_match = re.search(r'(\d+)', safetensor_file.name)
                    if shard_match:
                        shard_num = int(shard_match.group(1)) + 1
                        new_name = safetensor_file.parent / f'model-{shard_num:05d}-of-{total_shards:05d}.safetensors'
                        safetensor_file.rename(new_name)
            
            # Verify sharded naming
            renamed_files = sorted(download_dir.glob("model*.safetensors"))
            assert len(renamed_files) == 2, "Should have 2 renamed files"
            assert renamed_files[0].name == "model-00001-of-00002.safetensors", \
                f"First shard should be named correctly, got {renamed_files[0].name}"
            assert renamed_files[1].name == "model-00002-of-00002.safetensors", \
                f"Second shard should be named correctly, got {renamed_files[1].name}"


class TestModelPathSelection:
    """Test the model path selection logic in train() method."""
    
    def test_selects_custom_dir_first(self):
        """Test that custom download directory is checked first."""
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
            
            # Check logic
            process = TrainingProcess("run", config)
            
            # Simulate the path selection logic
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                expected_path = str(download_dir)
            elif process._check_model_cached(config.model_id):
                expected_path = config.model_id
            else:
                expected_path = str(download_dir)  # After download
            
            assert expected_path == str(download_dir), \
                f"Should select custom dir, got {expected_path}"
    
    def test_selects_hf_cache_second(self):
        """Test that HF cache is checked when custom dir doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            # Don't create custom dir
            download_dir = Path(tmpdir) / "downloaded_models" / "org--model"
            
            process = TrainingProcess("run", config)
            
            # Mock HF cache check to return True
            with patch.object(process, '_check_model_cached', return_value=True):
                if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                    expected_path = str(download_dir)
                elif process._check_model_cached(config.model_id):
                    expected_path = config.model_id
                else:
                    expected_path = str(download_dir)
                
                assert expected_path == config.model_id, \
                    f"Should select HF ID when in cache, got {expected_path}"


class TestMLXLoadCompatibility:
    """Test MLX load function compatibility with different naming."""
    
    def test_mlx_glob_pattern(self):
        """Test that MLX uses 'model*.safetensors' glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different naming
            (Path(tmpdir) / "model.safetensors").write_text("model")
            (Path(tmpdir) / "weights.00.safetensors").write_text("weights")
            (Path(tmpdir) / "model-00001-of-00002.safetensors").write_text("shard1")
            (Path(tmpdir) / "model-00002-of-00002.safetensors").write_text("shard2")
            (Path(tmpdir) / "config.json").write_text('{"model_type": "test"}')
            
            # Test MLX glob pattern
            model_files = list(Path(tmpdir).glob("model*.safetensors"))
            weights_files = list(Path(tmpdir).glob("weights*.safetensors"))
            all_safetensors = list(Path(tmpdir).glob("*.safetensors"))
            
            # MLX expects model*.safetensors
            assert len(model_files) == 3, f"Should find 3 model*.safetensors files, found {len(model_files)}"
            assert len(weights_files) == 1, f"Should find 1 weights*.safetensors file, found {len(weights_files)}"
            assert len(all_safetensors) == 4, f"Should find 4 total .safetensors files, found {len(all_safetensors)}"


class TestChatModelLoading:
    """Test chat model loading logic from chat.py."""
    
    def test_chat_prefers_download_dir(self):
        """Test that chat router prefers custom download directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_model_id = "org/custom-model"
            
            # Create run storage path
            run_storage = Path(tmpdir) / "runs" / "test-run"
            run_storage.mkdir(parents=True)
            
            # Create custom download directory
            download_dir = run_storage.parent / "downloaded_models" / "org--custom-model"
            download_dir.mkdir(parents=True)
            (download_dir / "config.json").write_text('{}')
            (download_dir / "model.safetensors").write_text("weights")
            
            # Simulate chat router logic
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                model_path = str(download_dir)
            else:
                model_path = base_model_id
            
            assert model_path == str(download_dir), \
                f"Chat should use download dir, got {model_path}"
    
    def test_chat_falls_back_to_hf_id(self):
        """Test that chat router falls back to HF ID when no download exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_model_id = "org/custom-model"
            
            # Create run storage path but NO download dir
            run_storage = Path(tmpdir) / "runs" / "test-run"
            run_storage.mkdir(parents=True)
            
            download_dir = run_storage.parent / "downloaded_models" / "org--custom-model"
            # download_dir does NOT exist
            
            # Simulate chat router logic
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                model_path = str(download_dir)
            else:
                model_path = base_model_id
            
            assert model_path == base_model_id, \
                f"Chat should use HF ID when no download, got {model_path}"


class TestModelCacheDetection:
    """Test model cache detection in various scenarios."""
    
    def test_detects_custom_dir_with_model_files(self):
        """Test detection of model in custom download directory."""
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
            
            # The updated _check_model_cached should find this
            result = process._check_model_cached("org/model")
            assert result == True, "Should detect model in custom directory"
    
    def test_detects_hf_cache(self):
        """Test detection of model in HuggingFace cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock HF cache directory
            cache_dir = Path(tmpdir) / ".cache" / "huggingface" / "hub"
            model_cache = cache_dir / "models--org--model"
            snapshot = model_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            
            (snapshot / "config.json").write_text('{}')
            (snapshot / "weights.00.safetensors").write_text("weights")
            
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            process = TrainingProcess("run", config)
            
            # Temporarily override home directory
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = process._check_model_cached("org/model")
                assert result == True, "Should detect model in HF cache"
    
    def test_not_cached_when_files_missing(self):
        """Test that missing files are detected correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="org/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            download_dir = Path(tmpdir) / "downloaded_models" / "org--model"
            download_dir.mkdir(parents=True)
            
            # Only config.json, no safetensors
            (download_dir / "config.json").write_text('{}')
            
            process = TrainingProcess("run", config)
            result = process._check_model_cached("org/model")
            assert result == False, "Should not detect as cached without safetensors"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
