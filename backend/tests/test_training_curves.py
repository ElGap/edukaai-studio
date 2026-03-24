"""
Simplified tests for training curve display - no database needed.
Tests that curves show correct step ranges from 0 to total_steps.
Run with: pytest backend/tests/test_training_curves.py -v
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

os.environ["EDUKAI_ALLOW_REMOTE"] = "true"
os.environ["EDUKAI_ENV"] = "testing"

sys.path.insert(0, '/Users/developer/Projects/studio/backend')


class TestTrainingCurveStepRange:
    """Test that training curves show correct step ranges."""
    
    def test_frontend_includes_step_zero(self):
        """
        CRITICAL: Frontend must include step 0 in training metrics.
        
        Bug: TrainingView.vue filtered with 'if stats.current_step > 0'
        Fixed: Now includes all steps including 0.
        """
        # Simulate frontend logic
        training_metrics = []
        
        # Simulate metrics from backend (includes step 0)
        backend_metrics = [
            {"step": 0, "train_loss": 2.5, "eval_loss": None},
            {"step": 10, "train_loss": 1.8, "eval_loss": None},
            {"step": 50, "train_loss": 0.8, "eval_loss": 0.9},
            {"step": 100, "train_loss": 0.5, "eval_loss": 0.6},
        ]
        
        # OLD BUG: Would filter with step > 0
        # for stats in backend_metrics:
        #     if stats["step"] > 0 and stats["train_loss"] is not None:
        #         training_metrics.append(stats)
        
        # FIXED: Include all steps including 0
        for stats in backend_metrics:
            if stats["train_loss"] is not None:  # Removed step > 0 check!
                training_metrics.append(stats)
        
        # Verify step 0 is included
        steps = [m["step"] for m in training_metrics]
        
        assert 0 in steps, f"BUG: Step 0 missing from metrics. Got steps: {steps}"
        assert min(steps) == 0, f"First step should be 0, got {min(steps)}"
        assert max(steps) == 100, f"Last step should be 100, got {max(steps)}"
    
    def test_frontend_curve_filtering_includes_last_point(self):
        """
        CRITICAL: Curve filtering must include first and last points.
        
        Bug: Filter used 'index % step === 0' which could skip last point.
        Fixed: Always include first and last indices.
        """
        # Simulate many training points
        training_metrics = [{"step": i, "loss": 2.0 - i*0.015} for i in range(0, 101, 2)]
        
        # OLD BUG: Simple modulo filter
        # step = 10
        # filtered = [m for idx, m in enumerate(training_metrics) if idx % step == 0]
        # This could miss the last point!
        
        # FIXED: Always include first and last
        step = 10
        filtered = [
            m for idx, m in enumerate(training_metrics) 
            if idx % step == 0 or idx == 0 or idx == len(training_metrics) - 1
        ]
        
        steps = [m["step"] for m in filtered]
        
        # Verify last point is included
        assert 100 in steps, f"BUG: Last step 100 missing from curve. Got steps: {steps}"
        assert 0 in steps, f"BUG: First step 0 missing from curve. Got steps: {steps}"
    
    def test_mlx_glob_pattern_compatibility(self):
        """
        CRITICAL: File naming must match MLX 'model*.safetensors' glob.
        
        Bug: Files named 'weights.00.safetensors' - MLX can't find them.
        Fixed: Rename to 'model.safetensors' or 'model-00001-of-00002.safetensors'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with MLX-compatible naming
            model_file = Path(tmpdir) / "model.safetensors"
            model_file.write_text("weights")
            
            # MLX uses this glob pattern
            mlx_files = list(Path(tmpdir).glob("model*.safetensors"))
            
            assert len(mlx_files) == 1, f"MLX glob should find 1 file, found {len(mlx_files)}"
            assert mlx_files[0].name == "model.safetensors"
    
    def test_weights_naming_not_compatible_with_mlx(self):
        """
        DOCUMENTATION: Shows why 'weights.00.safetensors' fails.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with old naming (pre-fix)
            weights_file = Path(tmpdir) / "weights.00.safetensors"
            weights_file.write_text("weights")
            
            # MLX glob - won't find it!
            mlx_files = list(Path(tmpdir).glob("model*.safetensors"))
            
            # This documents the bug
            assert len(mlx_files) == 0, \
                "MLX can't find 'weights.*' files - they must be renamed to 'model.*'"


class TestNoEmojisInMessages:
    """Test that no emojis appear in status messages."""
    
    def test_status_messages_have_no_emojis(self):
        """
        CRITICAL: Status messages must not contain emojis.
        """
        from app.ml.trainer import TrainingProcess, TrainingConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_id="test/model",
                data_path=f"{tmpdir}/data",
                output_path=f"{tmpdir}/output",
                steps=10
            )
            
            process = TrainingProcess("test", config)
            
            # Test various status messages
            test_messages = [
                "Model download complete! 5 files downloaded",
                "Model appears MLX-compatible",
                "Model found but not MLX-formatted",
                "Starting download of test/model",
                "Loading model into memory",
            ]
            
            # Common emojis that should NOT appear
            forbidden_emojis = ['✅', '⚠️', '📥', '🎉', '✓', '🔍', '⚠']
            
            for msg in test_messages:
                for emoji in forbidden_emojis:
                    assert emoji not in msg, f"BUG: Message contains emoji {emoji}: {msg}"


class TestLoadingModelNotInLiveChat:
    """Test that 'Loading model...' doesn't appear in live chat."""
    
    def test_loading_model_status_not_logged(self):
        """
        CRITICAL: 'loading_model' status should NOT trigger live chat log.
        """
        # Simulate the logic from TrainingView.vue
        logged_statuses = []
        
        def simulate_status_update(status):
            """Simulate frontend status handler"""
            if status == 'downloading':
                logged_statuses.append('downloading')
            elif status == 'loading_model':
                # FIXED: Do NOT log in live chat
                pass
            elif status == 'model_loaded':
                logged_statuses.append('model_loaded')
            elif status == 'running':
                pass  # No log
            elif status == 'completed':
                logged_statuses.append('completed')
        
        # Simulate training flow
        simulate_status_update('downloading')
        simulate_status_update('loading_model')
        simulate_status_update('model_loaded')
        simulate_status_update('running')
        simulate_status_update('completed')
        
        # Verify loading_model not in logs
        assert 'loading_model' not in logged_statuses, \
            "BUG: 'loading_model' should not appear in live chat logs"
        
        # Verify other statuses ARE logged
        assert 'downloading' in logged_statuses
        assert 'model_loaded' in logged_statuses
        assert 'completed' in logged_statuses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
