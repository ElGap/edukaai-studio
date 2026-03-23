"""Comprehensive tests for My Models functionality.

Tests the TrainedModelsRegistry and My Models tab including:
- Model registration and management
- Discovery from outputs/ directory
- Metadata extraction from training_summary.json
- Orphaned model detection and cleanup
- UI functions (scan, filter, format)
- Integration with training workflow
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edukaai_studio.core.trained_models_registry import (
    TrainedModelsRegistry,
    TrainedModel,
    get_registry,
    format_model_for_display,
    REGISTRY_FILE
)

from edukaai_studio.ui.tabs.my_models import (
    scan_for_models,
    filter_models,
    get_model_details,
    update_model_notes,
    update_model_tags,
    delete_model,
    load_model_for_chat,
    download_model_file,
    cleanup_orphaned_models
)


# ============ FIXTURES ============

@pytest.fixture
def mock_registry_file(tmp_path):
    """Create a temporary registry file for testing."""
    registry_file = tmp_path / "trained_models_registry.json"
    registry_file.write_text(json.dumps({"models": [], "version": "1.0"}))
    return registry_file


@pytest.fixture
def mock_outputs_dir(tmp_path):
    """Create a mock outputs directory with training results."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    
    # Create a complete training output
    model_dir = outputs_dir / "test_model_20260101_120000"
    model_dir.mkdir()
    
    # Create adapters directory
    adapters_dir = model_dir / "adapters"
    adapters_dir.mkdir()
    (adapters_dir / "adapters.safetensors").write_text("mock adapter weights")
    
    # Create training_summary.json
    summary = {
        "timestamp": "2026-01-01T12:00:00",
        "model": "mlx-community/test-model",
        "training_config": {
            "iterations": 100,
            "learning_rate": 0.0001,
            "batch_size": 1,
            "grad_accumulation_steps": 32
        },
        "validation_losses": {
            "10": 2.5,
            "50": 1.8,
            "100": 1.5
        },
        "best_iteration": 100,
        "best_val_loss": 1.5,
        "fusion_success": True
    }
    (model_dir / "training_summary.json").write_text(json.dumps(summary))
    
    return outputs_dir


@pytest.fixture
def sample_trained_model():
    """Create a sample TrainedModel for testing."""
    return TrainedModel(
        id="abc123def456",
        created_at="2026-01-01T12:00:00",
        updated_at="2026-01-01T12:30:00",
        base_model_id="mlx-community/Phi-3-mini",
        base_model_name="Phi-3 Mini",
        iterations=100,
        learning_rate="1e-4",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        batch_size=1,
        grad_accumulation=32,
        dataset_path="/path/to/data.jsonl",
        dataset_size=150,
        output_dir="outputs/Phi_3_mini_20260101_120000",
        best_loss=1.234,
        final_loss=1.234,
        best_iteration=100,
        train_losses={"10": 2.5, "50": 1.8, "100": 1.234},
        val_losses={"50": 1.9, "100": 1.234},
        training_duration_minutes=25.5,
        exports={
            "adapter": "outputs/Phi_3_mini_20260101_120000/adapters/adapters.safetensors",
            "fused": "outputs/Phi_3_mini_20260101_120000/fused_model",
            "gguf": None
        },
        status="completed",
        tags=["test", "phi3"],
        notes="Test model for unit testing"
    )


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Create an isolated registry for testing."""
    test_registry_file = tmp_path / "test_registry.json"
    test_registry_file.write_text(json.dumps({"models": [], "version": "1.0"}))
    
    # Patch the registry file location
    from edukaai_studio.core import trained_models_registry
    original_file = trained_models_registry.REGISTRY_FILE
    trained_models_registry.REGISTRY_FILE = test_registry_file
    
    # Reset the singleton
    trained_models_registry._registry = None
    
    yield test_registry_file
    
    # Restore original
    trained_models_registry.REGISTRY_FILE = original_file
    trained_models_registry._registry = None


# ============ TRAINED MODELS REGISTRY TESTS ============

class TestTrainedModelsRegistry:
    """Test TrainedModelsRegistry core functionality."""
    
    def test_registry_singleton(self, isolated_registry):
        """Test that get_registry returns singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
    
    def test_register_model(self, isolated_registry):
        """Test registering a new model."""
        registry = get_registry()
        
        model_data = {
            'output_dir': 'outputs/test_model_20260101_120000',
            'base_model_id': 'mlx-community/test-model',
            'base_model_name': 'Test Model',
            'iterations': 100,
            'learning_rate': '1e-4',
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.0,
            'batch_size': 1,
            'grad_accumulation': 32,
            'dataset_path': '/path/to/data.jsonl',
            'dataset_size': 150,
            'best_loss': 1.5,
            'final_loss': 1.5,
            'best_iteration': 100,
            'train_losses': {},
            'val_losses': {},
            'training_duration_minutes': 30.0,
            'exports': {'adapter': None, 'fused': None, 'gguf': None},
            'status': 'completed',
            'tags': [],
            'notes': 'Test model'
        }
        
        model_id = registry.register_model(model_data)
        assert model_id is not None
        assert len(model_id) == 12  # MD5 hash truncated
        
        # Verify model was saved
        model = registry.get_model(model_id)
        assert model is not None
        assert model.base_model_name == 'Test Model'
        assert model.iterations == 100
    
    def test_register_model_updates_existing(self, isolated_registry):
        """Test that registering same model updates instead of duplicating."""
        registry = get_registry()
        
        model_data = {
            'output_dir': 'outputs/test_model',
            'base_model_id': 'mlx-community/test',
            'base_model_name': 'Test',
            'iterations': 50,
            'status': 'running'
        }
        
        # Register first time
        model_id1 = registry.register_model(model_data)
        
        # Update and register again
        model_data['iterations'] = 100
        model_data['status'] = 'completed'
        model_id2 = registry.register_model(model_data)
        
        # Should be same ID (update, not duplicate)
        assert model_id1 == model_id2
        
        # Verify updated
        model = registry.get_model(model_id1)
        assert model.iterations == 100
        assert model.status == 'completed'
    
    def test_get_model_not_found(self, isolated_registry):
        """Test getting non-existent model returns None."""
        registry = get_registry()
        model = registry.get_model("nonexistent_id")
        assert model is None
    
    def test_list_models_empty(self, isolated_registry):
        """Test listing models when registry is empty."""
        registry = get_registry()
        models = registry.list_models()
        assert models == []
    
    def test_list_models_with_filter(self, isolated_registry):
        """Test listing models with status filter."""
        registry = get_registry()
        
        # Register completed model
        registry.register_model({
            'output_dir': 'outputs/completed',
            'base_model_id': 'test',
            'base_model_name': 'Completed',
            'iterations': 100,
            'status': 'completed'
        })
        
        # Register running model
        registry.register_model({
            'output_dir': 'outputs/running',
            'base_model_id': 'test',
            'base_model_name': 'Running',
            'iterations': 50,
            'status': 'running'
        })
        
        # Filter by completed
        completed = registry.list_models(filter_status='completed')
        assert len(completed) == 1
        assert completed[0].status == 'completed'
        
        # Filter by running
        running = registry.list_models(filter_status='running')
        assert len(running) == 1
        assert running[0].status == 'running'
    
    def test_update_model(self, isolated_registry):
        """Test updating model metadata."""
        registry = get_registry()
        
        # Register model
        model_id = registry.register_model({
            'output_dir': 'outputs/test',
            'base_model_id': 'test',
            'base_model_name': 'Test',
            'iterations': 50,
            'status': 'running',
            'tags': ['initial']
        })
        
        # Update tags
        result = registry.update_model(model_id, tags=['updated', 'test'])
        assert result is True
        
        # Verify update
        model = registry.get_model(model_id)
        assert model.tags == ['updated', 'test']
    
    def test_delete_model(self, isolated_registry):
        """Test deleting model from registry."""
        registry = get_registry()
        
        model_id = registry.register_model({
            'output_dir': 'outputs/test',
            'base_model_id': 'test',
            'base_model_name': 'Test',
            'iterations': 100
        })
        
        # Delete without removing files
        result = registry.delete_model(model_id, delete_files=False)
        assert result is True
        
        # Verify deleted
        assert registry.get_model(model_id) is None


# ============ MODEL DISCOVERY TESTS ============

class TestModelDiscovery:
    """Test model discovery from outputs directory."""
    
    def test_scan_for_new_models(self, monkeypatch, tmp_path):
        """Test scanning for new models in outputs directory."""
        # Create mock outputs directory
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        # Create a training output
        model_dir = outputs_dir / "test_model_20260101_120000"
        model_dir.mkdir()
        
        # Create adapter
        adapters_dir = model_dir / "adapters"
        adapters_dir.mkdir()
        (adapters_dir / "adapters.safetensors").write_text("mock")
        
        # Create summary
        summary = {
            "timestamp": "2026-01-01T12:00:00",
            "model": "mlx-community/test-model",
            "training_config": {"iterations": 100, "learning_rate": 0.0001},
            "validation_losses": {"100": 1.5},
            "best_iteration": 100,
            "best_val_loss": 1.5
        }
        (model_dir / "training_summary.json").write_text(json.dumps(summary))
        
        # Patch outputs directory
        original_cwd = Path.cwd()
        monkeypatch.chdir(tmp_path)
        
        try:
            from edukaai_studio.core.trained_models_registry import TrainedModelsRegistry
            registry = TrainedModelsRegistry()
            registry._data = {"models": [], "version": "1.0"}  # Clear existing
            
            # Scan for models
            new_models = registry.scan_for_new_models()
            
            assert len(new_models) == 1
            assert new_models[0].iterations == 100
            assert new_models[0].best_loss == 1.5
        finally:
            monkeypatch.chdir(original_cwd)
    
    def test_scan_skips_existing_models(self, monkeypatch, tmp_path):
        """Test that scan skips models already in registry."""
        # Setup same as above
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        model_dir = outputs_dir / "test_model"
        model_dir.mkdir()
        (model_dir / "adapters").mkdir()
        (model_dir / "adapters" / "adapters.safetensors").write_text("mock")
        
        summary = {"model": "test", "training_config": {"iterations": 100}}
        (model_dir / "training_summary.json").write_text(json.dumps(summary))
        
        original_cwd = Path.cwd()
        monkeypatch.chdir(tmp_path)
        
        try:
            from edukaai_studio.core.trained_models_registry import TrainedModelsRegistry
            registry = TrainedModelsRegistry()
            registry._data = {"models": [], "version": "1.0"}
            
            # First scan
            new_models_1 = registry.scan_for_new_models()
            assert len(new_models_1) == 1
            
            # Second scan should find nothing (already registered)
            new_models_2 = registry.scan_for_new_models()
            assert len(new_models_2) == 0
        finally:
            monkeypatch.chdir(original_cwd)
    
    def test_extract_metadata_from_summary(self, monkeypatch, tmp_path):
        """Test extracting metadata from training_summary.json."""
        from edukaai_studio.core.trained_models_registry import TrainedModelsRegistry
        
        registry = TrainedModelsRegistry()
        
        # Create summary file
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()
        
        summary_path = output_dir / "training_summary.json"
        summary = {
            "timestamp": "2026-01-01T12:00:00",
            "model": "mlx-community/Phi-3-mini",
            "training_config": {
                "iterations": 200,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "grad_accumulation_steps": 32
            },
            "validation_losses": {
                "50": 2.5,
                "100": 2.0,
                "150": 1.8,
                "200": 1.5
            },
            "best_iteration": 200,
            "best_val_loss": 1.5
        }
        summary_path.write_text(json.dumps(summary))
        
        # Extract metadata
        metadata = registry._extract_metadata_from_summary(summary_path, output_dir)
        
        assert metadata['base_model_id'] == "mlx-community/Phi-3-mini"
        assert metadata['iterations'] == 200
        assert metadata['learning_rate'] == "0.0001"
        assert metadata['best_loss'] == 1.5
        assert len(metadata['val_losses']) == 4


# ============ ORPHANED MODEL TESTS ============

class TestOrphanedModels:
    """Test orphaned model detection and cleanup."""
    
    def test_list_models_excludes_orphaned_by_default(self, isolated_registry, monkeypatch, tmp_path):
        """Test that list_models excludes orphaned models by default."""
        registry = get_registry()
        
        # Register model with non-existent output directory
        registry.register_model({
            'output_dir': str(tmp_path / 'non_existent_dir'),
            'base_model_id': 'test',
            'base_model_name': 'Orphaned',
            'iterations': 100,
            'status': 'completed'
        })
        
        # List without include_orphaned (default)
        models = registry.list_models()
        assert len(models) == 0  # Should be excluded
        
        # List with include_orphaned=True
        all_models = registry.list_models(include_orphaned=True)
        assert len(all_models) == 1
    
    def test_get_statistics_counts_orphaned(self, isolated_registry, monkeypatch, tmp_path):
        """Test that get_statistics counts orphaned models."""
        registry = get_registry()
        
        # Register orphaned model
        registry.register_model({
            'output_dir': str(tmp_path / 'deleted'),
            'base_model_id': 'test',
            'base_model_name': 'Orphaned',
            'status': 'completed'
        })
        
        stats = registry.get_statistics()
        assert stats['total_models'] == 1
        assert stats['orphaned'] == 1
    
    def test_cleanup_orphaned_models(self, isolated_registry, monkeypatch, tmp_path):
        """Test cleaning up orphaned models."""
        registry = get_registry()
        
        # Register orphaned model
        registry.register_model({
            'output_dir': str(tmp_path / 'deleted'),
            'base_model_id': 'test',
            'base_model_name': 'Orphaned',
            'status': 'completed'
        })
        
        # Cleanup
        total, removed = registry.cleanup_orphaned_models(delete_registry_entries=True)
        
        assert total == 1
        assert removed == 1
        
        # Verify removed
        models = registry.list_models(include_orphaned=True)
        assert len(models) == 0


# ============ UI FUNCTION TESTS ============

class TestMyModelsUI:
    """Test My Models tab UI functions."""
    
    def test_scan_for_models_empty(self, isolated_registry):
        """Test scan_for_models when no models exist."""
        data, status = scan_for_models()
        
        assert isinstance(data, list)
        assert len(data) == 0
        assert "0" in status or "Found" in status
    
    def test_scan_for_models_finds_models(self, isolated_registry, monkeypatch, tmp_path):
        """Test scan_for_models finds models."""
        # Setup mock outputs
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        model_dir = outputs_dir / "test_20260101"
        model_dir.mkdir()
        (model_dir / "adapters").mkdir()
        (model_dir / "adapters" / "adapters.safetensors").write_text("mock")
        
        summary = {
            "model": "test",
            "training_config": {"iterations": 100},
            "validation_losses": {},
            "best_iteration": 100
        }
        (model_dir / "training_summary.json").write_text(json.dumps(summary))
        
        monkeypatch.chdir(tmp_path)
        
        try:
            data, status = scan_for_models()
            assert len(data) >= 1
            assert len(data[0]) == 8  # 8 columns: ID, Status, Model, Dataset, Iterations, Best Loss, Exports, Output Dir
        finally:
            monkeypatch.chdir(Path.cwd())
    
    def test_format_model_for_display(self, sample_trained_model):
        """Test formatting model for display."""
        display = format_model_for_display(sample_trained_model)
        
        assert 'id' in display
        assert 'status' in display
        assert 'model_name' in display
        assert 'best_loss' in display
        assert 'exports' in display
        
        # Check status formatting
        assert 'Completed' in display['status'] or '✅' in display['status']
        
        # Check exports formatting
        assert 'Adapter' in display['exports']
        assert 'Fused' in display['exports']
    
    def test_get_model_details(self, isolated_registry, sample_trained_model):
        """Test getting model details."""
        # Register sample model
        registry = get_registry()
        model_id = registry.register_model({
            'output_dir': sample_trained_model.output_dir,
            'base_model_id': sample_trained_model.base_model_id,
            'base_model_name': sample_trained_model.base_model_name,
            'iterations': sample_trained_model.iterations,
            'learning_rate': sample_trained_model.learning_rate,
            'best_loss': sample_trained_model.best_loss,
            'best_iteration': sample_trained_model.best_iteration,
            'status': sample_trained_model.status,
            'notes': sample_trained_model.notes,
            'tags': sample_trained_model.tags,
            'exports': sample_trained_model.exports
        })
        
        details = get_model_details(model_id)
        
        assert len(details) == 8  # info, exports, config, metrics, notes, tags, status, plot
        assert isinstance(details[0], str)  # info_text
        assert isinstance(details[1], str)  # exports_text
        assert isinstance(details[2], str)  # config_text
        assert isinstance(details[3], str)  # metrics_text
    
    def test_get_model_details_not_found(self):
        """Test getting details for non-existent model."""
        details = get_model_details("nonexistent")
        assert details[0] == "Model not found"
    
    def test_update_model_notes(self, isolated_registry):
        """Test updating model notes."""
        registry = get_registry()
        
        model_id = registry.register_model({
            'output_dir': 'outputs/test',
            'base_model_id': 'test',
            'base_model_name': 'Test',
            'notes': 'Initial notes'
        })
        
        status = update_model_notes(model_id, "Updated notes")
        assert "✅" in status or "updated" in status.lower()
        
        model = registry.get_model(model_id)
        assert model.notes == "Updated notes"
    
    def test_update_model_tags(self, isolated_registry):
        """Test updating model tags."""
        registry = get_registry()
        
        model_id = registry.register_model({
            'output_dir': 'outputs/test',
            'base_model_id': 'test',
            'base_model_name': 'Test',
            'tags': []
        })
        
        status = update_model_tags(model_id, "tag1, tag2, tag3")
        assert "✅" in status or "updated" in status.lower()
        
        model = registry.get_model(model_id)
        assert "tag1" in model.tags
        assert "tag2" in model.tags
    
    def test_filter_models(self, isolated_registry):
        """Test filtering models."""
        registry = get_registry()
        
        # Register models
        registry.register_model({
            'output_dir': 'outputs/phi3',
            'base_model_id': 'mlx-community/Phi-3',
            'base_model_name': 'Phi-3',
            'iterations': 100,
            'status': 'completed'
        })
        
        registry.register_model({
            'output_dir': 'outputs/llama',
            'base_model_id': 'mlx-community/Llama',
            'base_model_name': 'Llama',
            'iterations': 200,
            'status': 'completed'
        })
        
        # Filter all
        data, status = filter_models('all', '')
        assert len(data) == 2
        
        # Filter completed
        data, status = filter_models('completed', '')
        assert len(data) == 2
        
        # Filter with search
        data, status = filter_models('all', 'Phi')
        assert len(data) == 1
    
    def test_delete_model(self, isolated_registry):
        """Test deleting model from UI."""
        registry = get_registry()
        
        model_id = registry.register_model({
            'output_dir': 'outputs/test',
            'base_model_id': 'test',
            'base_model_name': 'Test',
            'iterations': 100
        })
        
        # Delete
        data, status = delete_model(model_id, delete_files=False)
        
        assert "deleted" in status.lower() or "✅" in status
        assert registry.get_model(model_id) is None


# ============ INTEGRATION TESTS ============

class TestMyModelsIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow_register_scan_display(self, isolated_registry, monkeypatch, tmp_path):
        """Test complete workflow: register, scan, display."""
        # Create training output
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        model_dir = outputs_dir / "phi3_20260101_120000"
        model_dir.mkdir()
        (model_dir / "adapters").mkdir()
        (model_dir / "adapters" / "adapters.safetensors").write_text("mock")
        
        summary = {
            "model": "mlx-community/Phi-3-mini",
            "training_config": {"iterations": 200, "learning_rate": 0.0001},
            "validation_losses": {"200": 1.5},
            "best_iteration": 200,
            "best_val_loss": 1.5
        }
        (model_dir / "training_summary.json").write_text(json.dumps(summary))
        
        monkeypatch.chdir(tmp_path)
        
        try:
            # Scan
            data, status = scan_for_models()
            assert len(data) == 1
            
            # Verify data format
            row = data[0]
            assert len(row) == 8
            assert row[1]  # Status
            assert row[2]  # Model name
            assert row[4]  # Iterations
            assert row[5]  # Best loss
            
            # Get details
            model_id = row[0]
            details = get_model_details(model_id)
            assert len(details) == 8
            
        finally:
            monkeypatch.chdir(Path.cwd())
    
    def test_training_to_registry_workflow(self, isolated_registry, monkeypatch, tmp_path):
        """Test workflow from training registration to My Models display."""
        from edukaai_studio.core.trained_models_registry import get_registry
        
        registry = get_registry()
        
        # Simulate training starting
        model_data = {
            'output_dir': str(tmp_path / 'outputs' / 'training_20260101'),
            'base_model_id': 'mlx-community/Phi-3-mini',
            'base_model_name': 'Phi-3 Mini',
            'iterations': 200,
            'learning_rate': '1e-4',
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.0,
            'batch_size': 1,
            'grad_accumulation': 32,
            'dataset_path': str(tmp_path / 'data.jsonl'),
            'dataset_size': 150,
            'best_loss': float('inf'),
            'final_loss': float('inf'),
            'best_iteration': 0,
            'train_losses': {},
            'val_losses': {},
            'training_duration_minutes': 0.0,
            'exports': {'adapter': None, 'fused': None, 'gguf': None},
            'status': 'running',
            'tags': ['in-progress'],
            'notes': 'Training started'
        }
        
        # Register running model
        model_id = registry.register_model(model_data)
        
        # Verify registered
        model = registry.get_model(model_id)
        assert model.status == 'running'
        assert 'in-progress' in model.tags
        
        # Simulate training completion
        registry.update_model(model_id,
            status='completed',
            best_loss=1.234,
            best_iteration=200,
            tags=[],
            notes='Training completed'
        )
        
        # Verify updated
        model = registry.get_model(model_id)
        assert model.status == 'completed'
        assert model.best_loss == 1.234
        assert 'in-progress' not in model.tags


# ============ EDGE CASE TESTS ============

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_register_model_with_invalid_data(self, isolated_registry):
        """Test registering model with minimal/invalid data."""
        registry = get_registry()
        
        # Register with minimal data
        model_id = registry.register_model({
            'output_dir': 'outputs/test',
            # Missing many fields
        })
        
        # Should still work with defaults
        model = registry.get_model(model_id)
        assert model is not None
        assert model.base_model_id == 'unknown'  # Default
        assert model.iterations == 0  # Default
    
    def test_format_model_with_none_values(self):
        """Test formatting model with None values."""
        model = TrainedModel(
            id="test123",
            created_at=None,
            updated_at=None,
            base_model_id=None,
            base_model_name=None,
            iterations=0,
            learning_rate=None,
            lora_rank=0,
            lora_alpha=0,
            lora_dropout=0.0,
            batch_size=0,
            grad_accumulation=0,
            dataset_path=None,
            dataset_size=0,
            output_dir=None,
            best_loss=float('inf'),
            final_loss=float('inf'),
            best_iteration=0,
            train_losses={},
            val_losses={},
            training_duration_minutes=0.0,
            exports={},
            status=None,
            tags=None,
            notes=None
        )
        
        display = format_model_for_display(model)
        
        # Should not crash
        assert 'id' in display
        assert 'status' in display
        assert 'best_loss' in display
    
    def test_scan_with_corrupted_summary(self, isolated_registry, monkeypatch, tmp_path):
        """Test scanning with corrupted training_summary.json."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        model_dir = outputs_dir / "corrupted"
        model_dir.mkdir()
        (model_dir / "adapters").mkdir()
        (model_dir / "adapters" / "adapters.safetensors").write_text("mock")
        
        # Write corrupted JSON
        (model_dir / "training_summary.json").write_text("not valid json {{]")
        
        monkeypatch.chdir(tmp_path)
        
        try:
            # Should handle gracefully
            data, status = scan_for_models()
            # May or may not find the model depending on error handling
        finally:
            monkeypatch.chdir(Path.cwd())
    
    def test_cleanup_with_missing_directories(self, isolated_registry):
        """Test cleanup when directories are already missing."""
        registry = get_registry()
        
        # Register orphaned model
        registry.register_model({
            'output_dir': '/nonexistent/path',
            'base_model_id': 'test',
            'base_model_name': 'Test'
        })
        
        # Cleanup should handle gracefully
        total, removed = registry.cleanup_orphaned_models(delete_registry_entries=True)
        assert removed >= 0  # Should not crash


# ============ PERFORMANCE TESTS ============

@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""
    
    def test_scan_large_registry(self, isolated_registry, monkeypatch, tmp_path):
        """Test scanning with many models."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        
        # Create 100 models
        for i in range(100):
            model_dir = outputs_dir / f"model_{i}_20260101"
            model_dir.mkdir()
            (model_dir / "adapters").mkdir()
            (model_dir / "adapters" / "adapters.safetensors").write_text("mock")
            
            summary = {
                "model": f"test-model-{i}",
                "training_config": {"iterations": 100},
                "validation_losses": {},
                "best_iteration": 100
            }
            (model_dir / "training_summary.json").write_text(json.dumps(summary))
        
        monkeypatch.chdir(tmp_path)
        
        try:
            import time
            start = time.time()
            data, status = scan_for_models()
            duration = time.time() - start
            
            assert len(data) == 100
            assert duration < 5.0  # Should complete in under 5 seconds
        finally:
            monkeypatch.chdir(Path.cwd())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
