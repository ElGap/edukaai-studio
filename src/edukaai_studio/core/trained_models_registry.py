"""Trained Models Registry for EdukaAI Fine Tuning Studio.

Manages all fine-tuned adapters with metadata, exports, and discovery.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


# Registry file location
REGISTRY_FILE = Path(__file__).parent.parent.parent.parent / "data" / "trained_models_registry.json"


@dataclass
class TrainedModel:
    """Represents a single fine-tuned model/training session."""
    
    id: str
    created_at: str
    updated_at: str
    
    # Model Info
    base_model_id: str
    base_model_name: str
    
    # Training Config
    iterations: int
    learning_rate: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    grad_accumulation: int
    
    # Dataset Info
    dataset_path: str
    dataset_size: int
    
    # Results
    output_dir: str
    best_loss: float
    final_loss: float
    best_iteration: int
    train_losses: Dict[int, float] = field(default_factory=dict)
    val_losses: Dict[int, float] = field(default_factory=dict)
    training_duration_minutes: float = 0.0
    
    # Export Status
    exports: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "adapter": None,
        "fused": None,
        "gguf": None
    })
    
    # Metadata
    status: str = "completed"  # 'running', 'completed', 'failed', 'stopped'
    error_message: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainedModel":
        """Create from dictionary."""
        return cls(**data)


class TrainedModelsRegistry:
    """Registry for managing all fine-tuned models."""
    
    def __init__(self):
        self.registry_file = REGISTRY_FILE
        self._ensure_registry_exists()
        self._data = self._load()
    
    def _ensure_registry_exists(self):
        """Ensure registry file and directory exist."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self._save({"models": [], "version": "1.0"})
    
    def _load(self) -> Dict[str, Any]:
        """Load registry from JSON file."""
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"models": [], "version": "1.0"}
    
    def _save(self, data: Dict[str, Any]):
        """Save registry to JSON file."""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _generate_id(self, output_dir: str, timestamp: str) -> str:
        """Generate unique ID for a model."""
        hash_input = f"{output_dir}{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def register_model(self, model_data: Dict[str, Any]) -> str:
        """Register a new trained model.
        
        Args:
            model_data: Dictionary with model training info
            
        Returns:
            model_id: Unique identifier for the model
        """
        # Generate ID
        model_id = self._generate_id(
            model_data.get('output_dir', ''),
            model_data.get('created_at', datetime.now().isoformat())
        )
        
        # Create TrainedModel instance
        model = TrainedModel(
            id=model_id,
            created_at=model_data.get('created_at', datetime.now().isoformat()),
            updated_at=datetime.now().isoformat(),
            base_model_id=model_data.get('base_model_id', 'unknown'),
            base_model_name=model_data.get('base_model_name', 'Unknown Model'),
            iterations=model_data.get('iterations', 0),
            learning_rate=model_data.get('learning_rate', '1e-4'),
            lora_rank=model_data.get('lora_rank', 16),
            lora_alpha=model_data.get('lora_alpha', 32),
            lora_dropout=model_data.get('lora_dropout', 0.0),
            batch_size=model_data.get('batch_size', 1),
            grad_accumulation=model_data.get('grad_accumulation', 32),
            dataset_path=model_data.get('dataset_path', ''),
            dataset_size=model_data.get('dataset_size', 0),
            output_dir=model_data.get('output_dir', ''),
            best_loss=model_data.get('best_loss', float('inf')),
            final_loss=model_data.get('final_loss', float('inf')),
            best_iteration=model_data.get('best_iteration', 0),
            train_losses=model_data.get('train_losses', {}),
            val_losses=model_data.get('val_losses', {}),
            training_duration_minutes=model_data.get('training_duration_minutes', 0.0),
            exports=model_data.get('exports', {'adapter': None, 'fused': None, 'gguf': None}),
            status=model_data.get('status', 'completed'),
            error_message=model_data.get('error_message'),
            tags=model_data.get('tags', []),
            notes=model_data.get('notes', '')
        )
        
        # Check if model already exists (update instead of add)
        existing_idx = None
        for idx, m in enumerate(self._data['models']):
            if m.get('output_dir') == model.output_dir:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            # Update existing entry
            self._data['models'][existing_idx] = model.to_dict()
        else:
            # Add new entry
            self._data['models'].append(model.to_dict())
        
        # Sort by created_at (newest first)
        self._data['models'].sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        self._save(self._data)
        return model_id
    
    def get_model(self, model_id: str) -> Optional[TrainedModel]:
        """Get a specific model by ID."""
        for model_data in self._data['models']:
            if model_data.get('id') == model_id:
                return TrainedModel.from_dict(model_data)
        return None
    
    def get_model_by_output_dir(self, output_dir: str) -> Optional[TrainedModel]:
        """Get model by output directory path."""
        for model_data in self._data['models']:
            if model_data.get('output_dir') == output_dir:
                return TrainedModel.from_dict(model_data)
        return None
    
    def list_models(self, 
                    filter_status: Optional[str] = None,
                    filter_base_model: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    search_query: Optional[str] = None,
                    include_orphaned: bool = False) -> List[TrainedModel]:
        """List all models with optional filtering.
        
        Args:
            filter_status: Filter by status ('completed', 'running', etc.)
            filter_base_model: Filter by base model ID
            tags: Filter by tags (must have all specified tags)
            search_query: Search in name, notes, tags
            include_orphaned: If False, skip models whose output_dir no longer exists
            
        Returns:
            List of TrainedModel instances
        """
        models = []
        
        for model_data in self._data['models']:
            # Skip orphaned models by default (output_dir doesn't exist)
            if not include_orphaned:
                output_dir = model_data.get('output_dir', '')
                if output_dir and not Path(output_dir).exists():
                    continue  # Skip this orphaned model
            
            model = TrainedModel.from_dict(model_data)
            
            # Apply filters
            if filter_status and model.status != filter_status:
                continue
            
            if filter_base_model and filter_base_model not in model.base_model_id:
                continue
            
            if tags:
                if not all(tag in model.tags for tag in tags):
                    continue
            
            if search_query:
                query = search_query.lower()
                searchable_text = f"{model.base_model_name} {model.notes} {' '.join(model.tags)}".lower()
                if query not in searchable_text:
                    continue
            
            models.append(model)
        
        return models
    
    def update_model(self, model_id: str, **updates) -> bool:
        """Update model metadata.
        
        Args:
            model_id: Model ID to update
            **updates: Fields to update
            
        Returns:
            True if updated, False if not found
        """
        for idx, model_data in enumerate(self._data['models']):
            if model_data.get('id') == model_id:
                model_data.update(updates)
                model_data['updated_at'] = datetime.now().isoformat()
                self._save(self._data)
                return True
        return False
    
    def update_exports(self, model_id: str, export_type: str, export_path: str) -> bool:
        """Update export status for a model.
        
        Args:
            model_id: Model ID
            export_type: 'adapter', 'fused', or 'gguf'
            export_path: Path to exported file
            
        Returns:
            True if updated
        """
        for model_data in self._data['models']:
            if model_data.get('id') == model_id:
                if 'exports' not in model_data:
                    model_data['exports'] = {}
                model_data['exports'][export_type] = export_path
                model_data['updated_at'] = datetime.now().isoformat()
                self._save(self._data)
                return True
        return False
    
    def delete_model(self, model_id: str, delete_files: bool = False) -> bool:
        """Delete model from registry.
        
        Args:
            model_id: Model ID to delete
            delete_files: If True, also delete output files
            
        Returns:
            True if deleted
        """
        for idx, model_data in enumerate(self._data['models']):
            if model_data.get('id') == model_id:
                output_dir = model_data.get('output_dir', '')
                
                # Remove from registry
                self._data['models'].pop(idx)
                self._save(self._data)
                
                # Optionally delete files
                if delete_files and output_dir:
                    output_path = Path(output_dir)
                    if output_path.exists():
                        import shutil
                        shutil.rmtree(output_path)
                
                return True
        return False
    
    def scan_for_new_models(self) -> List[TrainedModel]:
        """Scan outputs/ directory for models not in registry.
        
        Returns:
            List of newly discovered models
        """
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            return []
        
        new_models = []
        registered_dirs = {m.get('output_dir', '') for m in self._data['models']}
        
        for output_dir in outputs_dir.iterdir():
            if not output_dir.is_dir():
                continue
            
            # Skip if already registered
            if str(output_dir) in registered_dirs:
                continue
            
            # Check if it's a valid training output
            adapter_path = output_dir / "adapters" / "adapters.safetensors"
            if not adapter_path.exists():
                adapter_path = output_dir / "best_adapter" / "adapters.safetensors"
            
            if adapter_path.exists():
                # Try to load metadata from training_summary.json
                summary_path = output_dir / "training_summary.json"
                metadata = self._extract_metadata_from_summary(summary_path, output_dir)
                
                # Register the model
                model_id = self.register_model(metadata)
                model = self.get_model(model_id)
                if model:
                    new_models.append(model)
        
        return new_models
    
    def _extract_metadata_from_summary(self, summary_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extract metadata from training summary file."""
        metadata = {
            'output_dir': str(output_dir),
            'created_at': datetime.fromtimestamp(output_dir.stat().st_mtime).isoformat(),
            'base_model_id': 'unknown',
            'base_model_name': output_dir.name[:30],
            'iterations': 0,
            'learning_rate': '1e-4',
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.0,
            'batch_size': 1,
            'grad_accumulation': 32,
            'dataset_path': '',
            'dataset_size': 0,
            'best_loss': float('inf'),
            'final_loss': float('inf'),
            'best_iteration': 0,
            'train_losses': {},
            'val_losses': {},
            'training_duration_minutes': 0.0,
            'exports': {
                'adapter': str(output_dir / "adapters" / "adapters.safetensors") 
                          if (output_dir / "adapters" / "adapters.safetensors").exists()
                          else str(output_dir / "best_adapter" / "adapters.safetensors")
                          if (output_dir / "best_adapter" / "adapters.safetensors").exists()
                          else None,
                'fused': None,
                'gguf': None
            },
            'status': 'completed',
            'tags': [],
            'notes': ''
        }
        
        # Try to load from training_summary.json
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                # Extract model info (actual structure from lora-train.py)
                metadata['base_model_id'] = summary.get('model', metadata['base_model_id'])
                
                # Extract training_config (actual structure)
                training_config = summary.get('training_config', {})
                metadata['iterations'] = training_config.get('iterations', 0)
                metadata['learning_rate'] = str(training_config.get('learning_rate', '1e-4'))
                metadata['batch_size'] = training_config.get('batch_size', 1)
                metadata['grad_accumulation'] = training_config.get('grad_accumulation_steps', 32)
                # LoRA params not in summary, keep defaults
                
                # Extract validation info (actual structure)
                validation_losses = summary.get('validation_losses', {})
                metadata['val_losses'] = validation_losses
                
                best_iteration = summary.get('best_iteration', 0)
                best_val_loss = summary.get('best_val_loss', None)
                
                metadata['best_iteration'] = best_iteration
                if best_val_loss is not None:
                    metadata['best_loss'] = best_val_loss
                    metadata['final_loss'] = best_val_loss  # Use best as final for now
                
                # Note: training_losses not stored in current summary format
                # Note: dataset info not stored in current summary format
                
            except Exception as e:
                print(f"[REGISTRY] Error reading summary from {summary_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Check for existing exports
        fused_path = output_dir / "fused_model"
        if fused_path.exists():
            metadata['exports']['fused'] = str(fused_path)
        
        # Check for GGUF files
        gguf_files = list(output_dir.glob("*.gguf"))
        if gguf_files:
            metadata['exports']['gguf'] = str(gguf_files[0])
        
        return metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        models = self._data.get('models', [])
        
        # Count orphaned models (files deleted but still in registry)
        orphaned = 0
        for model_data in models:
            output_dir = model_data.get('output_dir', '')
            if output_dir and not Path(output_dir).exists():
                orphaned += 1
        
        return {
            'total_models': len(models),
            'completed': len([m for m in models if m.get('status') == 'completed']),
            'failed': len([m for m in models if m.get('status') == 'failed']),
            'running': len([m for m in models if m.get('status') == 'running']),
            'orphaned': orphaned,
            'models_with_fused': len([m for m in models if m.get('exports', {}).get('fused')]),
            'models_with_gguf': len([m for m in models if m.get('exports', {}).get('gguf')]),
            'latest_model': models[0] if models else None
        }
    
    def cleanup_orphaned_models(self, delete_registry_entries: bool = True) -> Tuple[int, int]:
        """Clean up orphaned models (registry entries with no files on disk).
        
        Args:
            delete_registry_entries: If True, remove orphaned entries from registry.
                                   If False, just count them.
            
        Returns:
            Tuple of (total_checked, orphaned_removed)
        """
        models = self._data.get('models', [])
        orphaned_indices = []
        
        # Find orphaned models
        for idx, model_data in enumerate(models):
            output_dir = model_data.get('output_dir', '')
            if output_dir and not Path(output_dir).exists():
                orphaned_indices.append(idx)
        
        orphaned_count = len(orphaned_indices)
        
        if delete_registry_entries and orphaned_indices:
            # Remove in reverse order to maintain correct indices
            for idx in reversed(orphaned_indices):
                model_id = models[idx].get('id', 'unknown')
                print(f"[REGISTRY] Removing orphaned model: {model_id}")
                self._data['models'].pop(idx)
            
            self._save(self._data)
        
        return len(models), orphaned_count


# Global registry instance
_registry = None

def get_registry() -> TrainedModelsRegistry:
    """Get global registry instance."""
    global _registry
    if _registry is None:
        _registry = TrainedModelsRegistry()
    return _registry


def format_model_for_display(model: TrainedModel) -> Dict[str, Any]:
    """Format model data for UI display."""
    created_date = model.created_at[:10] if model.created_at and len(model.created_at) >= 10 else model.created_at
    
    # Format status with emoji
    status_emoji = {
        'running': '🏃',
        'completed': '✅',
        'failed': '❌',
        'stopped': '⏸️'
    }.get(model.status, '❓')
    status_display = f"{status_emoji} {model.status.title()}"
    
    # Format exports status
    exports_status = []
    if model.exports.get('adapter'):
        exports_status.append("✅ Adapter")
    else:
        exports_status.append("❌ Adapter")
    
    if model.exports.get('fused'):
        exports_status.append("✅ Fused")
    else:
        exports_status.append("❌ Fused")
    
    if model.exports.get('gguf'):
        exports_status.append("✅ GGUF")
    else:
        exports_status.append("❌ GGUF")
    
    # Format best loss
    if model.best_loss != float('inf') and model.best_loss > 0:
        best_loss_str = f"{model.best_loss:.4f}"
    else:
        best_loss_str = "N/A"
    
    # Format dataset
    dataset_name = "Unknown"
    if model.dataset_path:
        from pathlib import Path
        dataset_name = Path(model.dataset_path).name
    
    return {
        'id': model.id,
        'status': status_display,  # NEW: Status with emoji
        'date': created_date,
        'model_name': model.base_model_name[:30] if model.base_model_name else 'Unknown',
        'dataset': dataset_name,
        'iterations': model.iterations if model.iterations > 0 else '-',
        'best_loss': best_loss_str,
        'duration': f"{int(model.training_duration_minutes)}m" if model.training_duration_minutes and model.training_duration_minutes > 0 else "-",
        'exports': " | ".join(exports_status),
        'tags': ", ".join(model.tags) if model.tags else "",
        'output_dir': model.output_dir
    }
