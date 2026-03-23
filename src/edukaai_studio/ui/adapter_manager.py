"""
Adapter Manager Module

Manages the lifecycle of LoRA adapters including creation, loading, deletion,
and organization. Enables multi-adapter support for rapid task switching.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import zipfile


@dataclass
class AdapterMetadata:
    """Metadata for a LoRA adapter."""
    id: str
    name: str
    description: str
    created_at: str
    base_model: str
    training: Dict = field(default_factory=dict)
    performance: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[str] = None
    path: str = ""
    size_mb: float = 0.0
    is_active: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AdapterMetadata":
        """Create from dictionary."""
        return cls(**data)


class AdapterManager:
    """
    Manages LoRA adapters for multi-adapter support.
    
    Provides functionality for:
    - Creating adapters from training output
    - Loading/unloading adapters
    - Listing and organizing adapters
    - Importing/exporting adapters
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize adapter manager.
        
        Args:
            base_dir: Base directory for adapters. Defaults to ~/.football-lora
        """
        if base_dir is None:
            base_dir = os.path.expanduser("~/.football-lora")
        
        self.base_dir = Path(base_dir)
        self.adapters_dir = self.base_dir / "adapters"
        self.registry_path = self.base_dir / "adapter-registry.json"
        self.active_adapter_path = self.base_dir / "active-adapter.json"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load registry
        self.registry = self._load_registry()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_registry(self) -> Dict:
        """Load adapter registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"adapters": [], "active_adapter_id": None}
        return {"adapters": [], "active_adapter_id": None}
    
    def _save_registry(self):
        """Save adapter registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _generate_adapter_id(self, name: str) -> str:
        """Generate unique adapter ID from name."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '-').lower()
        return f"{safe_name}-{timestamp}"
    
    def _calculate_adapter_size(self, adapter_path: Path) -> float:
        """Calculate total size of adapter in MB."""
        total_size = 0
        if adapter_path.exists():
            for file_path in adapter_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)
    
    def create_adapter(
        self,
        training_output_dir: str,
        name: str,
        description: str,
        base_model: str,
        training_params: Dict,
        performance_metrics: Dict,
        tags: List[str] = None
    ) -> AdapterMetadata:
        """
        Create a new adapter from training output.
        
        Args:
            training_output_dir: Directory containing training output files
            name: User-friendly name for the adapter
            description: Description of what the adapter does
            base_model: Base model used for training
            training_params: Dictionary of training parameters
            performance_metrics: Dictionary of performance metrics
            tags: Optional list of tags for organization
            
        Returns:
            AdapterMetadata object for the created adapter
        """
        if tags is None:
            tags = []
        
        # Generate adapter ID
        adapter_id = self._generate_adapter_id(name)
        
        # Create adapter directory
        adapter_path = self.adapters_dir / adapter_id
        adapter_path.mkdir(parents=True, exist_ok=True)
        
        # Copy training files
        source_path = Path(training_output_dir)
        
        # Check for adapter files in the source directory
        # The training might put them directly in output_dir or in an 'adapters' subdirectory
        possible_source_paths = [
            source_path,  # Direct output
            source_path / "adapters",  # Subdirectory
            source_path / "best_adapter",  # Best adapter subdirectory
        ]
        
        # Find where the adapter files actually are
        adapter_source = None
        for test_path in possible_source_paths:
            if (test_path / "adapters.safetensors").exists():
                adapter_source = test_path
                break
        
        if adapter_source is None:
            # No adapter files found - create minimal config
            print(f"[ADAPTER] Warning: No adapter files found in {source_path}")
            # Create a basic adapter_config.json so the adapter is at least loadable
            basic_config = {
                "lora_rank": training_params.get("lora_rank", 16),
                "lora_alpha": training_params.get("lora_alpha", 32),
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "model": base_model
            }
            with open(adapter_path / "adapter_config.json", 'w') as f:
                json.dump(basic_config, f, indent=2)
        else:
            # Copy adapter weights
            adapter_files = [
                "adapters.safetensors",
                "adapter_model.safetensors",
                "adapter_config.json"
            ]
            
            for file_name in adapter_files:
                src_file = adapter_source / file_name
                if src_file.exists():
                    shutil.copy2(src_file, adapter_path / file_name)
                    print(f"[ADAPTER] Copied {file_name} from {adapter_source}")
        
        # Copy training report if exists
        report_file = source_path / "training_report.pdf"
        if report_file.exists():
            shutil.copy2(report_file, adapter_path / "training_report.pdf")
        
        # Create metadata
        metadata = AdapterMetadata(
            id=adapter_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            base_model=base_model,
            training=training_params,
            performance=performance_metrics,
            tags=tags,
            path=str(adapter_path),
            size_mb=self._calculate_adapter_size(adapter_path)
        )
        
        # Save metadata
        metadata_file = adapter_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update registry
        self.registry["adapters"].append({
            "id": adapter_id,
            "name": name,
            "path": str(adapter_path),
            "base_model": base_model,
            "size_mb": metadata.size_mb,
            "created": metadata.created_at
        })
        self._save_registry()
        
        return metadata
    
    def list_adapters(
        self,
        base_model: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[AdapterMetadata]:
        """
        List available adapters with optional filtering.
        
        Args:
            base_model: Filter by base model (optional)
            tags: Filter by tags (optional)
            
        Returns:
            List of AdapterMetadata objects
        """
        adapters = []
        
        for adapter_info in self.registry.get("adapters", []):
            adapter_path = Path(adapter_info["path"])
            metadata_file = adapter_path / "metadata.json"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        metadata = AdapterMetadata.from_dict(data)
                        
                        # Check if active
                        metadata.is_active = (self.registry.get("active_adapter_id") == metadata.id)
                        
                        # Apply filters
                        if base_model and metadata.base_model != base_model:
                            continue
                        
                        if tags and not any(tag in metadata.tags for tag in tags):
                            continue
                        
                        adapters.append(metadata)
                except (json.JSONDecodeError, IOError):
                    continue
        
        # Sort by last used (most recent first), then by created date
        adapters.sort(
            key=lambda x: (x.last_used or "", x.created_at),
            reverse=True
        )
        
        return adapters
    
    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """
        Get metadata for a specific adapter.
        
        Args:
            adapter_id: ID of the adapter
            
        Returns:
            AdapterMetadata or None if not found
        """
        adapter_path = self.adapters_dir / adapter_id
        metadata_file = adapter_path / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = AdapterMetadata.from_dict(data)
                    metadata.is_active = (self.registry.get("active_adapter_id") == adapter_id)
                    return metadata
            except (json.JSONDecodeError, IOError):
                return None
        
        return None
    
    def set_active_adapter(self, adapter_id: str) -> bool:
        """
        Set an adapter as the active one.
        
        Args:
            adapter_id: ID of the adapter to activate
            
        Returns:
            True if successful, False otherwise
        """
        # Verify adapter exists
        if not self.get_adapter(adapter_id):
            return False
        
        # Update registry
        self.registry["active_adapter_id"] = adapter_id
        self._save_registry()
        
        # Save active adapter info separately for quick access
        active_info = {
            "adapter_id": adapter_id,
            "activated_at": datetime.now().isoformat()
        }
        with open(self.active_adapter_path, 'w') as f:
            json.dump(active_info, f, indent=2)
        
        return True
    
    def get_active_adapter(self) -> Optional[AdapterMetadata]:
        """
        Get the currently active adapter.
        
        Returns:
            AdapterMetadata or None if no adapter is active
        """
        active_id = self.registry.get("active_adapter_id")
        if active_id:
            return self.get_adapter(active_id)
        return None
    
    def clear_active_adapter(self):
        """Clear the active adapter (use base model only)."""
        self.registry["active_adapter_id"] = None
        self._save_registry()
        
        if self.active_adapter_path.exists():
            self.active_adapter_path.unlink()
    
    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter permanently.
        
        Args:
            adapter_id: ID of the adapter to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Don't delete if it's the active adapter
        if self.registry.get("active_adapter_id") == adapter_id:
            return False
        
        adapter_path = self.adapters_dir / adapter_id
        
        if adapter_path.exists():
            try:
                shutil.rmtree(adapter_path)
                
                # Update registry
                self.registry["adapters"] = [
                    a for a in self.registry["adapters"]
                    if a["id"] != adapter_id
                ]
                self._save_registry()
                
                return True
            except OSError:
                return False
        
        return False
    
    def update_adapter_metadata(
        self,
        adapter_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update adapter metadata.
        
        Args:
            adapter_id: ID of the adapter to update
            name: New name (optional)
            description: New description (optional)
            tags: New tags (optional)
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self.get_adapter(adapter_id)
        if not metadata:
            return False
        
        if name is not None:
            metadata.name = name
        if description is not None:
            metadata.description = description
        if tags is not None:
            metadata.tags = tags
        
        # Save updated metadata
        adapter_path = Path(metadata.path)
        metadata_file = adapter_path / "metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update registry
            for adapter in self.registry["adapters"]:
                if adapter["id"] == adapter_id:
                    adapter["name"] = metadata.name
                    break
            
            self._save_registry()
            return True
        except IOError:
            return False
    
    def record_usage(self, adapter_id: str):
        """
        Record that an adapter was used.
        
        Args:
            adapter_id: ID of the adapter used
        """
        metadata = self.get_adapter(adapter_id)
        if metadata:
            metadata.usage_count += 1
            metadata.last_used = datetime.now().isoformat()
            
            # Save updated metadata
            adapter_path = Path(metadata.path)
            metadata_file = adapter_path / "metadata.json"
            
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
            except IOError:
                pass
    
    def export_adapter(self, adapter_id: str, export_path: str) -> bool:
        """
        Export an adapter to a ZIP file.
        
        Args:
            adapter_id: ID of the adapter to export
            export_path: Path for the exported ZIP file
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self.get_adapter(adapter_id)
        if not metadata:
            return False
        
        adapter_path = Path(metadata.path)
        
        try:
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in adapter_path.rglob('*'):
                    if file_path.is_file():
                        zf.write(
                            file_path,
                            arcname=file_path.relative_to(adapter_path)
                        )
            return True
        except (IOError, zipfile.BadZipFile):
            return False
    
    def import_adapter(self, import_path: str, name: Optional[str] = None) -> Optional[AdapterMetadata]:
        """
        Import an adapter from a ZIP file.
        
        Args:
            import_path: Path to the ZIP file to import
            name: Optional new name for the imported adapter
            
        Returns:
            AdapterMetadata if successful, None otherwise
        """
        try:
            with zipfile.ZipFile(import_path, 'r') as zf:
                # Generate new ID
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                adapter_id = f"imported-{timestamp}"
                
                # Extract to new directory
                adapter_path = self.adapters_dir / adapter_id
                adapter_path.mkdir(parents=True, exist_ok=True)
                zf.extractall(adapter_path)
                
                # Load and update metadata
                metadata_file = adapter_path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    
                    # Update metadata
                    data["id"] = adapter_id
                    data["created_at"] = datetime.now().isoformat()
                    if name:
                        data["name"] = name
                    data["path"] = str(adapter_path)
                    data["size_mb"] = self._calculate_adapter_size(adapter_path)
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # Update registry
                    self.registry["adapters"].append({
                        "id": adapter_id,
                        "name": data["name"],
                        "path": str(adapter_path),
                        "base_model": data.get("base_model", "unknown"),
                        "size_mb": data["size_mb"],
                        "created": data["created_at"]
                    })
                    self._save_registry()
                    
                    return AdapterMetadata.from_dict(data)
        
        except (IOError, zipfile.BadZipFile, json.JSONDecodeError):
            return None
        
        return None
    
    def get_total_storage_used(self) -> float:
        """
        Get total storage used by all adapters.
        
        Returns:
            Total size in MB
        """
        total = 0.0
        for adapter_info in self.registry.get("adapters", []):
            total += adapter_info.get("size_mb", 0)
        return round(total, 2)
    
    def get_compatible_adapters(self, base_model: str) -> List[AdapterMetadata]:
        """
        Get adapters compatible with a specific base model.
        
        Args:
            base_model: Base model to check compatibility with
            
        Returns:
            List of compatible AdapterMetadata objects
        """
        all_adapters = self.list_adapters()
        compatible = []
        
        for adapter in all_adapters:
            # Check for exact match or known compatible models
            if adapter.base_model == base_model:
                compatible.append(adapter)
            elif self._check_compatibility(adapter.base_model, base_model):
                compatible.append(adapter)
        
        return compatible
    
    def _check_compatibility(self, adapter_base: str, target_base: str) -> bool:
        """
        Check if two base models are compatible.
        
        Args:
            adapter_base: Base model the adapter was trained on
            target_base: Base model to check compatibility with
            
        Returns:
            True if compatible, False otherwise
        """
        # Define compatibility groups
        compatibility_groups = [
            ["phi", "phi-3", "phi-4"],
            ["llama-3", "llama-3.1", "llama-3.2"],
            ["mistral"],
            ["gemma"],
            ["qwen"]
        ]
        
        adapter_lower = adapter_base.lower()
        target_lower = target_base.lower()
        
        for group in compatibility_groups:
            if any(keyword in adapter_lower for keyword in group):
                if any(keyword in target_lower for keyword in group):
                    return True
        
        return False


# Global adapter manager instance
_adapter_manager = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager instance."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
