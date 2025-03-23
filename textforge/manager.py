import os 
import json 
import shutil
import pkg_resources
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from textforge.deployment import serve

class ModelManager:
    """Manages model artifacts and metadata for TextForge."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            base_path: Optional custom path for model storage. If None, uses package data directory.
        """
        if base_path is None:
            # Get package installation directory
            pkg_path = pkg_resources.resource_filename('textforge', '')
            self.base_path = Path(pkg_path) / 'models'
        else:
            self.base_path = Path(base_path)
            
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.registry_file = self.base_path / "registry.json"
        self._init_registry()

    def _init_registry(self):
        """Initialize or load the model registry."""
        if not self.registry_file.exists():
            self.registry = {}
            self._save_registry()
        else:
            with open(self.registry_file, "r") as f:
                self.registry = json.load(f)
    
    def _save_registry(self):
        """Save the registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self, 
        model_path: str | Path, 
        model_name: str,
        version: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        copy_files: bool = True
    ) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to model files
            model_name: Name of the model
            version: Optional version string (default: timestamp)
            metadata: Optional metadata dictionary
            copy_files: Whether to copy model files to package directory
            
        Returns:
            model_id: Unique identifier for the registered model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d%H%M%S")
        
        model_id = f"{model_name}_{version}"
        target_path = self.base_path / model_id

        if copy_files:
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(model_path, target_path)
            stored_path = target_path
        else:
            # Just store reference to original path
            stored_path = Path(model_path)

        self.registry[model_id] = {
            "model_name": model_name,
            "version": version,
            "path": str(stored_path),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "is_package_copy": copy_files
        }
        self._save_registry()
        return model_id
    
    def load_model(self, model_id: str) -> dict:
        """
        Load model information from registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary
            
        Raises:
            ValueError: If model_id not found
        """
        if model_id not in self.registry:
            raise ValueError(f"{model_id} not found in registry")
        return self.registry[model_id]
    
    def list_models(self) -> dict:
        """List all registered models."""
        return self.registry

    def delete_model(self, model_id: str):
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Raises:
            ValueError: If model_id not found
        """
        if model_id not in self.registry:
            raise ValueError(f"{model_id} not found in registry")
            
        model_info = self.registry[model_id]
        if model_info["is_package_copy"]:
            shutil.rmtree(self.base_path / model_id)
            
        del self.registry[model_id]
        self._save_registry()
    
    def serve_model(self, model_id: str):
        """
        Serve a model using the FastAPI server.
        
        Args:
            model_id: Model identifier
        """
        model_info = self.load_model(model_id)
        serve(model_info["path"])