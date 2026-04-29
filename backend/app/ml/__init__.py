# Make ml a package
# Use lazy imports to avoid crashing on non-Apple-Silicon machines where MLX isn't available

def __getattr__(name):
    if name in ('training_manager', 'TrainingConfig', 'export_model', 'load_model_for_inference', 'generate_response'):
        try:
            from .trainer import training_manager, TrainingConfig, export_model, load_model_for_inference, generate_response
            _exports = {
                'training_manager': training_manager,
                'TrainingConfig': TrainingConfig,
                'export_model': export_model,
                'load_model_for_inference': load_model_for_inference,
                'generate_response': generate_response,
            }
            if name in _exports:
                return _exports[name]
        except ImportError as e:
            raise ImportError(
                f"MLX is required for EdukaAI Studio training features. "
                f"Install on Apple Silicon: pip install mlx mlx-lm. Error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'training_manager',
    'TrainingConfig',
    'export_model',
    'load_model_for_inference',
    'generate_response'
]
