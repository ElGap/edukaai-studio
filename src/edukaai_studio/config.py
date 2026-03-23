"""
Centralized Configuration for Fine-Tuning Training System

This module contains all configurable parameters, replacing hardcoded values
throughout the codebase. Modify values here to change system behavior globally.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Iterations - Research shows 900-1000 is optimal (March 2026)
    # Effective batch size 3,840 significantly outperforms 128
    DEFAULT_ITERATIONS: int = 200
    MIN_ITERATIONS: int = 100
    MAX_ITERATIONS: int = 2000
    ITERATION_STEP: int = 50
    
    # Learning rates (as strings for dropdown compatibility)
    LEARNING_RATE_OPTIONS: List[str] = field(default_factory=lambda: ["5e-5", "1e-4", "5e-4"])
    DEFAULT_LEARNING_RATE: str = "1e-4"
    
    # Batch settings - Research: Use gradient accumulation 32-64 for Macs
    # to simulate effective batch size 3,840
    DEFAULT_BATCH_SIZE: int = 1
    DEFAULT_GRAD_ACCUMULATION: int = 32
    
    # LoRA settings
    DEFAULT_LORA_RANK: int = 16
    DEFAULT_LORA_ALPHA: int = 32
    DEFAULT_LORA_DROPOUT: float = 0.0
    DEFAULT_LORA_MODULES: str = "all"
    
    # Early stopping
    DEFAULT_EARLY_STOPPING_PATIENCE: int = 2
    
    # Validation
    STEPS_PER_EVAL: int = 50
    SAVE_EVERY: int = 50
    DEFAULT_VALIDATION_SPLIT_PCT: int = 10
    MIN_VALIDATION_SPLIT_PCT: int = 5
    MAX_VALIDATION_SPLIT_PCT: int = 25


@dataclass
class TrainingPresets:
    """Research-backed training configuration presets."""
    QUICK = {
        "name": "Quick Test",
        "description": "Fast training for testing - 200 iterations, grad accum 16",
        "iterations": 100,
        "grad_accumulation": 16,
        "learning_rate": "5e-4",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "early_stopping": 1
    }
    
    BALANCED = {
        "name": "Balanced",
        "description": "Good balance of speed and quality - 600 iterations, grad accum 32",
        "iterations": 400,
        "grad_accumulation": 32,
        "learning_rate": "1e-4",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "early_stopping": 2
    }
    
    MAXIMUM = {
        "name": "Maximum Quality",
        "description": "Best quality - 1000 iterations, grad accum 64, higher rank",
        "iterations": 1000,
        "grad_accumulation": 64,
        "learning_rate": "5e-5",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "early_stopping": 3
    }
    
    @classmethod
    def get_preset_names(cls) -> List[str]:
        """Return list of preset names."""
        return [cls.QUICK["name"], cls.BALANCED["name"], cls.MAXIMUM["name"]]
    
    @classmethod
    def get_preset(cls, name: str) -> Dict:
        """Get preset configuration by name."""
        presets = {
            cls.QUICK["name"]: cls.QUICK,
            cls.BALANCED["name"]: cls.BALANCED,
            cls.MAXIMUM["name"]: cls.MAXIMUM
        }
        return presets.get(name, cls.BALANCED)


@dataclass
class UIConfig:
    """UI/UX configuration settings."""
    # Chat settings
    DEFAULT_CHAT_TEMPERATURE: float = 0.3
    DEFAULT_MAX_TOKENS: int = 150  # Reduced from 200 to prevent <unk>
    MAX_TOKENS_LIMIT: int = 500
    MIN_TOKENS_LIMIT: int = 10
    
    # Response timeout (seconds)
    CHAT_TIMEOUT_SECONDS: int = 60
    
    # Log display settings
    MAX_LOG_HISTORY_ENTRIES: int = 500  # Total entries to keep in memory
    LOG_DISPLAY_LINES: int = 50  # Lines shown in UI
    
    # Data preview settings
    PREVIEW_TEXT_LENGTH: int = 100


@dataclass
class StudioModelConfig:
    """
    EdukaAI Studio - Simplified Model Configuration
    
    5 default models with per-model presets for beginner-friendly interface.
    Each model has Quick/Balanced/Maximum presets optimized for that specific model.
    Users can add more models by extending this configuration.
    
    Usage:
        from config import STUDIO_MODELS
        model = STUDIO_MODELS["phi-3-mini"]
        preset = model["presets"]["balanced"]
    """
    
    # 5 Default Models for EdukaAI Studio
    DEFAULT_MODELS: Dict = field(default_factory=lambda: {
        "phi-3-mini": {
            "id": "phi-3-mini",
            "name": "Phi-3 Mini",
            "description": "Quick & Light - Best for testing and fast iteration",
            "model_id": "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "size": "3.8B",
            "best_for": "Testing, learning, quick experiments",
            "presets": {
                "quick": {
                    "name": "Quick",
                    "description": "200 steps, ~20 minutes",
                    "iterations": 200,
                    "learning_rate": "5e-4",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "grad_accumulation": 16,
                    "estimated_time": "~20 min"
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "600 steps, ~45 minutes",
                    "iterations": 600,
                    "learning_rate": "1e-4",
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "grad_accumulation": 32,
                    "estimated_time": "~45 min"
                },
                "maximum": {
                    "name": "Maximum",
                    "description": "1000 steps, ~75 minutes",
                    "iterations": 1000,
                    "learning_rate": "5e-5",
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "grad_accumulation": 64,
                    "estimated_time": "~75 min"
                }
            }
        },
        "mistral-7b": {
            "id": "mistral-7b",
            "name": "Mistral 7B",
            "description": "Balanced Quality - Best for most use cases",
            "model_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "size": "7B",
            "best_for": "General tasks, good quality/speed ratio",
            "presets": {
                "quick": {
                    "name": "Quick",
                    "description": "200 steps, ~30 minutes",
                    "iterations": 200,
                    "learning_rate": "5e-4",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "grad_accumulation": 16,
                    "estimated_time": "~30 min"
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "600 steps, ~60 minutes",
                    "iterations": 600,
                    "learning_rate": "1e-4",
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "grad_accumulation": 32,
                    "estimated_time": "~60 min"
                },
                "maximum": {
                    "name": "Maximum",
                    "description": "1000 steps, ~100 minutes",
                    "iterations": 1000,
                    "learning_rate": "5e-5",
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "grad_accumulation": 64,
                    "estimated_time": "~100 min"
                }
            }
        },
        "llama-3.2-3b": {
            "id": "llama-3.2-3b",
            "name": "Llama 3.2 3B",
            "description": "Efficient & Modern - Latest Llama with great performance",
            "model_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "size": "3B",
            "best_for": "Fast training with latest Llama architecture",
            "presets": {
                "quick": {
                    "name": "Quick",
                    "description": "200 steps, ~20 minutes",
                    "iterations": 200,
                    "learning_rate": "5e-4",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "grad_accumulation": 16,
                    "estimated_time": "~20 min"
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "600 steps, ~45 minutes",
                    "iterations": 600,
                    "learning_rate": "1e-4",
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "grad_accumulation": 32,
                    "estimated_time": "~45 min"
                },
                "maximum": {
                    "name": "Maximum",
                    "description": "1000 steps, ~75 minutes",
                    "iterations": 1000,
                    "learning_rate": "5e-5",
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "grad_accumulation": 64,
                    "estimated_time": "~75 min"
                }
            }
        },
        "qwen-2.5-7b": {
            "id": "qwen-2.5-7b",
            "name": "Qwen 2.5 7B",
            "description": "Multilingual - Supports multiple languages",
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "size": "7B",
            "best_for": "Multilingual tasks, non-English data",
            "presets": {
                "quick": {
                    "name": "Quick",
                    "description": "200 steps, ~30 minutes",
                    "iterations": 200,
                    "learning_rate": "5e-4",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "grad_accumulation": 16,
                    "estimated_time": "~30 min"
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "600 steps, ~60 minutes",
                    "iterations": 600,
                    "learning_rate": "1e-4",
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "grad_accumulation": 32,
                    "estimated_time": "~60 min"
                },
                "maximum": {
                    "name": "Maximum",
                    "description": "1000 steps, ~100 minutes",
                    "iterations": 1000,
                    "learning_rate": "5e-5",
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "grad_accumulation": 64,
                    "estimated_time": "~100 min"
                }
            }
        },
        "gemma-3-4b": {
            "id": "gemma-3-4b",
            "name": "Gemma 3 4B",
            "description": "Efficient - Fast and lightweight",
            "model_id": "mlx-community/gemma-3-4b-it-4bit",
            "size": "4B",
            "best_for": "Fast training, low memory usage",
            "presets": {
                "quick": {
                    "name": "Quick",
                    "description": "200 steps, ~15 minutes",
                    "iterations": 200,
                    "learning_rate": "5e-4",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "grad_accumulation": 16,
                    "estimated_time": "~15 min"
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "600 steps, ~40 minutes",
                    "iterations": 600,
                    "learning_rate": "1e-4",
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "grad_accumulation": 32,
                    "estimated_time": "~40 min"
                },
                "maximum": {
                    "name": "Maximum",
                    "description": "1000 steps, ~65 minutes",
                    "iterations": 1000,
                    "learning_rate": "5e-5",
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "grad_accumulation": 64,
                    "estimated_time": "~65 min"
                }
            }
        }
    })
    
    # Default model for Studio
    DEFAULT_STUDIO_MODEL: str = field(default="phi-3-mini")
    
    # Learning rate modes (maps to actual values)
    LEARNING_RATE_MODES: Dict = field(default_factory=lambda: {
        "conservative": "5e-5",  # Safer, slower learning
        "balanced": "1e-4",      # Optimal for most cases
        "aggressive": "5e-4"     # Faster learning, may overfit
    })
    
    # Advanced settings ranges
    LORA_RANK_OPTIONS: List[int] = field(default_factory=lambda: [8, 16, 32])
    GRAD_ACCUM_OPTIONS: List[int] = field(default_factory=lambda: [16, 32, 64])
    MAX_SEQ_OPTIONS: List[int] = field(default_factory=lambda: [512, 2048, 4096])
    
    def get_model(self, model_id: str) -> Dict:
        """Get model configuration by ID."""
        return self.DEFAULT_MODELS.get(model_id, {})
    
    def get_preset(self, model_id: str, preset_name: str) -> Dict:
        """Get preset configuration for a specific model."""
        model = self.get_model(model_id)
        presets = model.get("presets", {})
        return presets.get(preset_name, {})
    
    def get_all_models(self) -> List[Dict]:
        """Get list of all available models."""
        return list(self.DEFAULT_MODELS.values())


@dataclass
class SystemPromptConfig:
    """System prompt configuration."""
    DEFAULT: str = "You are a helpful assistant trained on specific data. Answer questions accurately based on your training."
    
    @classmethod
    def get_preset(cls, name: str) -> str:
        """Get system prompt preset by name (only Default available)."""
        return cls.DEFAULT


@dataclass
class SystemConfig:
    """System and monitoring settings."""
    # Polling intervals (seconds)
    TRAINING_TIMER_INTERVAL: float = 1.0
    STATUS_TIMER_INTERVAL: float = 5.0


@dataclass
class ServerConfig:
    """Gradio server configuration."""
    # Server binding - restricted to localhost for security
    HOST: str = "127.0.0.1"  # Bind only to localhost (secure, not accessible from network)
    PORT: int = 7860  # Default Gradio port
    
    # Display settings
    SHOW_LOCALHOST_URL: bool = True
    LOCALHOST_URL: str = "http://localhost:7860"
    
    # Server options
    SHARE: bool = False  # Whether to create public link
    QUIET: bool = False  # Suppress startup messages


# Create global instances
TRAINING = TrainingConfig()
PRESETS = TrainingPresets()
UI = UIConfig()
SYSTEM_PROMPTS = SystemPromptConfig()
SYSTEM = SystemConfig()
SERVER = ServerConfig()

# EdukaAI Studio simplified models with per-model presets
STUDIO_MODELS = StudioModelConfig()


def get_config_summary() -> str:
    """Generate a summary of current configuration."""
    summary = []
    summary.append("=" * 70)
    summary.append("CURRENT SYSTEM CONFIGURATION")
    summary.append("=" * 70)
    
    summary.append(f"\n🎯 TRAINING:")
    summary.append(f"   Default iterations: {TRAINING.DEFAULT_ITERATIONS}")
    summary.append(f"   Default learning rate: {TRAINING.DEFAULT_LEARNING_RATE}")
    summary.append(f"   LoRA rank/alpha: {TRAINING.DEFAULT_LORA_RANK}/{TRAINING.DEFAULT_LORA_ALPHA}")
    summary.append(f"   Gradient accumulation: {TRAINING.DEFAULT_GRAD_ACCUMULATION}")
    
    summary.append(f"\n⚡ PRESETS:")
    for preset_name in PRESETS.get_preset_names():
        preset = PRESETS.get_preset(preset_name)
        summary.append(f"   {preset_name}: {preset['iterations']} iters")
    
    summary.append(f"\n💬 CHAT:")
    summary.append(f"   Temperature: {UI.DEFAULT_CHAT_TEMPERATURE}")
    summary.append(f"   Max tokens: {UI.DEFAULT_MAX_TOKENS}")
    summary.append(f"   Timeout: {UI.CHAT_TIMEOUT_SECONDS}s")
    
    summary.append(f"\n🤖 MODELS:")
    summary.append(f"   Default: {STUDIO_MODELS.DEFAULT_STUDIO_MODEL}")
    summary.append(f"   Available: {len(STUDIO_MODELS.get_all_models())} models")
    
    summary.append(f"\n🎭 SYSTEM PROMPT:")
    summary.append(f"   Default: {SYSTEM_PROMPTS.DEFAULT[:50]}...")
    
    summary.append(f"\n⏱️ SYSTEM:")
    summary.append(f"   Training poll interval: {SYSTEM.TRAINING_TIMER_INTERVAL}s")
    summary.append(f"   Status check interval: {SYSTEM.STATUS_TIMER_INTERVAL}s")
    
    summary.append("=" * 70)
    
    return "\n".join(summary)


if __name__ == "__main__":
    print(get_config_summary())
