"""
Per-architecture model configuration for MLX LoRA fine-tuning.

Single source of truth for LoRA target modules, stop strings,
chat template fallbacks, recommended training presets, and
architecture support tiers.

Tier 1: Fully tested, known LoRA keys, known templates
Tier 2: MLX-native, LoRA keys derivable, needs testing
Tier 3: MLX supports it, we haven't tested LoRA (allow with warnings)

Adding a new architecture:
1. Verify MLX support: python -c "import mlx_lm.models.{model_type}"
2. Add entry to ARCHITECTURE_CONFIG with all fields
3. Add mapping to ARCH_MAP
4. Set is_supported=True only after testing LoRA on it
5. Test: validate a model -> add it -> train 50 steps -> verify output
"""

import importlib
from typing import Dict, List, Optional, Any

ATTENTION_PROJ_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]

DEFAULT_ARCH = "qwen2"

ARCHITECTURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "llama": {
        "mlx_module": "llama",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<|eot_id|>", "<|end_of_text|>"],
        "eos_token": "<|eot_id|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n"
            "{{ message.content }}<|eot_id|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "tiny":   {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "small":  {"lora_rank": 8,  "lora_layers": 16, "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
            "large":  {"lora_rank": 32, "lora_layers": 16, "learning_rate": 1e-5, "batch_size": 1, "gradient_checkpointing": True},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "qwen2": {
        "mlx_module": "qwen2",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<|im_end|>"],
        "eos_token": "<|im_end|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "tiny":   {"lora_rank": 8,  "lora_layers": 16, "learning_rate": 1e-4, "batch_size": 4},
            "small":  {"lora_rank": 8,  "lora_layers": 16, "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
            "large":  {"lora_rank": 32, "lora_layers": 16, "learning_rate": 2e-5, "batch_size": 1, "gradient_checkpointing": True},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "phi3": {
        "mlx_module": "phi3",
        "lora_keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "stop_strings": ["<|end|>"],
        "eos_token": "<|end|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|{{ message.role }}|>\n{{ message.content }}<|end|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "gemma": {
        "mlx_module": "gemma2",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<end_of_turn>"],
        "eos_token": "<end_of_turn>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<start_of_turn>model\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "mistral": {
        "mlx_module": "llama",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["</s>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% if messages[0]['role'] == 'system' %}"
            "[INST] {{ messages[0]['content'] }} [/INST]"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "[INST] {{ message['content'] }} [/INST]"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content']}}</s>"
            "{% endif %}"
            "{% endfor %}"
        ),
        "recommended_presets": {
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
            "large":  {"lora_rank": 32, "lora_layers": 16, "learning_rate": 1e-5, "batch_size": 1, "gradient_checkpointing": True},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "qwen3": {
        "mlx_module": "qwen3",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<|im_end|>"],
        "eos_token": "<|im_end|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "tiny":   {"lora_rank": 8,  "lora_layers": 16, "learning_rate": 1e-4, "batch_size": 4},
            "small":  {"lora_rank": 8,  "lora_layers": 16, "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
            "large":  {"lora_rank": 32, "lora_layers": 16, "learning_rate": 2e-5, "batch_size": 1, "gradient_checkpointing": True},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "qwen3_5": {
        "mlx_module": "qwen3_5",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<|im_end|>"],
        "eos_token": "<|im_end|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 16, "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": False,
        "min_ram_gb": 8,
    },
    "gemma3_text": {
        "mlx_module": "gemma3_text",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<end_of_turn>"],
        "eos_token": "<end_of_turn>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<start_of_turn>model\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "gemma4": {
        "mlx_module": "gemma4",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<end_of_turn>"],
        "eos_token": "<end_of_turn>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<start_of_turn>model\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
            "large":  {"lora_rank": 32, "lora_layers": 16, "learning_rate": 1e-5, "batch_size": 1, "gradient_checkpointing": True},
        },
        "is_moe": True,
        "is_supported": True,
        "min_ram_gb": 24,
    },
    "internlm2": {
        "mlx_module": "internlm2",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["</s>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "minicpm": {
        "mlx_module": "minicpm",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["</s>", "<|end|>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "olmo": {
        "mlx_module": "olmo",
        "lora_keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "stop_strings": ["</s>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "small":  {"lora_rank": 8,  "lora_layers": 8,  "learning_rate": 1e-4, "batch_size": 4},
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": False,
        "is_supported": True,
        "min_ram_gb": 8,
    },
    "mixtral": {
        "mlx_module": "mixtral",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["</s>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% if messages[0]['role'] == 'system' %}"
            "[INST] {{ messages[0]['content'] }} [/INST]"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "[INST] {{ message['content'] }} [/INST]"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content']}}</s>"
            "{% endif %}"
            "{% endfor %}"
        ),
        "recommended_presets": {
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
            "large":  {"lora_rank": 32, "lora_layers": 16, "learning_rate": 1e-5, "batch_size": 1, "gradient_checkpointing": True},
        },
        "is_moe": True,
        "is_supported": True,
        "min_ram_gb": 16,
    },
    "qwen2_moe": {
        "mlx_module": "qwen2_moe",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<|im_end|>"],
        "eos_token": "<|im_end|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": True,
        "is_supported": True,
        "min_ram_gb": 16,
    },
    "qwen3_moe": {
        "mlx_module": "qwen3_moe",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["<|im_end|>"],
        "eos_token": "<|im_end|>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": True,
        "is_supported": True,
        "min_ram_gb": 16,
    },
    "deepseek_v2": {
        "mlx_module": "deepseek_v2",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["</s>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": True,
        "is_supported": False,
        "min_ram_gb": 32,
    },
    "deepseek_v3": {
        "mlx_module": "deepseek_v3",
        "lora_keys": list(ATTENTION_PROJ_KEYS),
        "stop_strings": ["</s>"],
        "eos_token": "</s>",
        "chat_template_fallback": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        ),
        "recommended_presets": {
            "medium": {"lora_rank": 16, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 2},
        },
        "is_moe": True,
        "is_supported": False,
        "min_ram_gb": 32,
    },
}

ARCH_MAP: Dict[str, str] = {
    "llama": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "llama4": "llama",
    "mistral": "mistral",
    "mistral3": "mistral",
    "mixtral": "mixtral",
    "qwen": "qwen2",
    "qwen2": "qwen2",
    "qwen2.5": "qwen2",
    "qwen25": "qwen2",
    "qwen3": "qwen3",
    "qwen3.5": "qwen3_5",
    "qwen35": "qwen3_5",
    "qwen2_moe": "qwen2_moe",
    "qwen2moe": "qwen2_moe",
    "qwen3_moe": "qwen3_moe",
    "qwen3moe": "qwen3_moe",
    "qwen3_5_moe": "qwen3_moe",
    "qwen35moe": "qwen3_moe",
    "phi": "phi3",
    "phi3": "phi3",
    "phi3small": "phi3",
    "phimoe": "phi3",
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma3_text",
    "gemma3_text": "gemma3_text",
    "gemma3text": "gemma3_text",
    "gemma3n": "gemma3_text",
    "gemma4": "gemma4",
    "gemma4text": "gemma4",
    "gemma4_text": "gemma4",
    "internlm2": "internlm2",
    "minicpm": "minicpm",
    "minicpm3": "minicpm",
    "olmo": "olmo",
    "olmoe": "olmo",
    "deepseek": "deepseek_v3",
    "deepseek_v2": "deepseek_v2",
    "deepseekv2": "deepseek_v2",
    "deepseek_v3": "deepseek_v3",
    "deepseekv3": "deepseek_v3",
    "cohere": "llama",
    "dbrx": "mixtral",
    "stablelm": "llama",
    "starcoder2": "llama",
}

SIZE_CATEGORIES: Dict[str, int] = {
    "tiny": 500_000_000,
    "small": 2_000_000_000,
    "medium": 5_000_000_000,
    "large": 10_000_000_000,
}

PARAM_COUNT_DEFAULTS: Dict[str, int] = {
    "0.5b": 500_000_000,
    "1b": 1_000_000_000,
    "1.5b": 1_500_000_000,
    "2b": 2_000_000_000,
    "3b": 3_000_000_000,
    "4b": 4_000_000_000,
    "7b": 7_000_000_000,
    "8b": 8_000_000_000,
    "13b": 13_000_000_000,
    "26b": 26_000_000_000,
}

FALLBACK_PARAM_COUNT = 3_000_000_000

CONTEXT_LENGTH_CAP = 32768

DEFAULT_CONTEXT_LENGTH = 8192


def resolve_architecture(model_type: Optional[str]) -> str:
    """Map raw model_type to canonical architecture name."""
    if not model_type:
        return DEFAULT_ARCH
    key = model_type.lower().replace("-", "").replace("_", "")
    return ARCH_MAP.get(key, ARCH_MAP.get(model_type.lower(), DEFAULT_ARCH))


def get_arch_config(architecture: str) -> Dict[str, Any]:
    """Get config for a canonical architecture. Falls back to default."""
    return ARCHITECTURE_CONFIG.get(architecture, ARCHITECTURE_CONFIG[DEFAULT_ARCH])


def get_lora_keys(architecture: str) -> List[str]:
    """Get LoRA target module keys for an architecture."""
    return list(get_arch_config(architecture)["lora_keys"])


def get_stop_strings(architecture: str) -> List[str]:
    """Get stop strings for an architecture."""
    return list(get_arch_config(architecture)["stop_strings"])


def get_chat_template_fallback(architecture: str) -> str:
    """Get Jinja chat template fallback for an architecture."""
    return get_arch_config(architecture)["chat_template_fallback"]


def get_size_category(param_count: int) -> str:
    """Classify model size from parameter count."""
    for cat, threshold in sorted(SIZE_CATEGORIES.items(), key=lambda x: x[1]):
        if param_count <= threshold:
            return cat
    return "large"


def get_recommended_preset(architecture: str, param_count: int) -> Dict[str, Any]:
    """Get recommended training preset for a model."""
    config = get_arch_config(architecture)
    presets = config.get("recommended_presets", {})
    size = get_size_category(param_count)
    for cat in [size, "small"]:
        if cat in presets:
            return dict(presets[cat])
    if presets:
        return dict(list(presets.values())[0])
    return {"lora_rank": 8, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 4}


def is_mlx_supported(model_type: str) -> bool:
    """Check if model_type has a corresponding mlx_lm.models module."""
    if not model_type:
        return False
    arch = resolve_architecture(model_type)
    mlx_module = get_arch_config(arch).get("mlx_module", arch)
    try:
        importlib.import_module(f"mlx_lm.models.{mlx_module}")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def get_param_count_from_name(huggingface_id: str) -> int:
    """Extract parameter count from model ID string (e.g., '3B' -> 3_000_000_000)."""
    import re as _re
    match = _re.search(r'(\d+\.?\d*)[Bb]', huggingface_id)
    if match:
        return int(float(match.group(1)) * 1_000_000_000)
    for key, value in PARAM_COUNT_DEFAULTS.items():
        if key in huggingface_id.lower():
            return value
    return 0


def detect_architecture_from_id(huggingface_id: str, tags: Optional[List[str]] = None) -> Optional[str]:
    """Try to detect architecture from model ID and tags when config.model_type is absent."""
    id_lower = huggingface_id.lower()
    tag_lower = [t.lower() for t in (tags or [])]
    for raw_type, canonical in ARCH_MAP.items():
        if raw_type in id_lower or any(raw_type in t for t in tag_lower):
            return raw_type
    return None


def validate_lora_keys_against_model(model, stored_keys: List[str]) -> List[str]:
    """
    After model is loaded, verify stored LoRA keys exist in model layers.
    Fall back to discovered attention projection keys if mismatch.
    Never target embedding/output layers.
    """
    try:
        import mlx.nn as nn
        available_keys = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, 'input_dims'):
                available_keys.add(name)

        valid_keys = [k for k in stored_keys if k in available_keys]

        if valid_keys:
            return valid_keys

        for key in sorted(available_keys):
            if any(p in key for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                valid_keys.append(key)

        return valid_keys if valid_keys else list(available_keys)[:4]
    except Exception:
        return stored_keys


def _get_hf_cache_root() -> Path:
    """Return the active HuggingFace cache directory.
    Must respect HF_HUB_CACHE env var set in main.py before any import.
    """
    import os
    from pathlib import Path
    return Path(
        os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
    )


def _is_model_complete_sync(huggingface_id: str) -> bool:
    """Check if a model is fully downloaded in the native HF cache."""
    from pathlib import Path

    cache_root = _get_hf_cache_root()
    model_cache = cache_root / f"models--{huggingface_id.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    if not snapshots.exists():
        return False

    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        config_file = snapshot / "config.json"
        safetensors = list(snapshot.glob("*.safetensors"))
        if config_file.exists() and config_file.stat().st_size > 0 and safetensors:
            return True
    return False


def get_cached_snapshot_path_sync(huggingface_id: str) -> Optional[str]:
    """Return the snapshot directory path if model is fully cached, else None."""
    from pathlib import Path

    cache_root = _get_hf_cache_root()
    model_cache = cache_root / f"models--{huggingface_id.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    if not snapshots.exists():
        return None

    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        config_file = snapshot / "config.json"
        safetensors = list(snapshot.glob("*.safetensors"))
        if config_file.exists() and config_file.stat().st_size > 0 and safetensors:
            return str(snapshot)
    return None
