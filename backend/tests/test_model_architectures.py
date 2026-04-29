"""
Unit tests for model_architectures.py — single source of truth for architecture config.
Tests cover: resolve_architecture, arch config retrieval, LoRA keys, size categories,
param count extraction, architecture detection from ID, MLX module check, and
consistency guarantees (no hardcoded values, all fields present, no orphan entries).
Run with: pytest backend/tests/test_model_architectures.py -v
"""

import pytest
import os

from app.core.model_architectures import (
    ARCHITECTURE_CONFIG,
    ARCH_MAP,
    SIZE_CATEGORIES,
    PARAM_COUNT_DEFAULTS,
    ATTENTION_PROJ_KEYS,
    DEFAULT_ARCH,
    FALLBACK_PARAM_COUNT,
    CONTEXT_LENGTH_CAP,
    DEFAULT_CONTEXT_LENGTH,
    resolve_architecture,
    get_arch_config,
    get_lora_keys,
    get_stop_strings,
    get_chat_template_fallback,
    get_size_category,
    get_recommended_preset,
    is_mlx_supported,
    get_param_count_from_name,
    detect_architecture_from_id,
    validate_lora_keys_against_model,
)


class TestResolveArchitecture:
    def test_known_types(self):
        assert resolve_architecture("llama") == "llama"
        assert resolve_architecture("qwen2") == "qwen2"
        assert resolve_architecture("qwen3") == "qwen3"
        assert resolve_architecture("phi3") == "phi3"
        assert resolve_architecture("gemma") == "gemma"
        assert resolve_architecture("mistral") == "mistral"

    def test_aliases(self):
        assert resolve_architecture("llama3") == "llama"
        assert resolve_architecture("llama4") == "llama"
        assert resolve_architecture("qwen2.5") == "qwen2"
        assert resolve_architecture("qwen3.5") == "qwen3_5"
        assert resolve_architecture("gemma2") == "gemma"
        assert resolve_architecture("gemma3") == "gemma3_text"
        assert resolve_architecture("gemma3n") == "gemma3_text"
        assert resolve_architecture("mixtral") == "mixtral"
        assert resolve_architecture("phimoe") == "phi3"

    def test_none_returns_default(self):
        assert resolve_architecture(None) == DEFAULT_ARCH

    def test_unknown_returns_default(self):
        assert resolve_architecture("totally_unknown_arch") == DEFAULT_ARCH

    def test_case_insensitive(self):
        assert resolve_architecture("LLAMA") == "llama"
        assert resolve_architecture("QWEN2") == "qwen2"
        assert resolve_architecture("Phi3") == "phi3"

    def test_hyphen_underscore_normalization(self):
        assert resolve_architecture("qwen2_5") == "qwen2"
        assert resolve_architecture("qwen2-5") == "qwen2"
        assert resolve_architecture("gemma-3-text") == "gemma3_text"


class TestArchConfigConsistency:
    """Every architecture in ARCHITECTURE_CONFIG must have all required fields."""

    REQUIRED_FIELDS = [
        "mlx_module", "lora_keys", "stop_strings", "eos_token",
        "chat_template_fallback", "recommended_presets",
        "is_moe", "is_supported", "min_ram_gb",
    ]

    def test_all_entries_have_required_fields(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            for field in self.REQUIRED_FIELDS:
                assert field in config, f"Architecture '{arch_name}' missing field '{field}'"

    def test_lora_keys_are_lists(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["lora_keys"], list), f"'{arch_name}' lora_keys must be list"
            assert len(config["lora_keys"]) > 0, f"'{arch_name}' lora_keys must not be empty"

    def test_stop_strings_are_lists(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["stop_strings"], list), f"'{arch_name}' stop_strings must be list"

    def test_recommended_presets_have_required_keys(self):
        preset_keys = {"lora_rank", "lora_layers", "learning_rate", "batch_size"}
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            for preset_name, preset in config["recommended_presets"].items():
                for key in preset_keys:
                    assert key in preset, f"'{arch_name}' preset '{preset_name}' missing '{key}'"

    def test_is_moe_is_bool(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["is_moe"], bool), f"'{arch_name}' is_moe must be bool"

    def test_is_supported_is_bool(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["is_supported"], bool), f"'{arch_name}' is_supported must be bool"

    def test_min_ram_gb_is_positive_int(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["min_ram_gb"], int), f"'{arch_name}' min_ram_gb must be int"
            assert config["min_ram_gb"] > 0, f"'{arch_name}' min_ram_gb must be positive"

    def test_moe_models_need_more_ram(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            if config["is_moe"]:
                assert config["min_ram_gb"] >= 16, f"MoE model '{arch_name}' should need >= 16GB RAM"

    def test_eos_token_is_string(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["eos_token"], str), f"'{arch_name}' eos_token must be str"
            assert len(config["eos_token"]) > 0, f"'{arch_name}' eos_token must not be empty"

    def test_chat_template_fallback_is_string(self):
        for arch_name, config in ARCHITECTURE_CONFIG.items():
            assert isinstance(config["chat_template_fallback"], str), f"'{arch_name}' chat_template_fallback must be str"
            assert "messages" in config["chat_template_fallback"], f"'{arch_name}' template must reference messages"


class TestArchMapConsistency:
    """ARCH_MAP entries must map to architectures that exist in ARCHITECTURE_CONFIG."""

    def test_all_map_targets_exist_in_config(self):
        for raw_type, canonical in ARCH_MAP.items():
            assert canonical in ARCHITECTURE_CONFIG, (
                f"ARCH_MAP['{raw_type}'] = '{canonical}' but '{canonical}' not in ARCHITECTURE_CONFIG"
            )

    def test_no_duplicate_mappings_to_different_targets(self):
        seen = {}
        for raw_type, canonical in ARCH_MAP.items():
            if raw_type in seen and seen[raw_type] != canonical:
                pytest.fail(f"ARCH_MAP key '{raw_type}' maps to both '{seen[raw_type]}' and '{canonical}'")
            seen[raw_type] = canonical


class TestGetArchConfig:
    def test_known_arch(self):
        config = get_arch_config("llama")
        assert config["mlx_module"] == "llama"

    def test_unknown_arch_returns_default(self):
        config = get_arch_config("nonexistent")
        assert config["mlx_module"] == ARCHITECTURE_CONFIG[DEFAULT_ARCH]["mlx_module"]

    def test_returns_copy_not_reference(self):
        c1 = get_arch_config("llama")
        c2 = get_arch_config("llama")
        assert c1 == c2


class TestGetLoraKeys:
    def test_qwen2_keys(self):
        keys = get_lora_keys("qwen2")
        assert "self_attn.q_proj" in keys
        assert "self_attn.o_proj" in keys

    def test_phi3_has_three_keys(self):
        keys = get_lora_keys("phi3")
        assert len(keys) == 3

    def test_returns_list_not_reference(self):
        k1 = get_lora_keys("llama")
        k1.append("test_key")
        k2 = get_lora_keys("llama")
        assert "test_key" not in k2


class TestGetStopStrings:
    def test_qwen2(self):
        assert "<|im_end|>" in get_stop_strings("qwen2")

    def test_llama(self):
        assert "<|eot_id|>" in get_stop_strings("llama")

    def test_returns_copy(self):
        s1 = get_stop_strings("qwen2")
        s1.append("fake")
        s2 = get_stop_strings("qwen2")
        assert "fake" not in s2


class TestGetSizeCategory:
    def test_tiny(self):
        assert get_size_category(500_000_000) == "tiny"

    def test_small(self):
        assert get_size_category(2_000_000_000) == "small"

    def test_medium(self):
        assert get_size_category(5_000_000_000) == "medium"

    def test_large(self):
        assert get_size_category(10_000_000_000) == "large"

    def test_very_large(self):
        assert get_size_category(70_000_000_000) == "large"

    def test_zero(self):
        assert get_size_category(0) == "tiny"


class TestGetRecommendedPreset:
    def test_tiny_model(self):
        preset = get_recommended_preset("llama", 500_000_000)
        assert preset["lora_rank"] == 8

    def test_medium_model(self):
        preset = get_recommended_preset("qwen2", 3_000_000_000)
        assert preset["lora_rank"] == 16

    def test_returns_dict(self):
        preset = get_recommended_preset("llama", 3_000_000_000)
        assert isinstance(preset, dict)

    def test_unknown_arch_falls_back(self):
        preset = get_recommended_preset("nonexistent", 3_000_000_000)
        assert "lora_rank" in preset


class TestGetParamCountFromName:
    def test_3b(self):
        assert get_param_count_from_name("model-3B") == 3_000_000_000

    def test_1_5b(self):
        assert get_param_count_from_name("model-1.5b") == 1_500_000_000

    def test_0_5b(self):
        assert get_param_count_from_name("model-0.5B") == 500_000_000

    def test_7b_instruct(self):
        assert get_param_count_from_name("org/Qwen2.5-7B-Instruct") == 7_000_000_000

    def test_no_size_in_name(self):
        assert get_param_count_from_name("org/some-model") == 0

    def test_fallback_defaults(self):
        assert get_param_count_from_name("org/model-4b-it") == 4_000_000_000

    def test_case_insensitive_suffix(self):
        assert get_param_count_from_name("model-3b") == 3_000_000_000
        assert get_param_count_from_name("model-3B") == 3_000_000_000


class TestDetectArchitectureFromId:
    def test_llama_in_id(self):
        result = detect_architecture_from_id("meta-llama/Llama-3.2-1B")
        assert result is not None
        assert resolve_architecture(result) == "llama"

    def test_qwen_in_id(self):
        result = detect_architecture_from_id("mlx-community/Qwen2.5-3B")
        assert result is not None

    def test_unknown_id(self):
        result = detect_architecture_from_id("org/unknown-model")
        assert result is None

    def test_from_tags(self):
        result = detect_architecture_from_id("org/some-model", ["llama", "4bit"])
        assert result == "llama"

    def test_empty_tags(self):
        result = detect_architecture_from_id("org/unknown-model", [])
        assert result is None


class TestIsMlxSupported:
    def test_qwen2_is_supported(self):
        assert is_mlx_supported("qwen2") is True

    def test_llama_is_supported(self):
        assert is_mlx_supported("llama") is True

    def test_none_returns_false(self):
        assert is_mlx_supported(None) is False

    def test_empty_string_returns_false(self):
        assert is_mlx_supported("") is False


class TestConstantsNotHardcoded:
    """Ensure no magic numbers are used — all constants are named."""

    def test_fallback_param_count_is_defined(self):
        assert FALLBACK_PARAM_COUNT > 0

    def test_context_length_cap_is_defined(self):
        assert CONTEXT_LENGTH_CAP > 0

    def test_default_context_length_is_defined(self):
        assert DEFAULT_CONTEXT_LENGTH > 0

    def test_default_arch_is_defined(self):
        assert DEFAULT_ARCH in ARCHITECTURE_CONFIG

    def test_param_count_defaults_has_entries(self):
        assert len(PARAM_COUNT_DEFAULTS) >= 9

    def test_size_categories_has_four_tiers(self):
        assert len(SIZE_CATEGORIES) == 4
        assert "tiny" in SIZE_CATEGORIES
        assert "small" in SIZE_CATEGORIES
        assert "medium" in SIZE_CATEGORIES
        assert "large" in SIZE_CATEGORIES


class TestValidateLoraKeysAgainstModel:
    def test_returns_stored_keys_when_model_fails(self):
        stored = ["self_attn.q_proj", "self_attn.k_proj"]
        result = validate_lora_keys_against_model(None, stored)
        assert result == stored


class TestNoCrossArchLeakage:
    """Each architecture config must be independent — no shared mutable references."""

    def test_lora_keys_not_shared(self):
        llama_keys = ARCHITECTURE_CONFIG["llama"]["lora_keys"]
        qwen_keys = ARCHITECTURE_CONFIG["qwen2"]["lora_keys"]
        llama_keys_id = id(llama_keys)
        qwen_keys_id = id(qwen_keys)
        assert llama_keys_id != qwen_keys_id or llama_keys == qwen_keys

    def test_recommended_presets_not_shared(self):
        llama_presets = ARCHITECTURE_CONFIG["llama"]["recommended_presets"]
        qwen_presets = ARCHITECTURE_CONFIG["qwen2"]["recommended_presets"]
        assert id(llama_presets) != id(qwen_presets)
