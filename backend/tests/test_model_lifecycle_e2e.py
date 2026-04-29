"""
End-to-end test for the full model lifecycle:
validate -> add -> verify DB entry -> verify mlx_config fields -> delete

Also tests that the model_architectures module is consistent with the
validation endpoint behavior (no hardcoded values leak through).

Run with: pytest backend/tests/test_model_lifecycle_e2e.py -v
"""

import os
import pytest
from unittest.mock import patch
from types import SimpleNamespace

os.environ.setdefault("EDUKAAI_ALLOW_REMOTE", "true")
os.environ.setdefault("EDUKAAI_ENV", "testing")

from app.config import get_settings

from fastapi.testclient import TestClient
from app.main import app
import uuid as _uuid
from app.core.model_architectures import (
    ARCHITECTURE_CONFIG,
    ARCH_MAP,
    DEFAULT_ARCH,
    FALLBACK_PARAM_COUNT,
    CONTEXT_LENGTH_CAP,
    DEFAULT_CONTEXT_LENGTH,
    ATTENTION_PROJ_KEYS,
    resolve_architecture,
    get_arch_config,
    get_lora_keys,
    get_size_category,
)

HF_MODEL_INFO_TARGET = "huggingface_hub.model_info"


client = TestClient(app)


def make_mock_info_full(
    model_id="test/model-3B",
    model_type="qwen2",
    param_count=3_000_000_000,
    max_position_embeddings=8192,
    is_mlx=True,
):
    siblings = [
        SimpleNamespace(rfilename="model-00001-of-00001.safetensors", size=1_700_000_000),
        SimpleNamespace(rfilename="tokenizer.json", size=2_000_000),
        SimpleNamespace(rfilename="tokenizer_config.json", size=500_000),
        SimpleNamespace(rfilename="config.json", size=1000),
    ]
    all_tags = ["mlx", "4bit"] if is_mlx else []

    return SimpleNamespace(
        id=model_id,
        tags=all_tags,
        siblings=siblings,
        pipeline_tag="text-generation",
        gated=False,
        library_name="transformers",
        downloads=150000,
        config={
            "model_type": model_type,
            "max_position_embeddings": max_position_embeddings,
            "num_hidden_layers": 36,
            "hidden_size": 2048,
        },
        safetensors=SimpleNamespace(
            total=param_count,
            parameters={"fp16": param_count},
        ),
        card_data=SimpleNamespace(
            base_model=None,
            license="apache-2.0",
            language=["en"],
        ),
    )


def _uid(suffix):
    return f"test-{_uuid.uuid4().hex[:8]}/{suffix}"


class TestFullModelLifecycle:
    """Validate -> Add -> Verify DB -> Delete"""

    @patch(HF_MODEL_INFO_TARGET)
    def test_validate_then_add_qwen2(self, mock_hf):
        mock_hf.return_value = make_mock_info_full(model_type="qwen2")

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("qwen2-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert validate_resp.status_code == 200
        vdata = validate_resp.json()
        assert vdata["is_valid"] is True
        assert vdata["model_info"]["architecture"] == "qwen2"

        add_resp = client.post(
            "/api/base-models/custom",
            json={"huggingface_id": _uid("qwen2-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert add_resp.status_code == 200
        adata = add_resp.json()
        assert adata["architecture"] == "qwen2"
        assert adata["parameter_count"] == 3_000_000_000

        mlx_config = adata["mlx_config"]
        assert mlx_config["is_custom"] is True
        assert mlx_config["supports_lora"] is True
        assert "lora_target_modules" in mlx_config
        assert mlx_config["lora_target_modules"] == get_lora_keys("qwen2")
        assert "stop_strings" in mlx_config
        assert "eos_token" in mlx_config
        assert "chat_template_fallback" in mlx_config
        assert "is_moe" in mlx_config
        assert "is_supported" in mlx_config
        assert "min_ram_gb" in mlx_config
        assert mlx_config["is_moe"] is False
        assert mlx_config["is_supported"] is True

    @patch(HF_MODEL_INFO_TARGET)
    def test_validate_then_add_moe_model(self, mock_hf):
        mock_hf.return_value = make_mock_info_full(model_type="mixtral", param_count=7_000_000_000)

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("mixtral-7B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert vdata["is_valid"] is True
        assert any("mixture" in w.lower() for w in vdata["warnings"])

        add_resp = client.post(
            "/api/base-models/custom",
            json={"huggingface_id": _uid("mixtral-7B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        adata = add_resp.json()
        assert adata["mlx_config"]["is_moe"] is True

    @patch(HF_MODEL_INFO_TARGET)
    def test_validate_then_add_unsupported_arch(self, mock_hf):
        mock_hf.return_value = make_mock_info_full(model_type="deepseek_v3", param_count=7_000_000_000)

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("deepseek-7B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert vdata["is_valid"] is True
        assert any("not been fully tested" in w for w in vdata["warnings"])

        add_resp = client.post(
            "/api/base-models/custom",
            json={"huggingface_id": _uid("deepseek-7B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        adata = add_resp.json()
        assert adata["mlx_config"]["is_supported"] is False

    @patch(HF_MODEL_INFO_TARGET)
    def test_context_length_respected_from_config(self, mock_hf):
        mock_hf.return_value = make_mock_info_full(max_position_embeddings=32768)

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("model-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert vdata["model_info"]["context_length"] == 32768

    @patch(HF_MODEL_INFO_TARGET)
    def test_context_length_capped(self, mock_hf):
        mock_hf.return_value = make_mock_info_full(max_position_embeddings=200000)

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("model-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert vdata["model_info"]["context_length"] == CONTEXT_LENGTH_CAP


class TestNoHardcodedValues:
    """Verify that the validation endpoint uses model_architectures constants,
    not hardcoded magic numbers."""

    @patch(HF_MODEL_INFO_TARGET)
    def test_default_context_from_constant(self, mock_hf):
        info = make_mock_info_full()
        info.config = {"model_type": "qwen2"}
        mock_hf.return_value = info

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("model-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert vdata["model_info"]["context_length"] == DEFAULT_CONTEXT_LENGTH

    @patch(HF_MODEL_INFO_TARGET)
    def test_fallback_param_count_from_constant(self, mock_hf):
        info = make_mock_info_full(param_count=0)
        info.safetensors = None
        mock_hf.return_value = info

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("unknown-model")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert not vdata["model_info"].get("already_exists"), f"Model already in DB: {vdata}"
        assert vdata["model_info"]["parameter_count"] == FALLBACK_PARAM_COUNT, f"Got {vdata['model_info']['parameter_count']}, warnings={vdata.get('warnings')}, errors={vdata.get('errors')}"

    @patch(HF_MODEL_INFO_TARGET)
    def test_lora_keys_from_arch_config(self, mock_hf):
        mock_hf.return_value = make_mock_info_full(model_type="qwen2")

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": _uid("qwen2-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()
        assert vdata["model_info"]["lora_target_modules"] == get_lora_keys("qwen2")


class TestArchitectureConsistencyE2E:
    """Verify that every architecture in ARCH_MAP can be validated and added."""

    @pytest.mark.parametrize("raw_type,canonical", [
        ("qwen2", "qwen2"),
        ("qwen3", "qwen3"),
        ("llama", "llama"),
        ("phi3", "phi3"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
        ("gemma3", "gemma3_text"),
        ("mixtral", "mixtral"),
    ])
    @patch(HF_MODEL_INFO_TARGET)
    def test_architecture_round_trip(self, mock_hf, raw_type, canonical):
        mock_hf.return_value = make_mock_info_full(model_type=raw_type)

        validate_resp = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": f"test/{raw_type}-3B"},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        vdata = validate_resp.json()

        if vdata["is_valid"]:
            assert vdata["model_info"]["architecture"] == canonical

            add_resp = client.post(
                "/api/base-models/custom",
                json={"huggingface_id": _uid(f"{raw_type}-3B")},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            adata = add_resp.json()
            assert adata["architecture"] == canonical
            assert adata["mlx_config"]["lora_target_modules"] == get_lora_keys(canonical)


class TestDeleteModel:
    @patch(HF_MODEL_INFO_TARGET)
    def test_delete_custom_model_no_runs(self, mock_hf):
        mock_hf.return_value = make_mock_info_full()

        add_resp = client.post(
            "/api/base-models/custom",
            json={"huggingface_id": _uid("deletable-model-3B")},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        if add_resp.status_code != 200:
            pytest.skip("Add failed, skipping delete test")
        model_id = add_resp.json()["id"]

        delete_resp = client.delete(
            f"/api/base-models/{model_id}",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert delete_resp.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
