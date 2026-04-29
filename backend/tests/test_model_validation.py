"""
Functional tests for model validation endpoint.
Tests the validate_custom_model and add_custom_model API endpoints
with mocked HuggingFace API calls.
Run with: pytest backend/tests/test_model_validation.py -v
"""

import os
import pytest
from unittest.mock import patch
from types import SimpleNamespace

os.environ.setdefault("EDUKAAI_ALLOW_REMOTE", "true")
os.environ.setdefault("EDUKAAI_ENV", "testing")

from app.config import get_settings, Settings

from fastapi.testclient import TestClient
from app.main import app
from app.core.model_architectures import (
    ARCHITECTURE_CONFIG, DEFAULT_ARCH, FALLBACK_PARAM_COUNT,
    CONTEXT_LENGTH_CAP, DEFAULT_CONTEXT_LENGTH,
)

HF_MODEL_INFO_TARGET = "huggingface_hub.model_info"


client = TestClient(app)


def make_mock_info(
    model_id="test/model-3B",
    model_type="qwen2",
    has_safetensors=True,
    has_tokenizer=True,
    has_chat_template=True,
    pipeline_tag="text-generation",
    is_gated=False,
    is_mlx=True,
    max_position_embeddings=8192,
    num_hidden_layers=36,
    hidden_size=2048,
    safetensors_total=3_000_000_000,
    safetensors_params=None,
    tags=None,
    license_name="apache-2.0",
    base_model=None,
    languages=None,
):
    siblings = []
    if has_safetensors:
        siblings.append(SimpleNamespace(rfilename="model-00001-of-00001.safetensors", size=1_700_000_000))
    if has_tokenizer:
        siblings.append(SimpleNamespace(rfilename="tokenizer.json", size=2_000_000))
        if has_chat_template:
            siblings.append(SimpleNamespace(rfilename="tokenizer_config.json", size=500_000))

    all_tags = list(tags or [])
    if is_mlx:
        all_tags.extend(["mlx", "4bit"])

    return SimpleNamespace(
        id=model_id,
        tags=all_tags,
        siblings=siblings,
        pipeline_tag=pipeline_tag,
        gated=is_gated,
        library_name="transformers",
        downloads=150000,
        config={
            "model_type": model_type,
            "max_position_embeddings": max_position_embeddings,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
        },
        safetensors=SimpleNamespace(
            total=safetensors_total,
            parameters=safetensors_params,
        ) if has_safetensors else None,
        card_data=SimpleNamespace(
            base_model=base_model,
            license=license_name,
            language=languages or ["en"],
        ) if license_name else None,
    )


class TestValidateEndpointFormat:
    def test_invalid_format_spaces(self):
        response = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": "invalid model name"},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False

    def test_invalid_format_empty(self):
        response = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": ""},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code in [200, 422]

    def test_invalid_format_path_traversal(self):
        response = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": "../../../etc/passwd"},
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False

    def test_valid_format(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info()
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "mlx-community/Qwen2.5-3B-Instruct-4bit"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert response.status_code == 200

    def test_url_normalization(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_id="mlx-community/Qwen2.5-3B-Instruct-4bit")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert response.status_code == 200


class TestValidateEndpointHFCallSplit:
    def test_expand_and_files_metadata_not_combined(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_info = make_mock_info()
            mock_files = make_mock_info()
            mock_hf.side_effect = [mock_info, mock_files]

            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert response.status_code == 200
            assert mock_hf.call_count >= 1

            for call_args in mock_hf.call_args_list:
                kwargs = call_args.kwargs
                has_expand = "expand" in kwargs and kwargs["expand"]
                has_files_meta = "files_metadata" in kwargs and kwargs["files_metadata"] is True
                assert not (has_expand and has_files_meta), (
                    f"hf_model_info called with both expand and files_metadata=True: {kwargs}"
                )


class TestValidateEndpointVerificationChecklist:
    def test_no_safetensors_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(has_safetensors=False)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is False
            assert any("safetensors" in e.lower() for e in data["errors"])

    def test_no_tokenizer_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(has_tokenizer=False)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is False
            assert any("tokenizer" in e.lower() for e in data["errors"])

    def test_wrong_pipeline_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(pipeline_tag="image-classification")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is False
            assert any("pipeline" in e.lower() for e in data["errors"])

    def test_gated_without_token_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf, \
             patch("app.routers.training.get_settings", return_value=Settings(
                 allow_remote=True, debug=True, secret_key="test", hf_token=None
             )):
            mock_hf.return_value = make_mock_info(is_gated=True)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is False
            assert any("gated" in e.lower() for e in data["errors"])

    def test_no_chat_template_warns_not_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(has_chat_template=False)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is True
            assert any("chat template" in w.lower() for w in data["warnings"])

    def test_non_mlx_warns_not_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(is_mlx=False)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is True
            assert any("mlx" in w.lower() for w in data["warnings"])

    def test_moe_model_warns_not_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type="mixtral")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/mixtral-model"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is True
            assert any("mixture" in w.lower() for w in data["warnings"])

    def test_unsupported_arch_warns_not_blocks(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type="deepseek_v3")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/deepseek-model"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is True
            assert any("not been fully tested" in w for w in data["warnings"])

    def test_valid_model_passes(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type="qwen2")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "mlx-community/Qwen2.5-3B-Instruct-4bit"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["is_valid"] is True


class TestValidateEndpointResponseFields:
    def test_response_has_is_moe(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type="qwen2")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert "is_moe" in data["model_info"]

    def test_response_has_is_supported(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type="qwen2")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert "is_supported" in data["model_info"]

    def test_response_has_min_ram_gb(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type="qwen2")
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert "min_ram_gb" in data["model_info"]

    def test_context_length_capped(self):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(max_position_embeddings=100000)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": "test/model-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            assert data["model_info"]["context_length"] <= CONTEXT_LENGTH_CAP


class TestArchitectureResolution:
    @pytest.mark.parametrize("model_type,expected_arch", [
        ("qwen2", "qwen2"),
        ("qwen3", "qwen3"),
        ("llama", "llama"),
        ("phi3", "phi3"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
        ("gemma3", "gemma3_text"),
        ("mixtral", "mixtral"),
        ("qwen2_moe", "qwen2_moe"),
    ])
    def test_architecture_resolution(self, model_type, expected_arch):
        with patch(HF_MODEL_INFO_TARGET) as mock_hf:
            mock_hf.return_value = make_mock_info(model_type=model_type)
            response = client.post(
                "/api/base-models/validate",
                json={"huggingface_id": f"test/{model_type}-3B"},
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            data = response.json()
            if data["is_valid"]:
                assert data["model_info"]["architecture"] == expected_arch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
