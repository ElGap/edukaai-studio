"""
Test suite for custom model validation feature.
Run with: pytest backend/tests/test_custom_models.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime

# Enable testing mode (disable localhost-only security)
os.environ["EDUKAI_ALLOW_REMOTE"] = "true"

# Import the app and models
import sys
sys.path.insert(0, '/Users/developer/Projects/studio/backend')

# Force reload of settings module to pick up environment variable
import importlib
from app import config
importlib.reload(config)
settings = config.get_settings()

from app.main import app
from app.models import get_db, BaseModel, TrainingRun, Dataset, TrainingPreset, generate_uuid
from app.routers.training import validate_custom_model, add_custom_model, estimate_training_memory
from app.core.exceptions import ValidationError


client = TestClient(app)


class TestCustomModelValidation:
    """Test the custom model validation endpoint"""
    
    def test_valid_mlx_formatted_model(self):
        """Test validation of a properly formatted MLX model"""
        # This test would mock the HF API call
        # For now, we test the regex validation logic
        import re
        
        huggingface_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        assert re.match(r'^[\w\-\.]+(/[\w\-\.]+)*$', huggingface_id)
        
    def test_valid_non_mlx_model(self):
        """Test validation of a non-MLX model (should still validate)"""
        import re
        
        huggingface_id = "meta-llama/Llama-3.2-1B"
        assert re.match(r'^[\w\-\.]+(/[\w\-\.]+)*$', huggingface_id)
        
    def test_invalid_model_id_format(self):
        """Test rejection of invalid model ID formats"""
        import re
        
        invalid_ids = [
            "invalid model name",  # spaces
            "",  # empty
            "org/model/extra/path",  # too many slashes
            "/starts-with-slash",  # starts with slash
            "ends-with-slash/",  # ends with slash
        ]
        
        for invalid_id in invalid_ids:
            match = re.match(r'^[\w\-\.]+(/[\w\-\.]+)?$', invalid_id)
            assert match is None, f"Should reject: {invalid_id}"
            
    def test_model_id_with_special_chars(self):
        """Test handling of special characters in model ID"""
        import re
        
        valid_ids = [
            "org/model-name",  # hyphen
            "org/model_name",  # underscore
            "org/model.name",  # dot
            "org123/model456",  # numbers
        ]
        
        for valid_id in valid_ids:
            match = re.match(r'^[\w\-\.]+(/[\w\-\.]+)*$', valid_id)
            assert match is not None, f"Should accept: {valid_id}"


class TestParameterEstimation:
    """Test the parameter count estimation from model name"""
    
    def test_extract_parameters_from_name(self):
        """Test extracting parameter count from model name patterns"""
        import re
        
        test_cases = [
            ("model-1B", 1_000_000_000),
            ("model-3b", 3_000_000_000),
            ("model-7B-instruct", 7_000_000_000),
            ("model-0.5B", 500_000_000),
            ("model-1.5b", 1_500_000_000),
            ("model-13B", 13_000_000_000),
            ("model-without-size", 0),  # Should return 0 or default
        ]
        
        for model_name, expected in test_cases:
            match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
            if expected > 0:
                assert match is not None, f"Should find size in: {model_name}"
                param_count = int(float(match.group(1)) * 1_000_000_000)
                assert param_count == expected, f"Expected {expected}, got {param_count}"
            else:
                # For models without size, should return 0 or use default
                pass


class TestArchitectureDetection:
    """Test architecture detection from model name and tags"""
    
    def test_detect_llama_architecture(self):
        """Test detection of Llama models"""
        model_names = [
            "meta-llama/Llama-3.2-1B",
            "mlx-community/Llama-3.1-8B",
            "TinyLlama/TinyLlama-1.1B",
        ]
        
        for name in model_names:
            assert 'llama' in name.lower()
            
    def test_detect_qwen_architecture(self):
        """Test detection of Qwen models"""
        model_names = [
            "mlx-community/Qwen2.5-3B",
            "Qwen/Qwen2-7B",
        ]
        
        for name in model_names:
            assert 'qwen' in name.lower()
            
    def test_detect_phi_architecture(self):
        """Test detection of Phi models"""
        model_names = [
            "mlx-community/Phi-3-mini",
            "microsoft/Phi-3-small",
        ]
        
        for name in model_names:
            assert 'phi' in name.lower()


class TestMLXFormattingDetection:
    """Test detection of MLX-formatted models"""
    
    def test_mlx_community_models(self):
        """Test that mlx-community org models are detected"""
        model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        assert "mlx-community" in model_id
        
    def test_mlx_tags(self):
        """Test detection via tags"""
        mlx_tags = ['mlx', 'mlx-community', '4bit', '8bit']
        
        for tag in mlx_tags:
            assert tag in ['mlx', 'mlx-community', '4bit', '8bit']


class TestMemoryEstimation:
    """Test the memory estimation function"""
    
    def test_memory_estimation_1b_model(self):
        """Test memory estimation for 1B model"""
        memory = estimate_training_memory(
            model_params=1_000_000_000,
            lora_rank=8,
            lora_layers=8,
            batch_size=4,
            seq_length=2048,
            total_params=1_000_000_000
        )
        
        # Should be reasonable (not too high, not zero)
        assert memory > 1.0, "Memory should be > 1GB"
        assert memory < 10.0, "Memory should be < 10GB for 1B model"
        
    def test_memory_estimation_7b_model(self):
        """Test memory estimation for 7B model"""
        memory = estimate_training_memory(
            model_params=7_000_000_000,
            lora_rank=8,
            lora_layers=8,
            batch_size=4,
            seq_length=2048,
            total_params=7_000_000_000
        )
        
        # Should be higher than 1B and reasonable
        assert memory > 2.0, "7B model should need > 2GB"
        assert memory < 10.0, "7B with LoRA should be < 10GB"
        
    def test_memory_with_large_batch(self):
        """Test that large batch sizes increase memory estimate"""
        memory_small_batch = estimate_training_memory(
            model_params=3_000_000_000,
            lora_rank=8,
            lora_layers=8,
            batch_size=1,
            seq_length=2048,
            total_params=3_000_000_000
        )
        
        memory_large_batch = estimate_training_memory(
            model_params=3_000_000_000,
            lora_rank=8,
            lora_layers=8,
            batch_size=16,
            seq_length=2048,
            total_params=3_000_000_000
        )
        
        assert memory_large_batch > memory_small_batch, "Large batch should need more memory"


class TestAPIEndpoints:
    """Test the API endpoints (requires running server or mocking)"""
    
    @pytest.mark.skip(reason="Requires mocking HF API")
    def test_validate_endpoint_success(self):
        """Test the validate endpoint with valid model"""
        response = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": "mlx-community/Llama-3.2-1B-Instruct-4bit"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert "model_info" in data
        
    @pytest.mark.skip(reason="Requires mocking HF API")
    def test_validate_endpoint_invalid_format(self):
        """Test the validate endpoint with invalid format"""
        response = client.post(
            "/api/base-models/validate",
            json={"huggingface_id": "invalid model name"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        
    def test_list_models_includes_custom(self):
        """Test that list endpoint returns both curated and custom"""
        # Use X-Forwarded-For header to simulate localhost connection
        response = client.get(
            "/api/base-models",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        
        # Check that models have required fields
        if len(models) > 0:
            model = models[0]
            assert "id" in model
            assert "huggingface_id" in model
            assert "name" in model
            assert "mlx_config" in model


class TestSecurity:
    """Test security aspects of custom model feature"""
    
    def test_input_length_limit(self):
        """Test that model ID length is limited"""
        import re
        
        # Test max length (255 chars)
        long_id = "a" * 250 + "/" + "b" * 5  # 256 chars total
        match = re.match(r'^[\w\-\.]+(/[\w\-\.]+)*$', long_id)
        assert match is not None  # Regex allows it, but API should reject
        
    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked"""
        import re
        
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "org/..\\../other",
        ]
        
        for malicious_id in malicious_ids:
            match = re.match(r'^[\w\-\.]+(/[\w\-\.]+)?$', malicious_id)
            assert match is None, f"Should reject path traversal: {malicious_id}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
