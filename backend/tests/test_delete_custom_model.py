"""
Test custom model deletion functionality.
Uses in-memory test database via conftest fixture.
"""
import os
import pytest
from fastapi.testclient import TestClient
import uuid

os.environ.setdefault("EDUKAAI_ALLOW_REMOTE", "true")
os.environ.setdefault("EDUKAAI_ENV", "testing")

from app.main import app
from app.models import ModelRegistry

client = TestClient(app)


class TestDeleteCustomModel:
    """Test DELETE /base-models/{model_id} endpoint"""

    def test_delete_nonexistent_model_returns_404(self, test_db_session):
        response = client.delete(
            "/api/base-models/nonexistent-id",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 404

    def test_delete_curated_model_returns_400(self, test_db_session):
        db = test_db_session
        model_id = f"test-curated-{uuid.uuid4().hex[:8]}"
        model = ModelRegistry(
            id=model_id,
            huggingface_id=f"test/curated-model-{uuid.uuid4().hex[:8]}",
            name="Test Curated Model",
            architecture="llama",
            parameter_count=1000000000,
            context_length=2048,
            is_curated=True,
            is_active=True
        )
        db.add(model)
        db.commit()

        try:
            response = client.delete(
                f"/api/base-models/{model_id}",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert response.status_code == 400
            assert "curated" in response.json()["detail"].lower()
        finally:
            db.query(ModelRegistry).filter(ModelRegistry.id == model_id).delete()
            db.commit()

    def test_delete_unused_custom_model_hard_deletes(self, test_db_session):
        db = test_db_session
        model_id = f"test-custom-{uuid.uuid4().hex[:8]}"
        model = ModelRegistry(
            id=model_id,
            huggingface_id=f"test/custom-model-{uuid.uuid4().hex[:8]}",
            name="Test Custom Model",
            architecture="llama",
            parameter_count=1000000000,
            context_length=2048,
            is_curated=False,
            is_active=True
        )
        db.add(model)
        db.commit()

        try:
            response = client.delete(
                f"/api/base-models/{model_id}",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["deleted"] is True

            model_check = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
            assert model_check is None
        finally:
            db.query(ModelRegistry).filter(ModelRegistry.id == model_id).delete()
            db.commit()

    def test_delete_custom_model_not_in_list_after_deletion(self, test_db_session):
        db = test_db_session
        model_id = f"test-custom-list-{uuid.uuid4().hex[:8]}"
        model = ModelRegistry(
            id=model_id,
            huggingface_id=f"test/custom-model-{uuid.uuid4().hex[:8]}",
            name="Test Custom Model For List",
            architecture="llama",
            parameter_count=1000000000,
            context_length=2048,
            is_curated=False,
            is_active=True
        )
        db.add(model)
        db.commit()

        try:
            list_response = client.get(
                "/api/base-models",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert list_response.status_code == 200
            models = list_response.json()
            model_ids = [m["id"] for m in models]
            assert model_id in model_ids

            delete_response = client.delete(
                f"/api/base-models/{model_id}",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert delete_response.status_code == 200

            list_response2 = client.get(
                "/api/base-models",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert list_response2.status_code == 200
            models2 = list_response2.json()
            model_ids2 = [m["id"] for m in models2]
            assert model_id not in model_ids2
        finally:
            db.query(ModelRegistry).filter(ModelRegistry.id == model_id).delete()
            db.commit()
