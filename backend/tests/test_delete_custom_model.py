"""
Test custom model deletion functionality.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models import Base, BaseModel, get_db
from datetime import datetime
import uuid

# Use real database
from app.config import get_settings
settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestDeleteCustomModel:
    """Test DELETE /base-models/{model_id} endpoint"""
    
    def test_delete_nonexistent_model_returns_404(self):
        """Test deleting a model that doesn't exist"""
        response = client.delete(
            "/api/base-models/nonexistent-id",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        
        assert response.status_code == 404
        print("✓ Returns 404 for nonexistent model")
    
    def test_delete_curated_model_returns_400(self):
        """Test that curated models cannot be deleted"""
        # Create a curated model
        db = TestingSessionLocal()
        model_id = f"test-curated-{uuid.uuid4().hex[:8]}"
        model = BaseModel(
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
        db.close()
        
        try:
            response = client.delete(
                f"/api/base-models/{model_id}",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            
            assert response.status_code == 400
            assert "curated" in response.json()["detail"].lower()
            print("✓ Cannot delete curated models")
        finally:
            # Cleanup
            db = TestingSessionLocal()
            db.query(BaseModel).filter(BaseModel.id == model_id).delete()
            db.commit()
            db.close()
    
    def test_delete_unused_custom_model_hard_deletes(self):
        """Test that unused custom models are hard deleted"""
        # Create a custom model
        db = TestingSessionLocal()
        model_id = f"test-custom-{uuid.uuid4().hex[:8]}"
        hf_id = f"test/custom-model-{uuid.uuid4().hex[:8]}"
        model = BaseModel(
            id=model_id,
            huggingface_id=hf_id,
            name="Test Custom Model",
            architecture="llama",
            parameter_count=1000000000,
            context_length=2048,
            is_curated=False,
            is_active=True
        )
        db.add(model)
        db.commit()
        db.close()
        
        try:
            # Delete the model
            response = client.delete(
                f"/api/base-models/{model_id}",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["deleted"] is True  # Hard deleted
            print("✓ Unused custom model is hard deleted")
            
            # Verify it's actually gone
            db = TestingSessionLocal()
            model_check = db.query(BaseModel).filter(BaseModel.id == model_id).first()
            db.close()
            assert model_check is None
            print("✓ Model is actually removed from database")
        finally:
            # Cleanup (in case test failed)
            db = TestingSessionLocal()
            db.query(BaseModel).filter(BaseModel.id == model_id).delete()
            db.commit()
            db.close()
    
    def test_delete_custom_model_not_in_list_after_deletion(self):
        """Test that deleted custom model doesn't appear in the list"""
        # Create a custom model
        db = TestingSessionLocal()
        model_id = f"test-custom-list-{uuid.uuid4().hex[:8]}"
        hf_id = f"test/custom-model-{uuid.uuid4().hex[:8]}"
        model = BaseModel(
            id=model_id,
            huggingface_id=hf_id,
            name="Test Custom Model For List",
            architecture="llama",
            parameter_count=1000000000,
            context_length=2048,
            is_curated=False,
            is_active=True
        )
        db.add(model)
        db.commit()
        db.close()
        
        try:
            # Verify it appears in the list
            list_response = client.get(
                "/api/base-models",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert list_response.status_code == 200
            models = list_response.json()
            model_ids = [m["id"] for m in models]
            assert model_id in model_ids
            print("✓ Custom model appears in list before deletion")
            
            # Delete it
            delete_response = client.delete(
                f"/api/base-models/{model_id}",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert delete_response.status_code == 200
            
            # Verify it's NOT in the list anymore
            list_response2 = client.get(
                "/api/base-models",
                headers={"X-Forwarded-For": "127.0.0.1"}
            )
            assert list_response2.status_code == 200
            models2 = list_response2.json()
            model_ids2 = [m["id"] for m in models2]
            assert model_id not in model_ids2
            print("✓ Deleted custom model does not appear in list")
        finally:
            # Cleanup
            db = TestingSessionLocal()
            db.query(BaseModel).filter(BaseModel.id == model_id).delete()
            db.commit()
            db.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING CUSTOM MODEL DELETE TESTS")
    print("="*70 + "\n")
    
    test_class = TestDeleteCustomModel()
    
    print("\n1. Testing delete nonexistent model...")
    test_class.test_delete_nonexistent_model_returns_404()
    
    print("\n2. Testing delete curated model...")
    test_class.test_delete_curated_model_returns_400()
    
    print("\n3. Testing delete unused custom model...")
    test_class.test_delete_unused_custom_model_hard_deletes()
    
    print("\n4. Testing deleted model not in list...")
    test_class.test_delete_custom_model_not_in_list_after_deletion()
    
    print("\n" + "="*70)
    print("ALL CUSTOM MODEL DELETE TESTS PASSED!")
    print("="*70 + "\n")
