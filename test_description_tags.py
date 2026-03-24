"""Test that description and tags are returned from API endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.models import Base, TrainingRun, BaseModel, Dataset, get_db
from datetime import datetime

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_desc_tags.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create test database
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_get_run_returns_description_and_tags():
    """Verify GET /training/runs/{id} returns description and tags."""
    # Create test data
    db = TestingSessionLocal()
    
    # Create base model
    base_model = BaseModel(
        id="test-model-id",
        huggingface_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        name="Test Model",
        architecture="llama",
        parameter_count=1000000000,
        context_length=2048,
        mlx_config={"is_curated": True}
    )
    db.add(base_model)
    
    # Create dataset
    dataset = Dataset(
        id="test-dataset-id",
        name="Test Dataset",
        format="alpaca",
        file_path="/tmp/test.jsonl",
        num_samples=100
    )
    db.add(dataset)
    db.commit()
    
    # Create training run with description and tags
    run = TrainingRun(
        id="test-run-id",
        name="Test Run",
        description="This is a test description",
        tags="test,llm,mlx",
        status="completed",
        current_step=100,
        total_steps=100,
        training_dataset_id="test-dataset-id",
        base_model_id="test-model-id",
        steps=100,
        learning_rate=0.0001,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=4,
        max_seq_length=2048,
        warmup_steps=10,
        gradient_accumulation_steps=1,
        early_stopping_patience=0,
        gradient_checkpointing=False,
        num_lora_layers=8,
        prompt_masking=True,
        validation_split_percent=10,
        storage_path="/tmp/test-run",
        created_at=datetime.now(),
        completed_at=datetime.now()
    )
    db.add(run)
    db.commit()
    db.close()
    
    # Test GET endpoint
    response = client.get("/api/training/runs/test-run-id")
    assert response.status_code == 200
    
    data = response.json()
    assert data["description"] == "This is a test description"
    assert data["tags"] == "test,llm,mlx"
    
    print("✓ GET /training/runs/{id} returns description and tags correctly")

def test_list_runs_returns_description_and_tags():
    """Verify GET /training/runs returns description and tags."""
    response = client.get("/api/training/runs")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) > 0
    
    run = data[0]
    assert "description" in run
    assert "tags" in run
    assert run["description"] == "This is a test description"
    assert run["tags"] == "test,llm,mlx"
    
    print("✓ GET /training/runs returns description and tags correctly")

if __name__ == "__main__":
    test_get_run_returns_description_and_tags()
    test_list_runs_returns_description_and_tags()
    print("\n✅ All description/tags tests passed!")
