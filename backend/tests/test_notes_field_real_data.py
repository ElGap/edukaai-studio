"""
Comprehensive integration test for the notes field functionality.
This test uses the REAL database and tests the actual API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Import app and models
from app.main import app
from app.models import Base, TrainingRun, BaseModel, Dataset, get_db
from app.config import get_settings

# Get real database URL from settings
settings = get_settings()
print(f"Using database: {settings.database_url}")

# Create engine with real database
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override get_db to use the real database"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Override the dependency
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

class TestNotesFieldWithRealData:
    """Test notes field with real database and API endpoints"""
    
    @pytest.fixture(scope="function")
    def test_data(self):
        """Create test data in real database"""
        import uuid
        unique_suffix = uuid.uuid4().hex[:8]
        
        db = TestingSessionLocal()
        
        # Create base model with unique IDs
        unique_hf_id = f"mlx-community/test-model-{unique_suffix}"
        unique_model_id = f"test-base-model-{unique_suffix}"
        base_model = BaseModel(
            id=unique_model_id,
            huggingface_id=unique_hf_id,
            name="Test Llama Model",
            architecture="llama",
            parameter_count=1000000000,
            context_length=2048,
            mlx_config={"is_curated": True},
            is_curated=True
        )
        db.add(base_model)
        
        # Create dataset with unique ID
        unique_dataset_id = f"test-dataset-{unique_suffix}"
        dataset = Dataset(
            id=unique_dataset_id,
            name="Test Dataset for Notes",
            description="Test dataset for notes testing",
            format="alpaca",
            file_path="/tmp/test_notes.jsonl",
            num_samples=100,
            size_bytes=1024000
        )
        db.add(dataset)
        db.commit()
        
        # Create training run WITHOUT notes initially
        unique_run_id = f"test-run-{unique_suffix}"
        run = TrainingRun(
            id=unique_run_id,
            name="Test Run Without Notes",
            description="A test run for testing the notes field",
            tags="test,notes,integration",
            # notes is NOT set initially - this tests the migration/default
            status="completed",
            current_step=100,
            total_steps=100,
            best_loss=0.5,
            best_step=95,
            training_dataset_id=unique_dataset_id,
            base_model_id=unique_model_id,
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
            storage_path="/tmp/test-notes-run",
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        db.add(run)
        db.commit()
        
        yield {
            "run_id": run.id,
            "model_id": base_model.id,
            "dataset_id": dataset.id
        }
        
        # Cleanup
        db.query(TrainingRun).filter(TrainingRun.id == run.id).delete()
        db.query(Dataset).filter(Dataset.id == dataset.id).delete()
        db.query(BaseModel).filter(BaseModel.id == base_model.id).delete()
        db.commit()
        db.close()
    
    def test_database_has_notes_column(self):
        """Verify the database actually has the notes column"""
        inspector = inspect(engine)
        columns = inspector.get_columns('training_runs')
        column_names = [col['name'] for col in columns]
        
        assert 'notes' in column_names, f"notes column not found in {column_names}"
        print(f"✓ Database has notes column. All columns: {column_names}")
    
    def test_get_run_without_notes_returns_null(self, test_data):
        """Test GET endpoint returns null for notes when not set"""
        run_id = test_data["run_id"]
        
        response = client.get(
            f"/api/training/runs/{run_id}",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify run exists
        assert data["id"] == run_id
        assert data["name"] == "Test Run Without Notes"
        
        # Verify notes is None/null when not set
        assert data.get("notes") is None or data.get("notes") == ""
        print("✓ GET /training/runs/{id} returns null/empty notes when not set")
    
    def test_patch_adds_notes_to_run(self, test_data):
        """Test PATCH endpoint successfully adds notes to a run"""
        run_id = test_data["run_id"]
        
        notes_content = """# Fine-tuning Notes

## Observations
- Learning rate of 0.0001 worked well for this dataset
- LoRA rank 8 provided good balance of performance and speed
- Model converged at step 95 with loss 0.5

## What Worked
1. Using warmup steps prevented early overfitting
2. Batch size 4 fit well within memory constraints
3. Gradient checkpointing wasn't needed for this model size

## Insights
- Dataset quality was crucial - filtered out low-quality samples
- Validation split of 10% provided good monitoring
- Early stopping patience of 0 worked since we had fixed steps

## Future Improvements
- Try LoRA rank 16 for potentially better performance
- Experiment with different learning rate schedules
- Consider larger batch size if memory allows"""
        
        # PATCH the run with notes
        patch_response = client.patch(
            f"/api/training/runs/{run_id}",
            json={
                "name": "Test Run With Notes",
                "notes": notes_content
            },
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        assert patch_response.status_code == 200, f"PATCH failed: {patch_response.text}"
        patch_data = patch_response.json()
        
        # Verify response includes notes
        assert patch_data["name"] == "Test Run With Notes"
        assert patch_data["notes"] == notes_content
        print(f"✓ PATCH successfully added notes ({len(notes_content)} chars)")
        
        # Verify notes were actually saved to database
        db = TestingSessionLocal()
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        db.close()
        
        assert run.notes == notes_content
        print("✓ Notes were actually saved to the database")
    
    def test_get_run_with_notes_returns_notes(self, test_data):
        """Test GET endpoint returns notes after they were added"""
        run_id = test_data["run_id"]
        
        # First, add notes
        notes_content = "These are my fine-tuning observations and findings."
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": notes_content},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        # Now GET the run
        response = client.get(
            f"/api/training/runs/{run_id}",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify notes are returned
        assert data["notes"] == notes_content
        print("✓ GET /training/runs/{id} returns notes correctly")
    
    def test_list_runs_includes_notes(self, test_data):
        """Test LIST endpoint includes notes field"""
        run_id = test_data["run_id"]
        
        # Add notes first
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": "Notes for list test"},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        # List all runs
        response = client.get(
            "/api/training/runs",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Find our test run
        test_run = next((r for r in data if r["id"] == run_id), None)
        assert test_run is not None, "Test run not found in list"
        
        # Verify notes field exists and has value
        assert "notes" in test_run
        assert test_run["notes"] == "Notes for list test"
        print("✓ GET /training/runs includes notes field")
    
    def test_update_notes_replaces_existing(self, test_data):
        """Test that updating notes replaces existing notes"""
        run_id = test_data["run_id"]
        
        # First notes
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": "First version of notes"},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        # Updated notes
        updated_notes = "Updated version with more insights"
        response = client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": updated_notes},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["notes"] == updated_notes
        
        # Verify in database
        db = TestingSessionLocal()
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        db.close()
        
        assert run.notes == updated_notes
        print("✓ Updating notes replaces existing notes correctly")
    
    def test_patch_without_notes_preserves_existing(self, test_data):
        """Test that PATCH without notes field preserves existing notes"""
        run_id = test_data["run_id"]
        
        # Add notes
        original_notes = "Important findings that should be preserved"
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": original_notes},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        # Update only name, not notes
        response = client.patch(
            f"/api/training/runs/{run_id}",
            json={"name": "Updated Name Only"},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Name should be updated
        assert data["name"] == "Updated Name Only"
        # Notes should be preserved
        assert data["notes"] == original_notes
        print("✓ PATCH without notes preserves existing notes")

if __name__ == "__main__":
    # Run with pytest instead for proper test isolation
    import subprocess
    import sys
    print("Run with: python -m pytest tests/test_notes_field_real_data.py -v")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_notes_field_real_data.py", "-v"])
    sys.exit(result.returncode)
    
    # Run tests manually
    test_class = TestNotesFieldWithRealData()
    
    # Test 1: Database has notes column
    print("\n1. Testing database schema...")
    test_class.test_database_has_notes_column()
    
    # Create test data fixture manually
    db = TestingSessionLocal()
    
    # Create base model with unique huggingface_id
    import uuid
    unique_hf_id = f"mlx-community/test-model-{uuid.uuid4().hex[:8]}"
    base_model = BaseModel(
        id="test-base-model-notes-001",
        huggingface_id=unique_hf_id,
        name="Test Llama Model",
        architecture="llama",
        parameter_count=1000000000,
        context_length=2048,
        mlx_config={"is_curated": True},
        is_curated=True
    )
    db.add(base_model)
    
    # Create dataset
    dataset = Dataset(
        id="test-dataset-notes-001",
        name="Test Dataset for Notes",
        description="Test dataset for notes testing",
        format="alpaca",
        file_path="/tmp/test_notes.jsonl",
        num_samples=100,
        size_bytes=1024000
    )
    db.add(dataset)
    db.commit()
    
    # Create training run WITHOUT notes initially
    run = TrainingRun(
        id="test-run-notes-001",
        name="Test Run Without Notes",
        description="A test run for testing the notes field",
        tags="test,notes,integration",
        status="completed",
        current_step=100,
        total_steps=100,
        best_loss=0.5,
        best_step=95,
        training_dataset_id="test-dataset-notes-001",
        base_model_id="test-base-model-notes-001",
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
        storage_path="/tmp/test-notes-run",
        created_at=datetime.now(),
        completed_at=datetime.now()
    )
    db.add(run)
    db.commit()
    
    # Get the IDs before closing
    run_id = run.id
    model_id = base_model.id
    dataset_id = dataset.id
    
    db.close()
    
    test_data = {
        "run_id": run_id,
        "model_id": model_id,
        "dataset_id": dataset_id
    }
    
    # Test 2: GET without notes
    print("\n2. Testing GET endpoint without notes...")
    test_class.test_get_run_without_notes_returns_null(test_data)
    
    # Test 3: PATCH adds notes
    print("\n3. Testing PATCH endpoint to add notes...")
    test_class.test_patch_adds_notes_to_run(test_data)
    
    # Test 4: GET with notes
    print("\n4. Testing GET endpoint with notes...")
    test_class.test_get_run_with_notes_returns_notes(test_data)
    
    # Test 5: LIST includes notes
    print("\n5. Testing LIST endpoint includes notes...")
    test_class.test_list_runs_includes_notes(test_data)
    
    # Test 6: Update notes replaces existing
    print("\n6. Testing that updating notes replaces existing...")
    test_class.test_update_notes_replaces_existing(test_data)
    
    # Test 7: PATCH without notes preserves existing
    print("\n7. Testing that PATCH without notes preserves existing...")
    test_class.test_patch_without_notes_preserves_existing(test_data)
    
    # Cleanup
    print("\n8. Cleaning up test data...")
    db = TestingSessionLocal()
    db.query(TrainingRun).filter(TrainingRun.id == run.id).delete()
    db.query(Dataset).filter(Dataset.id == dataset.id).delete()
    db.query(BaseModel).filter(BaseModel.id == base_model.id).delete()
    db.commit()
    db.close()
    print("✓ Cleanup complete")
    
    print("\n" + "="*70)
    print("ALL NOTES FIELD INTEGRATION TESTS PASSED!")
    print("="*70 + "\n")
