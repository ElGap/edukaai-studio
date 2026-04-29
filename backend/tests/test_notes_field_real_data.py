"""
Comprehensive integration test for the notes field functionality.
Uses in-memory test database via conftest fixture.
"""
import os
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import uuid

os.environ.setdefault("EDUKAAI_ALLOW_REMOTE", "true")
os.environ.setdefault("EDUKAAI_ENV", "testing")

from app.main import app
from app.models import ModelRegistry, Dataset, TrainingRun

client = TestClient(app)


class TestNotesFieldWithRealData:
    """Test notes field with test database and API endpoints"""

    @pytest.fixture(scope="function")
    def test_data(self, test_db_session):
        db = test_db_session
        unique_suffix = uuid.uuid4().hex[:8]

        unique_hf_id = f"mlx-community/test-model-{unique_suffix}"
        unique_model_id = f"test-base-model-{unique_suffix}"
        base_model = ModelRegistry(
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

        unique_run_id = f"test-run-{unique_suffix}"
        run = TrainingRun(
            id=unique_run_id,
            name="Test Run Without Notes",
            description="A test run for testing the notes field",
            tags="test,notes,integration",
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

        db.query(TrainingRun).filter(TrainingRun.id == run.id).delete()
        db.query(Dataset).filter(Dataset.id == dataset.id).delete()
        db.query(ModelRegistry).filter(ModelRegistry.id == base_model.id).delete()
        db.commit()

    def test_database_has_notes_column(self, test_db_session):
        from sqlalchemy import inspect
        engine = test_db_session.get_bind()
        inspector = inspect(engine)
        columns = inspector.get_columns('training_runs')
        column_names = [col['name'] for col in columns]
        assert 'notes' in column_names, f"notes column not found in {column_names}"

    def test_get_run_without_notes_returns_null(self, test_data, test_db_session):
        run_id = test_data["run_id"]
        response = client.get(
            f"/api/training/runs/{run_id}",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == run_id
        assert data.get("notes") is None or data.get("notes") == ""

    def test_patch_adds_notes_to_run(self, test_data, test_db_session):
        run_id = test_data["run_id"]
        notes_content = "# Fine-tuning Notes\n\n## Observations\n- Learning rate of 0.0001 worked well"
        patch_response = client.patch(
            f"/api/training/runs/{run_id}",
            json={"name": "Test Run With Notes", "notes": notes_content},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        assert patch_response.status_code == 200, f"PATCH failed: {patch_response.text}"
        patch_data = patch_response.json()
        assert patch_data["notes"] == notes_content

    def test_get_run_with_notes_returns_notes(self, test_data, test_db_session):
        run_id = test_data["run_id"]
        notes_content = "These are my fine-tuning observations and findings."
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": notes_content},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        response = client.get(
            f"/api/training/runs/{run_id}",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 200
        assert response.json()["notes"] == notes_content

    def test_list_runs_includes_notes(self, test_data, test_db_session):
        run_id = test_data["run_id"]
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": "Notes for list test"},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        response = client.get(
            "/api/training/runs",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 200
        data = response.json()
        test_run = next((r for r in data if r["id"] == run_id), None)
        assert test_run is not None
        assert test_run["notes"] == "Notes for list test"

    def test_update_notes_replaces_existing(self, test_data, test_db_session):
        run_id = test_data["run_id"]
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": "First version of notes"},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        updated_notes = "Updated version with more insights"
        response = client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": updated_notes},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        assert response.status_code == 200
        assert response.json()["notes"] == updated_notes

    def test_patch_without_notes_preserves_existing(self, test_data, test_db_session):
        run_id = test_data["run_id"]
        original_notes = "Important findings that should be preserved"
        client.patch(
            f"/api/training/runs/{run_id}",
            json={"notes": original_notes},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        response = client.patch(
            f"/api/training/runs/{run_id}",
            json={"name": "Updated Name Only"},
            headers={"X-Forwarded-For": "127.0.0.1", "Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name Only"
        assert data["notes"] == original_notes
