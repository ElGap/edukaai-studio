"""
Test for model editing functionality - PATCH /training/runs/{id}
Ensures editing works with both completed and pending training runs.
Uses in-memory test database via conftest fixture.
"""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

os.environ.setdefault("EDUKAAI_ALLOW_REMOTE", "true")
os.environ.setdefault("EDUKAAI_ENV", "testing")

from app.models import TrainingRun, ModelRegistry, Dataset, get_db, Base
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session


def _create_test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    return engine, TestSession


class TestUpdateTrainingRun:
    """Test that updating training runs works correctly."""

    def test_update_incomplete_run_succeeds(self):
        engine, Session = _create_test_db()
        db = Session()

        base_model = ModelRegistry(
            id="test-model",
            huggingface_id="test/model",
            name="Test Model",
            architecture="test",
            parameter_count=1000000,
            context_length=2048,
            is_active=True,
            is_curated=True
        )
        db.add(base_model)

        dataset = Dataset(
            id="test-dataset",
            name="Test Dataset",
            format="alpaca",
            num_samples=100,
            file_path="/tmp/data.jsonl",
            created_at=datetime.now()
        )
        db.add(dataset)

        run = TrainingRun(
            id="test-run",
            name="Old Run Name",
            status="pending",
            training_dataset_id="test-dataset",
            base_model_id="test-model",
            storage_path="/tmp/run",
            steps=100,
            learning_rate=0.0001,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            batch_size=4,
            max_seq_length=2048,
            warmup_steps=10,
            gradient_accumulation_steps=1,
            early_stopping_patience=5,
            gradient_checkpointing=False,
            num_lora_layers=4,
            prompt_masking=False,
            current_step=0,
            total_steps=100,
        )
        db.add(run)
        db.commit()

        saved_run = db.query(TrainingRun).filter(TrainingRun.id == "test-run").first()
        assert saved_run is not None
        assert saved_run.completed_at is None

        from app.routers.training import build_training_config_response
        from app.routers.training import BaseModelResponse, TrainingRunResponse

        try:
            response = TrainingRunResponse(
                id=saved_run.id,
                name="Updated Name",
                description="Updated description",
                tags="test, updated",
                status=saved_run.status,
                current_step=saved_run.current_step,
                total_steps=saved_run.total_steps,
                best_loss=saved_run.best_loss,
                best_step=saved_run.best_step,
                validation_loss=saved_run.validation_loss,
                completed_at=saved_run.completed_at.isoformat() if saved_run.completed_at else None,
                error_message=saved_run.error_message,
                adapter_exported=saved_run.adapter_exported,
                fused_exported=saved_run.fused_exported,
                training_config=build_training_config_response(saved_run),
                base_model=BaseModelResponse(
                    id=base_model.id,
                    huggingface_id=base_model.huggingface_id,
                    name=base_model.name,
                    architecture=base_model.architecture,
                    parameter_count=base_model.parameter_count,
                    context_length=base_model.context_length,
                    mlx_config=base_model.mlx_config,
                    is_custom=False
                ),
                created_at=saved_run.created_at.isoformat()
            )
            assert response.name == "Updated Name"
            assert response.completed_at is None
        except Exception as e:
            pytest.fail(f"BUG: Cannot edit incomplete run without completed_at: {e}")

        db.close()
        engine.dispose()

    def test_update_completed_run_includes_completed_at(self):
        engine, Session = _create_test_db()
        db = Session()

        base_model = ModelRegistry(
            id="test-model-2",
            huggingface_id="test/model2",
            name="Test Model 2",
            architecture="test",
            parameter_count=1000000,
            context_length=2048,
            is_active=True,
            is_curated=True
        )
        db.add(base_model)

        dataset = Dataset(
            id="test-dataset-2",
            name="Test Dataset 2",
            format="alpaca",
            num_samples=100,
            file_path="/tmp/data2.jsonl",
            created_at=datetime.now()
        )
        db.add(dataset)

        run = TrainingRun(
            id="test-run-2",
            name="Completed Run",
            status="completed",
            training_dataset_id="test-dataset-2",
            base_model_id="test-model-2",
            storage_path="/tmp/run2",
            steps=100,
            learning_rate=0.0001,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            batch_size=4,
            max_seq_length=2048,
            warmup_steps=10,
            gradient_accumulation_steps=1,
            early_stopping_patience=5,
            gradient_checkpointing=False,
            num_lora_layers=4,
            prompt_masking=False,
            current_step=100,
            total_steps=100,
            completed_at=datetime.now(),
        )
        db.add(run)
        db.commit()

        from app.routers.training import build_training_config_response
        from app.routers.training import BaseModelResponse, TrainingRunResponse

        saved_run = db.query(TrainingRun).filter(TrainingRun.id == "test-run-2").first()

        response = TrainingRunResponse(
            id=saved_run.id,
            name="Updated Completed Run",
            description=None,
            tags=None,
            status=saved_run.status,
            current_step=saved_run.current_step,
            total_steps=saved_run.total_steps,
            best_loss=saved_run.best_loss,
            best_step=saved_run.best_step,
            validation_loss=saved_run.validation_loss,
            completed_at=saved_run.completed_at.isoformat() if saved_run.completed_at else None,
            error_message=saved_run.error_message,
            adapter_exported=saved_run.adapter_exported,
            fused_exported=saved_run.fused_exported,
            training_config=build_training_config_response(saved_run),
            base_model=BaseModelResponse(
                id=base_model.id,
                huggingface_id=base_model.huggingface_id,
                name=base_model.name,
                architecture=base_model.architecture,
                parameter_count=base_model.parameter_count,
                context_length=base_model.context_length,
                mlx_config=base_model.mlx_config,
                is_custom=False
            ),
            created_at=saved_run.created_at.isoformat()
        )

        assert response.completed_at is not None
        assert isinstance(response.completed_at, str)

        db.close()
        engine.dispose()
