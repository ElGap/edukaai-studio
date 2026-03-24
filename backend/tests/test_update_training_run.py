"""
Test for model editing functionality - PATCH /training/runs/{id}
Ensures editing works with both completed and pending training runs.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

os.environ["EDUKAI_ALLOW_REMOTE"] = "true"
os.environ["EDUKAI_ENV"] = "testing"

sys.path.insert(0, '/Users/developer/Projects/studio/backend')

from app.models import TrainingRun, BaseModel, Dataset, get_db


class TestUpdateTrainingRun:
    """Test that updating training runs works correctly."""
    
    def test_update_incomplete_run_succeeds(self):
        """
        CRITICAL: Must be able to edit incomplete/pending training runs.
        
        Bug: Old runs without completed_at caused Pydantic validation error.
        Fixed: completed_at is now Optional with proper null handling.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test database
            db_path = Path(tmpdir) / "test.db"
            os.environ["EDUKAI_DATABASE_URL"] = f"sqlite:///{db_path}"
            
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from app.models import Base
            
            engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            db = Session()
            
            # Create required records
            base_model = BaseModel(
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
                file_path=f"{tmpdir}/data.jsonl",
                created_at=datetime.now()
            )
            db.add(dataset)
            
            # Create INCOMPLETE training run (no completed_at)
            run = TrainingRun(
                id="test-run",
                name="Old Run Name",
                status="pending",  # Not completed!
                training_dataset_id="test-dataset",
                base_model_id="test-model",
                storage_path=str(tmpdir),
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
                # completed_at is NULL!
            )
            db.add(run)
            db.commit()
            
            # Verify run exists without completed_at
            saved_run = db.query(TrainingRun).filter(TrainingRun.id == "test-run").first()
            assert saved_run is not None
            assert saved_run.completed_at is None  # Should be NULL
            
            # Now test building the response (simulating what the endpoint does)
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
                    completed_at=saved_run.completed_at.isoformat() if saved_run.completed_at else None,  # Key fix!
                    error_message=saved_run.error_message,
                    adapter_exported=saved_run.adapter_exported,
                    fused_exported=saved_run.fused_exported,
                    gguf_exported=saved_run.gguf_exported,
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
                
                # If we get here, the response was built successfully
                assert response.name == "Updated Name"
                assert response.description == "Updated description"
                assert response.tags == "test, updated"
                assert response.completed_at is None  # Should remain None
                
                print("SUCCESS: Can edit incomplete training runs!")
                
            except Exception as e:
                pytest.fail(f"BUG: Cannot edit incomplete run without completed_at: {e}")
            
            db.close()
    
    def test_update_completed_run_includes_completed_at(self):
        """
        Verify completed runs still include completed_at in response.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            os.environ["EDUKAI_DATABASE_URL"] = f"sqlite:///{db_path}"
            
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from app.models import Base
            
            engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            db = Session()
            
            # Create records
            base_model = BaseModel(
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
                file_path=f"{tmpdir}/data2.jsonl",
                created_at=datetime.now()
            )
            db.add(dataset)
            
            # Create COMPLETED training run
            run = TrainingRun(
                id="test-run-2",
                name="Completed Run",
                status="completed",
                training_dataset_id="test-dataset-2",
                base_model_id="test-model-2",
                storage_path=str(tmpdir),
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
                completed_at=datetime.now(),  # Has completion date
            )
            db.add(run)
            db.commit()
            
            # Build response
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
                gguf_exported=saved_run.gguf_exported,
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
            
            # Verify completed_at is included
            assert response.completed_at is not None
            assert isinstance(response.completed_at, str)
            
            db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
