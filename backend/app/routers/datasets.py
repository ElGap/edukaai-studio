"""
Datasets router - Step 1: Dataset Upload
"""

import json
import os
import random
import shutil
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.exceptions import ValidationError, NotFoundError
from ..core import sanitize_filename, validate_jsonl_format, detect_format, sanitize_dataset_content, assert_safe_path
from ..core.logging import get_logger
from ..models import get_db, Dataset, TrainingRun, generate_uuid
from ..config import get_settings

router = APIRouter()
logger = get_logger(__name__)


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    format: str
    num_samples: int
    created_at: str
    has_validation_set: bool = False
    validation_set_id: Optional[str] = None
    validation_samples: int = 0
    
    class Config:
        from_attributes = True


class DatasetDetailResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    format: str
    num_samples: int
    total_raw_samples: int
    validation_report: dict
    preview_samples: List[dict]
    dataset_schema: dict
    is_validation_set: bool
    created_at: str
    
    class Config:
        from_attributes = True


@router.post("/datasets", response_model=DatasetDetailResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    is_validation_set: bool = Form(False),
    confirm_data_rights: bool = Form(True),  # Default: user confirms they have rights
    skip_pii_detection: bool = Form(True),   # Default: skip PII if user confirms rights
    db: Session = Depends(get_db)
):
    settings = get_settings()
    """Upload and validate a dataset JSONL file."""
    
    # Validate file type — block obviously non-text / non-JSON types
    blocked_prefixes = (
        'image/', 'audio/', 'video/', 'application/zip',
        'application/x-rar', 'application/x-7z', 'application/x-tar',
        'application/x-exe', 'application/x-msdownload', 'application/x-sh',
        'application/pdf', 'application/msword', 'application/vnd.openxmlformats',
    )
    if file.content_type and any(file.content_type.startswith(bp) for bp in blocked_prefixes):
        raise ValidationError(
            f"Blocked content type: '{file.content_type}'. Only plain-text / JSON files are accepted."
        )

    # Warn but still try to parse for other unexpected types (e.g. octet-stream from some browsers)
    allowed_content_types = [
        'application/json',
        'application/jsonl',
        'text/plain',
        'text/json',
        'application/octet-stream',
        None,
    ]
    if file.content_type and not any(file.content_type.startswith(ct) or ct is None for ct in allowed_content_types):
        logger.warning(f"Unexpected content type: {file.content_type}")

    # Generate ID and filename
    dataset_id = generate_uuid()

    if name is None:
        base_name = Path(file.filename).stem
        name = sanitize_filename(base_name)

    # Stream upload to a temporary file in chunks (do not read entire file into RAM)
    import tempfile
    max_size_bytes = min(settings.max_storage_gb * 1024 * 1024 * 1024, 500 * 1024 * 1024)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jsonl')
    total = 0
    chunk_size = 64 * 1024
    try:
        while chunk := await file.read(chunk_size):
            total += len(chunk)
            if total > max_size_bytes:
                raise ValidationError("File too large for processing. Maximum 500MB allowed.")
            os.write(tmp_fd, chunk)
        os.close(tmp_fd)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            content_str = f.read()
    except UnicodeDecodeError:
        raise ValidationError("File must be valid UTF-8 encoded text")
    finally:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    
    # Validate JSONL format
    is_valid, valid_samples, errors = validate_jsonl_format(content_str)
    
    # Log details for debugging
    logger.info(f"File validation: {len(valid_samples)} valid, {len(errors)} errors")
    if errors:
        logger.warning(f"Validation errors (first 3): {errors[:3]}")
    
    if not valid_samples:
        error_details = errors[:5] if errors else []
        raise ValidationError(f"No valid samples found in file. Errors: {error_details}")
    
    # Detect format
    format_type = detect_format(valid_samples)
    
    # Generate previews (first 10 + random 10 = 20 total for better visibility)
    preview_samples = []
    if valid_samples:
        # First 10
        preview_samples.extend([s["data"] for s in valid_samples[:10]])
        
        # Random 10 from remaining (ensure diversity)
        remaining = valid_samples[10:]
        if remaining:
            # If we have more than 10 remaining, pick 10 random ones
            # Otherwise take all remaining
            sample_count = min(10, len(remaining))
            if len(remaining) > sample_count:
                random_samples = random.sample(remaining, sample_count)
            else:
                random_samples = remaining
            preview_samples.extend([s["data"] for s in random_samples])
    
    # Create validation report
    validation_report = {
        "total_samples": len(valid_samples) + len(errors),
        "valid_samples": len(valid_samples),
        "invalid_samples": len(errors),
        "errors": errors[:10],  # Limit to first 10 errors
        "format_detected": format_type,
        "preview": {
            "first_10": [valid_samples[i]["data"] for i in range(min(10, len(valid_samples)))],
            "random_10": preview_samples[10:] if len(preview_samples) > 10 else []
        }
    }
    
    # SECURITY: Conditionally sanitize and anonymize based on user confirmation
    if skip_pii_detection and confirm_data_rights:
        # User confirmed rights - skip PII detection for faster upload
        logger.info(f"Skipping PII detection (user confirmed data rights): {file.filename}")
        sanitized_samples = valid_samples
        sanitization_warnings = []
        anonymization_report = {
            "total_samples": len(valid_samples),
            "samples_with_pii": 0,
            "total_replacements": 0,
            "types_found": {},
            "fields_affected": [],
            "skipped": True,
            "reason": "User confirmed data rights"
        }
    else:
        # User did not confirm rights - enable PII detection and anonymization
        logger.info(f"Sanitizing and anonymizing {len(valid_samples)} samples for security...")
        
        # Process entire content at once for efficiency
        raw_content = '\n'.join(json.dumps(s["data"]) for s in valid_samples)
        sanitized_content, sanitization_warnings, anonymization_report = sanitize_dataset_content(raw_content)
        
        # Parse sanitized content back
        sanitized_samples = []
        try:
            for line in sanitized_content.strip().split('\n'):
                if line:
                    sanitized_samples.append({"data": json.loads(line)})
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing sanitized content: {e}")
            # Fallback: use original samples
            sanitized_samples = [{"data": s["data"]} for s in valid_samples]
            sanitization_warnings.append(f"Anonymization parsing error: {e}")
        
        if sanitization_warnings:
            logger.warning(f"Dataset sanitization warnings: {len(sanitization_warnings)}")
            for warning in sanitization_warnings[:10]:  # Log first 10
                logger.warning(f"  - {warning}")
        
        # Log anonymization summary
        if anonymization_report['total_replacements'] > 0:
            logger.info(
                f"PII Anonymization Summary: {anonymization_report['total_replacements']} "
                f"replacements in {anonymization_report['samples_with_pii']} samples. "
                f"Types: {anonymization_report['types_found']}"
            )
    
    # Save file to storage
    storage_path = f"./storage/datasets/{dataset_id}.jsonl"
    Path("./storage/datasets").mkdir(parents=True, exist_ok=True)
    
    # Write sanitized samples
    with open(storage_path, 'w') as f:
        for sample in sanitized_samples:
            f.write(json.dumps(sample["data"]) + '\n')
    
    # Update validation report with sanitization and anonymization info
    validation_report["sanitization"] = {
        "warnings_count": len(sanitization_warnings),
        "warnings": sanitization_warnings[:10],  # Include in response
        "anonymization": anonymization_report  # Full PII anonymization report
    }
    
    # Create database record
    dataset = Dataset(
        id=dataset_id,
        name=name,
        description=description,
        format=format_type,
        file_path=storage_path,
        size_bytes=len(content),
        num_samples=len(sanitized_samples),  # Use sanitized count
        total_raw_samples=len(valid_samples) + len(errors),
        validation_report=validation_report,
        preview_samples=preview_samples,
        dataset_schema={},
        is_validation_set=is_validation_set
    )
    
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    return DatasetDetailResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        format=dataset.format,
        num_samples=dataset.num_samples,
        total_raw_samples=dataset.total_raw_samples,
        validation_report=dataset.validation_report,
        preview_samples=dataset.preview_samples,
        dataset_schema=dataset.dataset_schema,
        is_validation_set=dataset.is_validation_set,
        created_at=dataset.created_at.isoformat()
    )


@router.post("/datasets/{parent_id}/validation", response_model=DatasetDetailResponse)
async def upload_validation_dataset(
    parent_id: str,
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    confirm_data_rights: bool = Form(True),
    skip_pii_detection: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Upload a validation dataset linked to a parent training dataset."""
    
    # Log user acknowledgment
    if confirm_data_rights:
        logger.info(f"User confirmed data rights for validation upload: {file.filename}")
    else:
        logger.info(f"User did NOT confirm data rights, enabling PII detection for validation: {file.filename}")
    
    # Verify parent dataset exists
    parent_dataset = db.query(Dataset).filter(Dataset.id == parent_id).first()
    if not parent_dataset:
        raise NotFoundError(f"Parent dataset {parent_id} not found")
    
    # Check if parent already has a validation set
    existing_validation = db.query(Dataset).filter(
        Dataset.parent_dataset_id == parent_id
    ).first()
    
    if existing_validation:
        # Delete the existing validation set
        try:
            if existing_validation.file_path:
                assert_safe_path(existing_validation.file_path, ["./storage/datasets", str(Path("./storage/datasets").resolve())])
                os.remove(existing_validation.file_path)
        except FileNotFoundError:
            pass
        db.delete(existing_validation)
        logger.info(f"Replaced existing validation set for dataset {parent_id}")
    
    # Generate ID
    dataset_id = generate_uuid()
    
    if name is None:
        base_name = Path(file.filename).stem
        name = f"{sanitize_filename(base_name)} (Validation)"
    
    # Stream upload to a temporary file in chunks (do not read entire file into RAM)
    import tempfile
    settings = get_settings()
    max_size_bytes = min(settings.max_storage_gb * 1024 * 1024 * 1024, 500 * 1024 * 1024)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jsonl')
    total = 0
    chunk_size = 64 * 1024
    try:
        while chunk := await file.read(chunk_size):
            total += len(chunk)
            if total > max_size_bytes:
                raise ValidationError("File too large for processing. Maximum 500MB allowed.")
            os.write(tmp_fd, chunk)
        os.close(tmp_fd)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            content_str = f.read()
    except UnicodeDecodeError:
        raise ValidationError("File must be valid UTF-8 encoded text")
    finally:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    
    # Validate JSONL format
    is_valid, valid_samples, errors = validate_jsonl_format(content_str)
    
    logger.info(f"Validation file validation: {len(valid_samples)} valid, {len(errors)} errors")
    
    if not valid_samples:
        error_details = errors[:5] if errors else []
        raise ValidationError(f"No valid samples found in validation file. Errors: {error_details}")
    
    # Detect format (should match parent)
    format_type = detect_format(valid_samples)
    
    # Generate previews (first 10 + random 10)
    preview_samples = []
    if valid_samples:
        preview_samples.extend([s["data"] for s in valid_samples[:10]])
        remaining = valid_samples[10:]
        if remaining:
            sample_count = min(10, len(remaining))
            if len(remaining) > sample_count:
                random_samples = random.sample(remaining, sample_count)
            else:
                random_samples = remaining
            preview_samples.extend([s["data"] for s in random_samples])
    
    # Create validation report
    validation_report = {
        "total_samples": len(valid_samples) + len(errors),
        "valid_samples": len(valid_samples),
        "invalid_samples": len(errors),
        "errors": errors[:10],
        "format_detected": format_type,
        "preview": {
            "first_10": [valid_samples[i]["data"] for i in range(min(10, len(valid_samples)))],
            "random_10": preview_samples[10:] if len(preview_samples) > 10 else []
        }
    }
    
    # SECURITY: Conditionally sanitize and anonymize based on user confirmation
    if skip_pii_detection and confirm_data_rights:
        # User confirmed rights - skip PII detection for faster upload
        logger.info(f"Skipping PII detection for validation set (user confirmed data rights): {file.filename}")
        sanitized_samples = valid_samples
        sanitization_warnings = []
        anonymization_report = {
            "total_samples": len(valid_samples),
            "samples_with_pii": 0,
            "total_replacements": 0,
            "types_found": {},
            "fields_affected": [],
            "skipped": True,
            "reason": "User confirmed data rights"
        }
    else:
        # User did not confirm rights - enable PII detection and anonymization
        logger.info(f"Sanitizing and anonymizing {len(valid_samples)} validation samples for security...")
        
        # Process entire content at once
        raw_content = '\n'.join(json.dumps(s["data"]) for s in valid_samples)
        sanitized_content, sanitization_warnings, anonymization_report = sanitize_dataset_content(raw_content)
        
        # Parse sanitized content back
        sanitized_samples = []
        try:
            for line in sanitized_content.strip().split('\n'):
                if line:
                    sanitized_samples.append({"data": json.loads(line)})
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing sanitized validation content: {e}")
            sanitized_samples = [{"data": s["data"]} for s in valid_samples]
            sanitization_warnings.append(f"Anonymization parsing error: {e}")
        
        if sanitization_warnings:
            logger.warning(f"Validation set sanitization warnings: {len(sanitization_warnings)}")
        
        # Log anonymization summary
        if anonymization_report['total_replacements'] > 0:
            logger.info(
                f"Validation Set PII Anonymization: {anonymization_report['total_replacements']} "
                f"replacements in {anonymization_report['samples_with_pii']} samples"
            )
    
    # Update validation report
    validation_report["sanitization"] = {
        "warnings_count": len(sanitization_warnings),
        "warnings": sanitization_warnings[:10],
        "anonymization": anonymization_report
    }
    
    # Save file
    storage_path = f"./storage/datasets/{dataset_id}.jsonl"
    Path("./storage/datasets").mkdir(parents=True, exist_ok=True)
    
    with open(storage_path, 'w') as f:
        for sample in sanitized_samples:
            f.write(json.dumps(sample["data"]) + '\n')
    
    # Create validation dataset record linked to parent
    validation_dataset = Dataset(
        id=dataset_id,
        name=name,
        description=description or f"Validation set for {parent_dataset.name}",
        format=format_type,
        file_path=storage_path,
        size_bytes=len(content),
        num_samples=len(sanitized_samples),  # Use sanitized count
        total_raw_samples=len(valid_samples) + len(errors),
        validation_report=validation_report,
        preview_samples=preview_samples,
        dataset_schema={},
        is_validation_set=True,
        parent_dataset_id=parent_id
    )
    
    db.add(validation_dataset)
    db.commit()
    db.refresh(validation_dataset)
    
    logger.info(f"Created validation dataset {dataset_id} for parent {parent_id} with {len(sanitized_samples)} samples")
    
    return DatasetDetailResponse(
        id=validation_dataset.id,
        name=validation_dataset.name,
        description=validation_dataset.description,
        format=validation_dataset.format,
        num_samples=validation_dataset.num_samples,
        total_raw_samples=validation_dataset.total_raw_samples,
        validation_report=validation_dataset.validation_report,
        preview_samples=validation_dataset.preview_samples,
        dataset_schema=validation_dataset.dataset_schema,
        is_validation_set=validation_dataset.is_validation_set,
        created_at=validation_dataset.created_at.isoformat()
    )


@router.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    """List all datasets with their validation set info."""
    # Get all training datasets
    datasets = db.query(Dataset).filter(Dataset.is_validation_set == False).all()
    
    # Get all validation datasets
    validation_sets = db.query(Dataset).filter(Dataset.is_validation_set == True).all()
    
    # Create lookup for validation sets by parent_id
    validation_lookup = {v.parent_dataset_id: v for v in validation_sets if v.parent_dataset_id}
    
    return [
        DatasetResponse(
            id=d.id,
            name=d.name,
            description=d.description,
            format=d.format,
            num_samples=d.num_samples,
            created_at=d.created_at.isoformat(),
            has_validation_set=d.id in validation_lookup,
            validation_set_id=validation_lookup[d.id].id if d.id in validation_lookup else None,
            validation_samples=validation_lookup[d.id].num_samples if d.id in validation_lookup else 0
        )
        for d in datasets
    ]


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get dataset details."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise NotFoundError(f"Dataset {dataset_id} not found")
    
    return DatasetDetailResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        format=dataset.format,
        num_samples=dataset.num_samples,
        total_raw_samples=dataset.total_raw_samples,
        validation_report=dataset.validation_report,
        preview_samples=dataset.preview_samples,
        dataset_schema=dataset.dataset_schema,
        is_validation_set=dataset.is_validation_set,
        created_at=dataset.created_at.isoformat()
    )


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset and all associated training runs."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise NotFoundError(f"Dataset {dataset_id} not found")
    
    # Delete associated training runs first (to avoid foreign key constraint)
    training_runs = db.query(TrainingRun).filter(
        (TrainingRun.training_dataset_id == dataset_id) | 
        (TrainingRun.validation_dataset_id == dataset_id)
    ).all()
    
    for run in training_runs:
        # Delete run's storage directory if it exists
        try:
            if run.storage_path:
                assert_safe_path(run.storage_path, ["./storage/runs", str(Path("./storage/runs").resolve())])
                if os.path.exists(run.storage_path):
                    shutil.rmtree(run.storage_path)
        except Exception as e:
            logger.warning(f"Could not delete storage for run {run.id}: {e}")
        
        # Delete the run record
        db.delete(run)
    
    # Delete file
    try:
        if dataset.file_path:
            assert_safe_path(dataset.file_path, ["./storage/datasets", str(Path("./storage/datasets").resolve())])
            os.remove(dataset.file_path)
    except FileNotFoundError:
        pass  # Already deleted
    except Exception as e:
        logger.warning(f"Could not delete file for dataset {dataset_id}: {e}")
    
    # Delete database record
    db.delete(dataset)
    db.commit()
    
    logger.info(f"Dataset {dataset_id} deleted successfully (with {len(training_runs)} associated training runs)")
    
    return {"message": f"Dataset deleted successfully (removed {len(training_runs)} associated training runs)"}
