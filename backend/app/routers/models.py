"""
Models router - Model registry and exports
"""

from fastapi import APIRouter

router = APIRouter()

# Placeholder - will implement model management endpoints
@router.get("/models")
async def list_models():
    return {"message": "Model management - coming soon"}
