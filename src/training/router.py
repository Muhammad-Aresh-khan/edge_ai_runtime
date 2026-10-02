"""
Training Router: Endpoints for model training and fine-tuning jobs
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends

from src.training.schemas import TrainingTriggerRequest, TrainingStatusResponse
from src.training.service import TrainingService, get_training_service

router = APIRouter(prefix="/train", tags=["Model Training & Fine-Tuning"])


@router.post("/trigger", response_model=TrainingStatusResponse)
async def trigger_training(
    req: TrainingTriggerRequest,
    service: TrainingService = Depends(get_training_service),
):
    model_type = req.model_type.lower().strip()
    if model_type not in ("toddler", "hazard"):
        raise HTTPException(
            status_code=400,
            detail="Invalid model_type. Must be 'toddler' or 'hazard'.",
        )
    return service.trigger(req)


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
):
    status = service.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return status


@router.get("/jobs", response_model=List[TrainingStatusResponse])
async def list_all_jobs(service: TrainingService = Depends(get_training_service)):
    return service.list_jobs()
