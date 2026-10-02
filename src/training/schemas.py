"""
Training Schemas (Pydantic Models)
"""

from typing import Optional
from pydantic import BaseModel, Field


class TrainingTriggerRequest(BaseModel):
    model_type: str = Field(..., description="'toddler' or 'hazard'")
    data_yaml: str = Field(..., description="Path to dataset data.yaml file")
    epochs: int = Field(30, ge=1, le=300)
    batch_size: int = Field(16, ge=1, le=128)
    imgsz: int = Field(640, ge=320, le=1280)
    base_weights: Optional[str] = Field(None, description="Optional custom initial weights .pt")


class TrainingStatusResponse(BaseModel):
    job_id: str
    model_type: str
    status: str  # QUEUED, RUNNING, COMPLETED, FAILED
    start_time: str
    epochs: int
    progress: str
    message: str
    weights_path: Optional[str] = None
    error: Optional[str] = None
