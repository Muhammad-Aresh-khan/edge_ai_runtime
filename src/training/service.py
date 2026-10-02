"""
Training Service: Asynchronous YOLO model fine-tuning & background task management
"""

import os
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.config import DEFAULT_TODDLER_MODEL, DEFAULT_HAZARD_MODEL, BASE_DIR
from src.training.schemas import TrainingTriggerRequest, TrainingStatusResponse


class TrainingService:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def _worker(
        self,
        job_id: str,
        model_type: str,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        imgsz: int,
        base_weights: str,
    ):
        from ultralytics import YOLO

        job = self.jobs[job_id]
        job["status"] = "RUNNING"
        job["message"] = f"Training {model_type} on {data_yaml} ({epochs} epochs)..."

        try:
            model = YOLO(base_weights)
            save_dir = os.path.join(BASE_DIR, "runs", "train", job_id)

            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                project=os.path.join(BASE_DIR, "runs", "train"),
                name=job_id,
                exist_ok=True,
                verbose=True,
            )

            best_weights = os.path.join(save_dir, "weights", "best.pt")
            job["status"] = "COMPLETED"
            job["progress"] = f"100% ({epochs}/{epochs} epochs)"
            job["weights_path"] = best_weights if os.path.exists(best_weights) else "Finished"
            job["message"] = f"Training complete! Weights saved to {best_weights}"

        except Exception as e:
            job["status"] = "FAILED"
            job["error"] = str(e)
            job["message"] = f"Training failed: {str(e)}"

    def trigger(self, req: TrainingTriggerRequest) -> TrainingStatusResponse:
        model_type = req.model_type.lower().strip()

        if req.base_weights and os.path.exists(req.base_weights):
            base_weights = req.base_weights
        elif model_type == "toddler":
            base_weights = DEFAULT_TODDLER_MODEL if os.path.exists(DEFAULT_TODDLER_MODEL) else "yolov8n.pt"
        else:
            base_weights = DEFAULT_HAZARD_MODEL if os.path.exists(DEFAULT_HAZARD_MODEL) else "yolov8s.pt"

        job_id = f"train_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        job_record = {
            "job_id": job_id,
            "model_type": model_type,
            "status": "QUEUED",
            "start_time": datetime.now().isoformat(),
            "epochs": req.epochs,
            "progress": f"0/{req.epochs} epochs",
            "message": "Training job queued in background",
            "weights_path": None,
            "error": None,
        }
        self.jobs[job_id] = job_record

        thread = threading.Thread(
            target=self._worker,
            args=(
                job_id,
                model_type,
                req.data_yaml,
                req.epochs,
                req.batch_size,
                req.imgsz,
                base_weights,
            ),
            daemon=True,
        )
        thread.start()

        return TrainingStatusResponse(**job_record)

    def get_status(self, job_id: str) -> Optional[TrainingStatusResponse]:
        if job_id not in self.jobs:
            return None
        return TrainingStatusResponse(**self.jobs[job_id])

    def list_jobs(self) -> List[TrainingStatusResponse]:
        return [TrainingStatusResponse(**job) for job in self.jobs.values()]


_training_service_instance: Optional[TrainingService] = None


def get_training_service() -> TrainingService:
    global _training_service_instance
    if _training_service_instance is None:
        _training_service_instance = TrainingService()
    return _training_service_instance
