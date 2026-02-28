from datetime import datetime, timezone 
from PIL import Image
import numpy as np 
import uuid 
from typing import List

from app.schemas import (
    PlantAssessment, CropResult, qualityReport, OrganResult, triageResult,
    SeverityResult, Unsertainty, Next_Actions, Model_info, suspect, Evidence
)

def SimpleQualityChecks(img: Image.Image) -> qualityReport:
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    mean = float(arr.mean()) / 255.0

    issues = []
    if h < 1024 or w < 1024:
        issues.append("Image is too small")
    if mean < 0.5:
        issues.append("Image is too dark")
    if mean > 0.9:
        issues.append("Image is too bright")
    
    return qualityReport(
        quality_pass=len(issues) == 0,
        exposure_score=mean,
        subject_detected=True,
        issues=issues
    )

def AnalysePlantImage(img: Image.Image, user_crop: str | None, session_id: str | None = None) -> PlantAssessment:
    session_id = session_id or str(uuid.uuid4())
    start_time = datetime.now(timezone.utc).isoformat()
    quality = SimpleQualityChecks(img)

    triage_status = "unknown" if not quality.quality_pass else "abnormal"
    suspects = []

    if triage_status == "abnormal":
        suspects = [
            suspect(
                label="Unknown_issue",
                type="disease",
                probability=0.34,
                evidence=Evidence(
                    bbox=[[0, 0, 100, 100]],
                    mask_id="mask_id",
                    affected_area_percent=0.34
                )
            )
        ]

    needs_more_info = True
    capture_requests = [
        "closeup of leaf(top surface) filling most of the frame",
        "closeup of leaf(bottom surface) filling most of the frame",
        "closeup of leaf side (front) filling most of the frame",
        "closeup of leaf side (back) filling most of the frame"
    ]

    return PlantAssessment(
        session_id=session_id,
        session_start_time=start_time,
        session_end_time=datetime.now(timezone.utc).isoformat(),
        session_duration=0.0,
        image_id=str(uuid.uuid4()),
        crop=CropResult(
            user_selected=user_crop,
            predicted=user_crop,
            confidence=0.0,
            mismach=False
        ),
        quality_report=quality,
        organs=[OrganResult(type="leaf", confidence=0.0)],
        triage=triageResult(status=triage_status, confidence=0.0),
        suspects=suspects,
        severity=SeverityResult(Level=0, affected_area_percent=0.0, confidence=0.0),
        uncertainty=Unsertainty(low_confidence=True, reasons=["stub_model"]),
        next_actions=Next_Actions(needs_more_info=needs_more_info, capture_requests=capture_requests),
        model_info=Model_info(
            inference_timestamp=datetime.now(timezone.utc).isoformat(),
            model_version="Leaf_v1.0"
        )
    )
