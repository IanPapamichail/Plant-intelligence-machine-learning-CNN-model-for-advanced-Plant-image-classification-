from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Any

crop = Literal["Tomato", "Potato", "Other", "vine", "olive"]
organ = Literal["leaf", "stem", "fruit", "root", "flower", "other"]
triageStatus = Literal["healthy", "sick", "dead", "unknown", "abnormal"]
findingType = Literal["disease", "pest", "nutrient deficiency", "other"]
findingSeverity = Literal["mild", "moderate", "severe", "unknown"]

class qualityReport(BaseModel):
    quality_pass: bool
    blur_score: float = 0.0
    exposure_score: float = 0.0
    subject_detected: bool = True
    issues: List[str] = Field(default_factory=list)

class Evidence(BaseModel):
    bbox: Optional[List[List[int]]] = None
    mask_id: Optional[str] = None
    affected_area_percent: Optional[float] = None

class suspect(BaseModel):
   label: str
   type: findingType
   probability: float = 0.0
   evidence: Optional[Evidence] = None 

class CropResult(BaseModel):
    user_selected: Optional[crop] = None
    predicted: Optional[crop] = None
    confidence: float = 0.0
    mismach: bool = False
    
class OrganResult(BaseModel):
   type: Optional[organ] = None
   confidence: float = 0.0
   
class triageResult(BaseModel):
    status: triageStatus
    confidence: float = 0.0
    
class SeverityResult(BaseModel):
    Level: int = 0
    affected_area_percent: Optional[float] = None
    confidence: float = 0.0
    
class Unsertainty(BaseModel):
    low_confidence: bool = False
    reasons: list[str] = Field(default_factory=list)
    crop_unsertainty: float = 0.0
    organ_unsertainty: float = 0.0
    triage_unsertainty: float = 0.0
    severity_unsertainty: float = 0.0
    
class Next_Actions(BaseModel):
    needs_more_info: bool = False
    capture_requests: List[str] = Field(default_factory=list)

class Model_info(BaseModel):
    model_version: str = "Leaf_v1.0"
    threshold_profile : str = "default"
    inference_timestamp: str
    model_type: str = "Lite Model"
    model_description: str = "IoTREE-Leaf-v1.0 for fast crop Analytics"

class PlantAssessment(BaseModel):
    session_id: str
    session_start_time: str
    session_end_time: str
    session_duration: float
    crop: CropResult
    organs: List[OrganResult]
    triage: triageResult
    suspects: List[suspect] = Field(default_factory=list)   
    severity: SeverityResult
    uncertainty: Unsertainty
    next_actions: Next_Actions
    image_id: str
    quality_report: qualityReport
    model_info: Model_info
