from pydantic import BaseModel
from typing import Optional, Literal, Dict

class InferRequest(BaseModel):
    audit_id: int
    car_id: str
    line_code: str
    process: Literal["WASHER_FLUID"]
    video_s3_uri: str
    expected_type: Literal["BINARY","MULTICLASS"] = "BINARY"

class Prediction(BaseModel):
    label: Literal["NORMAL","ABNORMAL"]
    defect_type: Optional[str] = None
    confidence: float

class InferenceMetrics(BaseModel):
    coverage_ratio: Optional[float] = None
    asymmetry: Optional[float] = None
    spray_angle_deg: Optional[float] = None

class InferResponse(BaseModel):
    status: str
    max_frame_index: int
    max_frame_s3_uri: Optional[str]
    prediction: Prediction
    metrics: InferenceMetrics

class KafkaMessage(BaseModel):
    audit_id: int
    car_id: str
    line_code: str
    process: str
    result: Dict
    timestamp: str