# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict

# --- vehicleAudit의 TestStartedEventDTO 입력을 그대로 받기 ---
class TestStartedEventDTO(BaseModel):
    audit_id: int = Field(..., alias="auditId")
    model: str
    line_code: str = Field(..., alias="lineCode")
    inspection_id: int = Field(..., alias="inspectionId")
    inspection_type: str = Field(..., alias="inspectionType")  # 예: "WASHER_FLUID"
    collect_data_path: str = Field(..., alias="collectDataPath")  # S3 경로(영상/이미지)

    class Config:
        allow_population_by_field_name = True  # snake_case도 허용

# (참고) 기존 Prediction/InferResponse 등은 필요시 유지
class Prediction(BaseModel):
    label: Literal["NORMAL","ABNORMAL"]
    defect_type: Optional[str] = None
    confidence: float

class InferenceMetrics(BaseModel):
    # 필요 값들 정의(예시)
    frame_score_mean: Optional[float] = None
    frame_score_max: Optional[float] = None

class InferResponse(BaseModel):
    status: str
    max_frame_index: Optional[int]
    max_frame_s3_uri: Optional[str]
    prediction: Prediction
    metrics: InferenceMetrics

# --- vehicleAudit의 AiDiagnosisCompletedEventDTO로 카프카 발행 ---
class AiDiagnosisCompletedEventDTO(BaseModel):
    audit_id: int = Field(..., alias="auditId")
    inspection_id: int = Field(..., alias="inspectionId")
    inspection_type: str = Field(..., alias="inspectionType")   # "WASHER_FLUID"
    is_defect: bool = Field(..., alias="isDefect")
    collect_data_path: str = Field(..., alias="collectDataPath")
    result_data_path: Optional[str] = Field(None, alias="resultDataPath")  # 증거 프레임 등
    diagnosis_result: str = Field(..., alias="diagnosisResult")            # "NORMAL" 또는 "ABNORMAL: <defect>"

    class Config:
        allow_population_by_field_name = True
