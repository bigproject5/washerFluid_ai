# app/main.py
import os
from fastapi import FastAPI, HTTPException

# .env 사용하고 싶으면(선택)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .schemas import (
    TestStartedEventDTO,
    AiDiagnosisCompletedEventDTO,
    InferResponse,
    Prediction,
    InferenceMetrics,
)
from .frame_selector import select_topk_spray_frames
from .inference import WasherONNXModel
from .kafka_producer import publish_diagnosis
from .kafka_consumer import start_background_consumer   # ★ 컨슈머 시작용
from .config import TOPK, METHOD, THR_VIDEO, FRAME_EVERY_SEC
from .s3_io import download_s3_to_temp, upload_bytes_to_s3

app = FastAPI(title="washerFluid_ai (ONNX)")
MODEL = WasherONNXModel()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/infer", response_model=InferResponse)
def infer(req: TestStartedEventDTO):
    # 1) 입력 매핑
    video_uri = req.collect_data_path  # vehicleAudit DTO와 일치

    # 2) 영상 다운로드 & 전처리
    local_path = download_s3_to_temp(video_uri)
    best_list, metrics = select_topk_spray_frames(
        local_path, topk=TOPK, every_sec=FRAME_EVERY_SEC, method=METHOD
    )
    if not best_list:
        raise HTTPException(status_code=422, detail="no valid frame found")
    best = best_list[0]

    # 3) 분류 추론
    label_idx, confidence = MODEL.predict(best["image"])
    label = "ABNORMAL" if label_idx == 1 else "NORMAL"

    # 4) 증거 프레임 업로드
    frame_s3_uri = None
    if best.get("image_bytes"):
        frame_key = f"evidence/{req.audit_id}/{req.inspection_id}.jpg"
        frame_s3_uri = upload_bytes_to_s3(
            best["image_bytes"], key=frame_key, content_type="image/jpeg"
        )

    # 5) HTTP 응답(디버그/확인용)
    resp = InferResponse(
        status="OK",
        max_frame_index=int(best["idx"]),
        max_frame_s3_uri=frame_s3_uri,
        prediction=Prediction(label=label, defect_type=None, confidence=float(confidence)),
        metrics=InferenceMetrics(**metrics),
    )

    # 6) vehicleAudit에 보낼 카프카 이벤트(완료)
    done = AiDiagnosisCompletedEventDTO(
        audit_id=req.audit_id,
        inspection_id=req.inspection_id,
        inspection_type=req.inspection_type,   # "WASHER_FLUID"
        is_defect=(label == "ABNORMAL"),
        collect_data_path=video_uri,
        result_data_path=frame_s3_uri,
        diagnosis_result=label,
    )
    publish_diagnosis(
        key_audit_id=req.audit_id,
        payload=done.dict(by_alias=True, exclude_none=True),
    )
    return resp

@app.on_event("startup")
def _startup():
    start_background_consumer()  # 백그라운드로 test-started 구독 시작
