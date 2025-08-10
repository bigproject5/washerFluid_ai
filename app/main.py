from fastapi import FastAPI
from datetime import datetime, timezone
import os, cv2
import numpy as np

from .schemas import InferRequest, InferResponse, Prediction, InferenceMetrics
from .frame_selector import select_topk_spray_frames
from .inference import WasherONNXModel
from .kafka_producer import publish_diagnosis
from .config import ROI_POLYGON, TOPK, METHOD, THR_VIDEO, FRAME_EVERY_SEC
from .s3_io import download_s3_to_temp, upload_bytes_to_s3

app = FastAPI(title="washerFluid_ai (ONNX)")
MODEL = WasherONNXModel()

def to_np_polygon(poly_list):
    if not poly_list:
        return None
    arr = np.array(poly_list, dtype=np.int32)
    return arr

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    video_path = download_s3_to_temp(req.video_s3_uri)

    roi_np = to_np_polygon(ROI_POLYGON)
    top_frames, metrics = select_topk_spray_frames(
        video_path, k=max(1, TOPK), sample_fps=FRAME_EVERY_SEC, roi_polygon=roi_np
    )

    # 추론 및 집계
    probs_abn = []
    best = top_frames[0]
    for item in top_frames:
        prob = MODEL.predict_proba(item["frame"])  # dict like {"NORMAL":0.2,"ABNORMAL":0.8}
        p_abn = float(prob.get("ABNORMAL", 0.0))
        probs_abn.append(p_abn)

    if METHOD == "max":
        p_video = max(probs_abn)
    elif METHOD == "vote":
        votes = sum(1 for p in probs_abn if p >= 0.5)
        p_video = votes / len(probs_abn)
    else:  # mean (default)
        p_video = float(np.mean(probs_abn))

    label = "ABNORMAL" if p_video >= THR_VIDEO else "NORMAL"
    confidence = p_video if label == "ABNORMAL" else 1.0 - p_video
    defect = None

    # 증거 프레임은 분사 스코어가 가장 큰 프레임(=top[0])
    ok, jpg = cv2.imencode(".jpg", best["frame"])
    evidence_key = (
        f"evidence/{req.audit_id}/washer/"
        f"{os.path.basename(req.video_s3_uri)}_f{best['idx']:04d}.jpg"
    )
    frame_s3_uri = upload_bytes_to_s3(jpg.tobytes(), evidence_key, "image/jpeg")

    # 동기 응답
    resp = InferResponse(
        status="ok",
        max_frame_index=int(best["idx"]),
        max_frame_s3_uri=frame_s3_uri,
        prediction=Prediction(label=label, defect_type=defect, confidence=round(confidence, 4)),
        metrics=InferenceMetrics(**metrics),
    )

    # Kafka 발행
    publish_diagnosis(
        key_audit_id=req.audit_id,
        payload={
            "audit_id": req.audit_id,
            "car_id": req.car_id,
            "line_code": req.line_code,
            "process": "WASHER_FLUID",
            "result": {
                "label": label,
                "defect_type": defect,
                "confidence": round(confidence, 4),
                "metrics": {**metrics, "video_abnormal_prob": round(p_video, 4)},
                "evidence": {"max_frame_s3_uri": frame_s3_uri, "frame_index": int(best['idx'])},
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return resp
