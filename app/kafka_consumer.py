import threading, json, time
from kafka import KafkaConsumer
from .config import KAFKA_BOOTSTRAP
from .schemas import Prediction, InferenceMetrics
from .s3_io import download_s3_to_temp, upload_bytes_to_s3
from .frame_selector import select_topk_spray_frames
from .inference import WasherONNXModel
from .config import TOPK, METHOD, THR_VIDEO, FRAME_EVERY_SEC
from .kafka_producer import publish_diagnosis

MODEL = WasherONNXModel()

def _process(event: dict):
    # vehicleAudit TestStartedEventDTO (camelCase) → 필드 매핑
    audit_id       = event["auditId"]
    inspection_id  = event["inspectionId"]
    inspection_type= event["inspectionType"]         # "WASHER_FLUID"
    video_uri      = event["collectDataPath"]

    print(f"[kafka_consumer] Processing audit_id: {audit_id}, inspection_id: {inspection_id}")
    print(f"[kafka_consumer] inspection_type: {inspection_type}, video_uri: {video_uri}")

    # inspectionType 필터링 - WASHER_FLUID만 처리
    if inspection_type != "WASHER_FLUID":
        print(f"[washer_ai] Skipping inspection_type: {inspection_type} (audit_id: {audit_id})")
        return

    try:
        print(f"[kafka_consumer] Starting S3 download...")
        local = download_s3_to_temp(video_uri)
        print(f"[kafka_consumer] S3 download completed. Local file: {local}")

        print(f"[kafka_consumer] Starting frame selection...")
        best_list, metrics = select_topk_spray_frames(local, TOPK, FRAME_EVERY_SEC, METHOD)
        print(f"[kafka_consumer] Frame selection completed. Found {len(best_list)} frames")

        if not best_list:
            print(f"[kafka_consumer] ERROR: No frames found in video")
            return

        best = best_list[0]
        print(f"[kafka_consumer] Best frame index: {best.get('idx', 'unknown')}")

        print(f"[kafka_consumer] Starting AI inference...")
        label_idx, p = MODEL.predict(best["frame"])   # 네 모델에 맞춰 호출
        print(f"[kafka_consumer] AI inference completed. label_idx: {label_idx}, confidence: {p}")

        is_defect = (label_idx == 1)

        # 증거 프레임 업로드
        print(f"[kafka_consumer] Uploading evidence frame...")
        import cv2
        ok, jpg = cv2.imencode(".jpg", best["frame"])
        if not ok:
            print(f"[kafka_consumer] ERROR: Failed to encode frame to JPEG")
            return

        evidence_key = f"evidence/{audit_id}/{inspection_id}.jpg"
        frame_s3 = upload_bytes_to_s3(jpg.tobytes(), evidence_key, "image/jpeg")
        print(f"[kafka_consumer] Evidence frame uploaded to: {frame_s3}")

        print(f"[kafka_consumer] Publishing diagnosis result...")
        publish_diagnosis(
            key_audit_id=audit_id,
            payload={
                "auditId": audit_id,
                "inspectionId": inspection_id,
                "inspectionType": inspection_type,
                "isDefect": bool(is_defect),
                "collectDataPath": video_uri,
                "resultDataPath": frame_s3,
                "diagnosisResult": "ABNORMAL" if is_defect else "NORMAL"
            }
        )
        print(f"[kafka_consumer] Process completed successfully for audit_id: {audit_id}")

    except Exception as e:
        print(f"[kafka_consumer] ERROR in _process for audit_id {audit_id}: {e}")
        import traceback
        traceback.print_exc()

def run_consumer():
    consumer = KafkaConsumer(
        "test-started",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="washer-fluid-ai",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    for msg in consumer:
        try:
            _process(msg.value)
        except Exception as e:
            print("[washer_ai] process error:", e)

def start_background_consumer():
    t = threading.Thread(target=run_consumer, daemon=True)
    t.start()
