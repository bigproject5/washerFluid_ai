import threading, json, time
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from config import KAFKA_BOOTSTRAP
from schemas import Prediction, InferenceMetrics
from s3_io import download_s3_to_temp, upload_bytes_to_s3
from frame_selector import select_topk_spray_frames
from inference import WasherONNXModel  # ← 원본과 동일하게 유지
from config import TOPK, METHOD, THR_VIDEO, FRAME_EVERY_SEC
from kafka_producer import publish_diagnosis

MODEL = WasherONNXModel()

def convert_s3_to_https(s3_url):
    """S3 URL을 HTTPS URL로 변환"""
    if s3_url and s3_url.startswith('s3://'):
        # s3://aivle-5/evidence/612/612.jpg -> https://aivle-5.s3.amazonaws.com/evidence/612/612.jpg
        path = s3_url.replace('s3://', '')
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return s3_url

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
        label = "ABNORMAL" if is_defect else "NORMAL"
        print(f"[DEBUG] label_idx={label_idx}, is_defect={is_defect}, bool={bool(is_defect)}")

        # ★ 상세 판정 문장 생성 - 함수를 지연 import로 안전하게 처리
        print(f"[kafka_consumer] Generating detailed diagnosis...")
        try:
            from inference import generate_comprehensive_diagnosis  # ← 지연 import
            detailed_diagnosis = generate_comprehensive_diagnosis(label, p)
            print(f"[kafka_consumer] Detailed diagnosis: {detailed_diagnosis}")
        except ImportError as e:
            print(f"[kafka_consumer] Warning: Could not import diagnosis function: {e}")
            detailed_diagnosis = "ABNORMAL" if is_defect else "NORMAL"  # fallback
        except Exception as e:
            print(f"[kafka_consumer] Warning: Diagnosis generation failed: {e}")
            detailed_diagnosis = "ABNORMAL" if is_defect else "NORMAL"  # fallback

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

        # S3 URL을 HTTPS URL로 변환
        frame_https = convert_s3_to_https(frame_s3)
        print(f"[kafka_consumer] Converted S3 URL to HTTPS: {frame_https}")

        print(f"[kafka_consumer] Publishing diagnosis result...")

        payload_data = {
                "auditId": audit_id,
                "inspectionId": inspection_id,
                "inspectionType": inspection_type,
                "isDefect": bool(is_defect),
                "collectDataPath": video_uri,
                "resultDataPath": frame_https,  # S3 URL 대신 HTTPS URL 사용
                "diagnosisResult": detailed_diagnosis  # ← 상세 문장 또는 fallback
            }

        import json
        print(f"[DEBUG] 전송할 JSON: {json.dumps(payload_data, indent=2, ensure_ascii=False)}")

# 기존 publish_diagnosis 호출 수정 ↓ ↓ ↓
        publish_diagnosis(
            key_audit_id=audit_id,
            payload=payload_data
        )
        print(f"[kafka_consumer] Process completed successfully for audit_id: {audit_id}")

    except Exception as e:
        print(f"[kafka_consumer] ERROR in _process for audit_id {audit_id}: {e}")
        import traceback
        traceback.print_exc()

def run_consumer():
    """안전한 consumer 실행"""
    while True:
        consumer = None
        try:
            print("[kafka_consumer] Creating new consumer...")
            consumer = KafkaConsumer(
                "test-started",
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id="washer-fluid-ai",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                # 안정성을 위한 추가 설정
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_interval_ms=300000,
                consumer_timeout_ms=1000,  # 1초 타임아웃
            )
            
            print("[kafka_consumer] Consumer created, listening for messages...")
            
            # poll 방식으로 메시지 처리
            while True:
                try:
                    message_batch = consumer.poll(timeout_ms=1000)
                    
                    if message_batch:
                        for topic_partition, messages in message_batch.items():
                            for msg in messages:
                                try:
                                    _process(msg.value)
                                except Exception as e:
                                    print(f"[kafka_consumer] Message processing error: {e}")
                    
                except Exception as e:
                    print(f"[kafka_consumer] Polling error: {e}")
                    break  # 내부 루프 종료하여 consumer 재생성
                    
        except KeyboardInterrupt:
            print("[kafka_consumer] Shutting down...")
            break
            
        except Exception as e:
            print(f"[kafka_consumer] Consumer error: {e}")
            
        finally:
            if consumer:
                try:
                    consumer.close()
                    print("[kafka_consumer] Consumer closed")
                except Exception as e:
                    print(f"[kafka_consumer] Error closing consumer: {e}")
                    
        print("[kafka_consumer] Retrying in 5 seconds...")
        time.sleep(5)

def start_background_consumer():
    t = threading.Thread(target=run_consumer, daemon=True)
    t.start()
    print("[kafka_consumer] Background consumer thread started")