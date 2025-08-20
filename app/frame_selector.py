import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List

# 최대 분출 프레임 상위 K 선택 + 간단 지표 산출 (이전 프레임 비교 방식으로 변경)

def _score_spray(frame_gray, prev_gray, roi_polygon=None):
    """이전 프레임과 비교해서 현재 움직임(분출 진행)을 감지"""
    diff = cv2.absdiff(frame_gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if roi_polygon is not None:
        roi_mask = np.zeros_like(mask)
        
        # 데이터 타입 변환 추가
        try:
            # roi_polygon을 올바른 형태로 변환
            if isinstance(roi_polygon, list):
                roi_polygon = np.array(roi_polygon, dtype=np.int32)
            elif isinstance(roi_polygon, np.ndarray):
                roi_polygon = roi_polygon.astype(np.int32)
            
            # 3차원 배열이 아닌 경우 reshape
            if roi_polygon.ndim == 2:
                roi_polygon = roi_polygon.reshape((-1, 1, 2))
                
            cv2.fillPoly(roi_mask, [roi_polygon], 255)
            mask = cv2.bitwise_and(mask, roi_mask)
            
        except Exception as e:
            print(f"[frame_selector] Warning: ROI polygon error, using full frame: {e}")
            # ROI 오류 시 전체 프레임 사용
    
    score = float((mask > 0).sum())
    return score, mask

def select_topk_spray_frames(video_path: str, k: int = 5, sample_fps: Optional[float] = None,
                             roi_polygon: Optional[np.ndarray] = None) -> Tuple[List[Dict], Dict]:
    print(f"[frame_selector] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[frame_selector] ERROR: Failed to open video file")
        raise RuntimeError(f"Cannot open video: {video_path}")

    print(f"[frame_selector] Reading first frame...")
    ok, first = cap.read()
    if not ok:
        print(f"[frame_selector] ERROR: Cannot read first frame")
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    print(f"[frame_selector] First frame read successfully")
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)  # 이전 프레임으로 사용
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[frame_selector] Video info - FPS: {native_fps}, Total frames: {total_frames}")

    # ★ 짧은 영상 대응: 최소 2-3개 프레임은 처리하도록 보장
    step = 1
    if sample_fps and native_fps > 0 and total_frames > 10:
        step = max(1, int(round(native_fps / float(sample_fps))))
        print(f"[frame_selector] Sampling every {step} frames (target FPS: {sample_fps})")
    else:
        print(f"[frame_selector] Short video detected - processing all frames")

    results = []  # each: {idx, frame, mask, score, image, image_bytes}
    idx = 0
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 첫 번째 프레임은 건너뛰고, 두 번째부터 비교 시작
            if idx > 0:  # processed_frames 대신 idx 사용
                score, mask = _score_spray(gray, prev_gray, roi_polygon)
                
                # 이미지 바이트 변환 (main.py에서 사용)
                _, img_bytes = cv2.imencode('.jpg', frame)
                
                results.append({
                    "idx": idx, 
                    "frame": frame, 
                    "mask": mask, 
                    "score": score,
                    "image": frame,  # AI 모델 입력용
                    "image_bytes": img_bytes.tobytes()  # S3 업로드용
                })
                
                if len(results) % 10 == 0:  # 10프레임마다 로그
                    print(f"[frame_selector] Processed {len(results)} frames...")
            
            # 현재 프레임을 다음 번 비교용 이전 프레임으로 저장
            prev_gray = gray.copy()
            processed_frames += 1

        except Exception as e:
            print(f"[frame_selector] ERROR processing frame {idx}: {e}")
            # 개별 프레임 오류는 건너뛰고 계속 진행

        idx += 1

    cap.release()
    print(f"[frame_selector] Video processing completed. Total processed frames: {processed_frames}")
    print(f"[frame_selector] Valid result frames: {len(results)}")

    # ★ 개선: 결과가 없을 때 fallback 처리
    if not results:
        print(f"[frame_selector] WARNING: No movement detected, using first frame as fallback")
        
        # 첫 번째 프레임을 fallback으로 사용
        _, img_bytes = cv2.imencode('.jpg', first)
        fallback_mask = np.zeros(prev_gray.shape, dtype=np.uint8)  # 빈 마스크
        
        results = [{
            "idx": 0,
            "frame": first,
            "mask": fallback_mask,
            "score": 0.0,  # 최소 점수
            "image": first,
            "image_bytes": img_bytes.tobytes()
        }]
        print(f"[frame_selector] Using fallback frame (first frame)")

    # score desc
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: max(1, k)]
    print(f"[frame_selector] Selected top {len(top)} frames with highest scores")

    # metrics from best (top[0])
    best = top[0]
    best_mask = best["mask"]
    coverage_ratio = float((best_mask > 0).sum()) / float(best_mask.size)
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle_deg = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle_deg = float(abs(rect[2]))
    h, w = best_mask.shape
    left = (best_mask[:, : w // 2] > 0).sum()
    right = (best_mask[:, w // 2 :] > 0).sum()
    asym = abs(left - right) / max(1, (left + right))

    metrics = {
        "coverage_ratio": round(coverage_ratio, 4),
        "asymmetry": round(float(asym), 4),
        "spray_angle_deg": round(angle_deg, 2) if angle_deg is not None else None,
    }

    return top, metrics