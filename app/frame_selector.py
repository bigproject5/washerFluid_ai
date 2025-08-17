import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List

# 최대 분출 프레임 상위 K 선택 + 간단 지표 산출

def _score_spray(frame_gray, first_gray, roi_polygon=None):
    diff = cv2.absdiff(frame_gray, first_gray)
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
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[frame_selector] Video info - FPS: {native_fps}, Total frames: {total_frames}")

    step = 1
    if sample_fps and native_fps > 0:
        step = max(1, int(round(native_fps / float(sample_fps))))
        print(f"[frame_selector] Sampling every {step} frames (target FPS: {sample_fps})")

    results = []  # each: {idx, frame, mask, score}
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
            score, mask = _score_spray(gray, first_gray, roi_polygon)
            results.append({"idx": idx, "frame": frame, "mask": mask, "score": score})
            processed_frames += 1

            if processed_frames % 10 == 0:  # 10프레임마다 로그
                print(f"[frame_selector] Processed {processed_frames} frames...")

        except Exception as e:
            print(f"[frame_selector] ERROR processing frame {idx}: {e}")
            # 개별 프레임 오류는 건너뛰고 계속 진행

        idx += 1

    cap.release()
    print(f"[frame_selector] Video processing completed. Total processed frames: {processed_frames}")

    if not results:
        print(f"[frame_selector] ERROR: No valid frames found")
        raise RuntimeError("Empty video or no frames decoded")

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