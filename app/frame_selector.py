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
        cv2.fillPoly(roi_mask, [roi_polygon], 255)
        mask = cv2.bitwise_and(mask, roi_mask)
    score = float((mask > 0).sum())
    return score, mask

def select_topk_spray_frames(video_path: str, k: int = 5, sample_fps: Optional[float] = None,
                             roi_polygon: Optional[np.ndarray] = None) -> Tuple[List[Dict], Dict]:
    cap = cv2.VideoCapture(video_path)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read video: {video_path}")

    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    if sample_fps and native_fps > 0:
        step = max(1, int(round(native_fps / float(sample_fps))))

    results = []  # each: {idx, frame, mask, score}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, mask = _score_spray(gray, first_gray, roi_polygon)
        results.append({"idx": idx, "frame": frame, "mask": mask, "score": score})
        idx += 1

    cap.release()
    if not results:
        raise RuntimeError("Empty video or no frames decoded")

    # score desc
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: max(1, k)]

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