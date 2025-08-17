import numpy as np
import cv2
from typing import Literal, Tuple, Dict
import onnxruntime as ort
from config import MODEL_PATH, IMG_SIZE, INDEX_TO_CLASS, ABN_IDX, NOR_IDX

def softmax(x):
    """Softmax 함수"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def sigmoid(x):
    """Sigmoid 함수"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # overflow 방지

class WasherONNXModel:
    def __init__(self, path: str = MODEL_PATH):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.classes = INDEX_TO_CLASS  # e.g., ["NORMAL","ABNORMAL"]
        
        # 디버깅: 클래스 정보 출력
        print(f"[inference] Loaded classes: {self.classes}")
        print(f"[inference] Model input name: {self.input_name}")

    def _preprocess(self, bgr):
        img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1]  # BGR->RGB
        
        # 모든 계산을 float32로 명시적 수행
        img = img.astype(np.float32) / 255.0
        
        # 정규화도 float32로 수행
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # 차원 변환
        img = np.transpose(img, (2, 0, 1))[None, ...]
        
        # 최종 확인: float32 타입 보장
        return img.astype(np.float32)

    def predict_proba(self, bgr):
        x = self._preprocess(bgr)
        
        # 디버깅: 데이터 타입 확인
        print(f"[inference] Input data type: {x.dtype}, shape: {x.shape}")
        
        logits = self.session.run(None, {self.input_name: x})[0]
        
        print(f"[inference] Raw logits shape: {logits.shape}")
        print(f"[inference] Raw logits: {logits}")
        print(f"[inference] Expected classes: {self.classes}")
        
        # 모델 출력 크기에 따른 처리
        if logits.shape[-1] == 1:
            # 단일 출력: ABNORMAL 확률 (sigmoid 적용)
            abnormal_prob = sigmoid(logits[0][0])
            normal_prob = 1.0 - abnormal_prob
            
            result = {
                "NORMAL": normal_prob,
                "ABNORMAL": abnormal_prob
            }
            print(f"[inference] Single output detected - ABNORMAL: {abnormal_prob:.4f}")
            
        elif logits.shape[-1] == 2:
            # 이중 출력: softmax 적용
            probs = softmax(logits[0])
            result = {
                "NORMAL": float(probs[0]),
                "ABNORMAL": float(probs[1])
            }
            print(f"[inference] Dual output detected - probs: {probs}")
            
        else:
            # 예상하지 못한 출력 크기 처리
            print(f"[inference] WARNING: Unexpected output size {logits.shape}")
            # 첫 번째 값을 ABNORMAL 확률로 사용
            raw_prob = float(logits[0][0])
            
            # logit 값인지 확률값인지 판단
            if abs(raw_prob) > 10:  # logit 값으로 추정
                abnormal_prob = sigmoid(raw_prob)
            elif 0 <= raw_prob <= 1:  # 이미 확률값
                abnormal_prob = raw_prob
            else:  # 다른 형태의 출력
                abnormal_prob = sigmoid(raw_prob)
            
            result = {
                "NORMAL": 1.0 - abnormal_prob,
                "ABNORMAL": abnormal_prob
            }
            print(f"[inference] Fallback processing - ABNORMAL: {abnormal_prob:.4f}")
        
        print(f"[inference] Final result: {result}")
        return result

    def predict(self, bgr, expected_type: Literal["BINARY", "MULTICLASS"] = "BINARY") -> Tuple[int, float]:
        prob = self.predict_proba(bgr)
        
        if expected_type == "BINARY":
            p_abn = float(prob.get("ABNORMAL", 0.0))
            p_nor = float(prob.get("NORMAL", 0.0))
            
            print(f"[inference] NORMAL: {p_nor:.4f}, ABNORMAL: {p_abn:.4f}")
            
            if p_abn >= p_nor:
                print(f"[inference] Final prediction: ABNORMAL (confidence: {p_abn:.4f})")
                return ABN_IDX, p_abn  # 1, confidence
            else:
                print(f"[inference] Final prediction: NORMAL (confidence: {p_nor:.4f})")
                return NOR_IDX, p_nor  # 0, confidence
        else:
            best_cls = max(prob, key=prob.get)
            best_idx = 1 if best_cls == "ABNORMAL" else 0
            confidence = float(prob[best_cls])
            
            print(f"[inference] Final prediction: {best_cls} (confidence: {confidence:.4f})")
            return best_idx, confidence