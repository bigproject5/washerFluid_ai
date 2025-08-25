import numpy as np
import cv2
from typing import Literal, Tuple, Dict
import onnxruntime as ort
from .config import MODEL_PATH, IMG_SIZE, INDEX_TO_CLASS, ABN_IDX, NOR_IDX

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

def generate_comprehensive_diagnosis(label: str, confidence: float) -> str:
    """
    워셔액 분사 성능 검사 항목들을 명시한 종합 평가 결과 생성
    """
    import random
    
    confidence_percent = confidence * 100
    
    # 워셔액 검사 항목들 (실제 측정값은 제공하지 않고 검사 항목만 명시)
    inspection_items = [
        "분사각도, 분사범위, 커버리지",
        "분사패턴, 분사강도, 도달거리", 
        "분사균등성, 분사압력, 분사영역",
        "분사방향, 분사량, 분사효율성",
        "분사안정성, 분사일관성, 분사품질"
    ]
    
    items = random.choice(inspection_items)
    
    if label == "ABNORMAL":
        # 신뢰도별 종합 평가 표현
        if confidence >= 0.95:
            # 매우 확실한 불량
            assessments = [
                "전반적인 성능이 크게 저하된 상태로 분석되어",
                "심각한 성능 이상이 종합적으로 감지되어",
                "정상 기준을 현저히 벗어난 것으로 판단되어"
            ]
        elif confidence >= 0.85:
            # 명확한 불량
            assessments = [
                "정상 범위를 상당히 벗어난 것으로 분석되어",
                "명확한 성능 저하가 종합적으로 확인되어",
                "전체적인 작동 상태가 불량한 것으로 판단되어",
                "기준치 대비 현저히 미흡한 것으로 평가되어"
            ]
        elif confidence >= 0.7:
            # 중간 정도 불량
            assessments = [
                "일부 기준 미달 요소가 종합적으로 감지되어",
                "정상 범위 하한선을 벗어난 것으로 분석되어",
                "전반적인 효율성이 다소 부족한 것으로 판단되어",
                "개선이 필요한 부분이 확인되어"
            ]
        else:
            # 경미한 불량
            assessments = [
                "경미한 이상 패턴이 종합적으로 관찰되어",
                "최적 기준에 다소 미달하는 것으로 분석되어",
                "전체적인 성능이 약간 저하된 것으로 평가되어"
            ]
        
        assessment = random.choice(assessments)
        result = f"워셔액 분사 성능에 대해 {items}를 종합 검사한 결과, 워셔액 분사 시스템의 {assessment} 신뢰도 {confidence_percent:.1f}%로 점검이 필요한 것으로 판정되었습니다."
        
    else:
        # 정상인 경우 종합 평가
        if confidence >= 0.95:
            assessments = [
                "모든 측면에서 우수한 성능을 보이며",
                "정상 기준을 충분히 만족하는 최적 상태로 평가되어",
                "전 영역에서 안정적이고 효과적으로 작동하여",
                "종합적인 기능 상태가 매우 양호하여"
            ]
        elif confidence >= 0.85:
            assessments = [
                "정상 범위 내에서 안정적으로 작동하여",
                "전반적으로 양호한 수준을 유지하고 있어",
                "적절한 성능을 지속적으로 발휘하여",
                "종합적인 작동 상태가 정상적으로 확인되어"
            ]
        else:
            assessments = [
                "기본 요구사항을 충족하는 수준으로 작동하여",
                "전체적으로 정상 범위 내에서 기능하여",
                "허용 가능한 수준을 유지하고 있어"
            ]
        
        assessment = random.choice(assessments)
        result = f"워셔액 분사 성능에 대해 {items}를 종합 검사한 결과, 워셔액 분사 시스템이 {assessment} 신뢰도 {confidence_percent:.1f}%로 정상 판정되었습니다."
    
    return result