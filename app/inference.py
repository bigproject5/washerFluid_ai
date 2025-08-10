import numpy as np
import cv2
from typing import Literal, Tuple, Dict
import onnxruntime as ort
from .config import MODEL_PATH, IMG_SIZE, INDEX_TO_CLASS, ABN_IDX, NOR_IDX

class WasherONNXModel:
    def __init__(self, path: str = MODEL_PATH):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.classes = INDEX_TO_CLASS  # e.g., ["NORMAL","ABNORMAL"]

    def _preprocess(self, bgr):
        img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def predict_proba(self, bgr) -> Dict[str, float]:
        x = self._preprocess(bgr)
        logits = self.session.run(None, {self.input_name: x})[0]
        exps = np.exp(logits - logits.max())
        probs = (exps / exps.sum(axis=1, keepdims=True))[0]
        return {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}

    def predict(self, bgr, expected_type: Literal["BINARY", "MULTICLASS"] = "BINARY") -> Tuple[str, float, str]:
        prob = self.predict_proba(bgr)
        if expected_type == "BINARY":
            p_abn = float(prob.get("ABNORMAL", 0.0))
            p_nor = float(prob.get("NORMAL", 0.0))
            if p_abn >= p_nor:
                return "ABNORMAL", p_abn, None
            else:
                return "NORMAL", p_nor, None
        else:
            best_cls = max(prob, key=prob.get)
            label = "NORMAL" if best_cls == "NORMAL" else "ABNORMAL"
            defect = None if label == "NORMAL" else best_cls
            return label, float(prob[best_cls]), defect
