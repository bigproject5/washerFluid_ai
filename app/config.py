import os, json

MODEL_PATH = os.getenv("MODEL_PATH", "models/washer_model.onnx")
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/washer_config.json")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ai-diagnosis-completed")
S3_BUCKET = os.getenv("S3_BUCKET", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "washer_fluid_ai")
PREDICT_THRESHOLD = float(os.getenv("PREDICT_THRESHOLD", "0.5"))  # fallback

# ---- Load user config (optional) ----
CFG_DEFAULT = {
    "model_name": "efficientnet_b0",
    "img_size": 224,
    "method": "mean",          # frame aggregation: mean|max|vote
    "thr_video": 0.5,          # abnormal probability threshold on aggregated score
    "topk": 1,                 # number of top spray frames to aggregate
    "frame_every_sec": None,   # sample FPS, e.g., 1.0 for 1 fps
    "label_map": {"normal": 0, "abnormal": 1},  # class index mapping
}

CFG = CFG_DEFAULT.copy()
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            user_cfg = json.load(f)
            CFG.update(user_cfg)
        except Exception:
            pass

# normalize label map to lowercase keys
_label_map = {str(k).lower(): int(v) for k, v in CFG.get("label_map", {}).items()}
if not _label_map:
    _label_map = {"normal": 0, "abnormal": 1}

# index -> CLASSNAME (UPPER)
INDEX_TO_CLASS = [None] * (max(_label_map.values()) + 1)
for name, idx in _label_map.items():
    INDEX_TO_CLASS[idx] = name.upper()

# useful constants
IMG_SIZE = int(CFG.get("img_size", 224))
TOPK = int(CFG.get("topk", 1))
METHOD = str(CFG.get("method", "mean")).lower()
THR_VIDEO = float(CFG.get("thr_video", 0.5))
FRAME_EVERY_SEC = CFG.get("frame_every_sec", None)
ABN_IDX = _label_map.get("abnormal", 1)
NOR_IDX = _label_map.get("normal", 0)
