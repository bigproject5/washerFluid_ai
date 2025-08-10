from kafka import KafkaProducer
import json
from .config import KAFKA_BOOTSTRAP, KAFKA_TOPIC

_producer = None

def get_producer():
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            linger_ms=10,
        )
    return _producer

def publish_diagnosis(key_audit_id: int, payload: dict):
    p = get_producer()
    p.send(KAFKA_TOPIC, key=key_audit_id, value=payload)
    p.flush()