from app.ollama_client import _normalize_model_json


def test_normalize_model_json_accepts_strictjson_wrapper():
    payload = {
        "strictJSON": '{"answer":"Binary search is O(log n). [S1]","enough_evidence":true,"citations":["S1"],"notes":{}}'
    }

    normalized = _normalize_model_json(payload)

    assert normalized["answer"] == "Binary search is O(log n). [S1]"
    assert normalized["enough_evidence"] is True
