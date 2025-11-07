# bedrock_logging_client.py
# -----------------------------------------------------------------------------
# Bedrock logger:
# - Uses IAM role (no keys) on SageMaker/EC2
# - Reads model IDs from env OR an Inference Profile ARN/ID
# - If BEDROCK_INFERENCE_PROFILE_* is provided, uses it as modelId
# - Uses cross-region inference profile IDs (us. prefix) by default
# - Logs full request, response JSON, headers (ResponseMetadata), status, timing
# - Writes pretty JSON per call in CURRENT DIR + appends bedrock_calls.jsonl
# -----------------------------------------------------------------------------

import os, json, time, uuid, hashlib, pathlib, logging
from datetime import datetime
from typing import Any, Dict, Optional, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# -------------------- ENV --------------------
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")

# Claude Sonnet 4 requires inference profile - use cross-region inference profile ID
# Cross-region inference profiles work across all regions without needing ARNs
BEDROCK_MODEL_ID_TEXT = os.getenv(
    "BEDROCK_MODEL_ID_TEXT",
    "us.anthropic.claude-sonnet-4-20250514-v1:0"
)

# Titan v2 embeddings - use cross-region inference profile
BEDROCK_MODEL_ID_EMBED = os.getenv(
    "BEDROCK_MODEL_ID_EMBED",
    "us.amazon.titan-embed-text-v2:0"
)

# If you must use a specific profile, set one of these (will override above):
BEDROCK_INFERENCE_PROFILE_ARN = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN", "")
BEDROCK_INFERENCE_PROFILE_ID  = os.getenv("BEDROCK_INFERENCE_PROFILE_ID", "")
# If either is set, we will pass it as the modelId directly. (Bedrock allows using
# a foundation model id OR an inference profile id/ARN as modelId.)  ← docs confirm. 

# -------------------- PRICING (optional) --------------------
PRICE_PER_1K = {
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.00, "output": 15.00},
    "anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.00, "output": 15.00},
    "us.amazon.titan-embed-text-v2:0": {"embed": 0.10},
    "amazon.titan-embed-text-v2:0": {"embed": 0.10},
}

# -------------------- FILES --------------------
RUN_DIR = pathlib.Path(".").resolve()
JSONL_PATH = RUN_DIR / "bedrock_calls.jsonl"
logging.basicConfig(level=logging.INFO)

# -------------------- HELPERS --------------------
def _now() -> float: return time.time()

def _hash_payload(payload: Dict[str, Any]) -> str:
    try:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    except Exception:
        return str(uuid.uuid4())

def _mk_filename(endpoint: str, status: int) -> pathlib.Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    base = endpoint.strip("/").replace("/", "_") or "root"
    return RUN_DIR / f"bedrock_call__{base}__{status}__{ts}.json"

def _safe_json_dump(obj: dict, path: pathlib.Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

def _write_jsonl(record: Dict[str, Any]) -> None:
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _estimate_cost(model_id: str, *, prompt_tokens=0, completion_tokens=0, embed_tokens=0) -> Optional[float]:
    price = PRICE_PER_1K.get(model_id)
    if not price: return None
    total = 0.0
    if prompt_tokens or completion_tokens:
        total += (prompt_tokens/1000.0)*price.get("input", 0.0)
        total += (completion_tokens/1000.0)*price.get("output", 0.0)
    if embed_tokens:
        total += (embed_tokens/1000.0)*price.get("embed", 0.0)
    return round(total, 6)

def _effective_model_id(fallback_mid: str) -> str:
    # Prefer an inference profile ARN/ID if provided; otherwise use the model ID as-is
    if BEDROCK_INFERENCE_PROFILE_ARN:
        return BEDROCK_INFERENCE_PROFILE_ARN
    if BEDROCK_INFERENCE_PROFILE_ID:
        return BEDROCK_INFERENCE_PROFILE_ID
    # Return the model ID unchanged - cross-region profiles need the "us." prefix
    return fallback_mid

def _store_full_call_json(*, endpoint, method, status_code, request_payload,
                          response_json, response_text_fallback, response_headers, t_start, t_end) -> pathlib.Path:
    record = {
        "http": {"method": method, "status_code": status_code, "headers": response_headers or {}},
        "request": {
            "endpoint": endpoint, "payload": request_payload,
            "sent_at": datetime.utcfromtimestamp(t_start).isoformat()+"Z",
            "request_hash": _hash_payload(request_payload),
        },
        "response": {
            "body": response_json if response_json is not None else {"non_json_body": response_text_fallback},
            "model": (response_json or {}).get("modelId") or (response_json or {}).get("model") or None,
        },
        "timing": {"started_at": t_start, "completed_at": t_end, "latency_ms": round((t_end-t_start)*1000, 2)},
    }
    out = _mk_filename(endpoint, status_code)
    _safe_json_dump(record, out)
    return out

def _emit_compact_log(*, endpoint, status_code, request_payload, response_json, headers, t_start, t_end, usage, cost, error=None):
    rec = {
        "ts": datetime.utcnow().isoformat()+"Z",
        "endpoint": endpoint, "status_code": status_code,
        "request_hash": _hash_payload(request_payload),
        "timing": {"latency_ms": round((t_end-t_start)*1000, 2), "started_at": t_start, "completed_at": t_end},
        "headers": headers, "usage": usage or {}, "cost_estimate_usd": cost, "error": error,
    }
    _write_jsonl(rec)
    logging.info(f"[{endpoint}] {status_code} • {rec['timing']['latency_ms']} ms • request-id:{headers.get('RequestId') or headers.get('x-amzn-requestid')}")

def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=Config(retries={"max_attempts": 3, "mode": "standard"}))

# -------------------- Converse (preferred) --------------------
def bedrock_converse(messages: List[Dict[str, Any]], model_id: str,
                     inference_config: Optional[Dict[str, Any]] = None,
                     additional_model_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    messages: [{"role":"user","content":[{"text":"Hello"}]}]
    For Claude, don't send both temperature and topP. Use one.
    """
    client = _bedrock_client()
    model_id = _effective_model_id(model_id)
    req_payload = {"modelId": model_id, "messages": messages}
    if inference_config: req_payload["inferenceConfig"] = inference_config
    if additional_model_fields: req_payload["additionalModelRequestFields"] = additional_model_fields

    endpoint, t0 = "/converse", _now()
    try:
        resp = client.converse(**req_payload)
        t1 = _now()
        meta = resp.get("ResponseMetadata", {})
        status = meta.get("HTTPStatusCode", 200)
        headers = {**meta.get("HTTPHeaders", {}), "RequestId": meta.get("RequestId")}
        body = resp

        usage = body.get("usage") or {}
        pt = usage.get("inputTokens", 0); ct = usage.get("outputTokens", 0); tt = usage.get("totalTokens", pt+ct)
        cost = _estimate_cost(model_id, prompt_tokens=pt, completion_tokens=ct)

        _store_full_call_json(endpoint=endpoint, method="POST", status_code=status, request_payload=req_payload,
                              response_json=body, response_text_fallback=None, response_headers=headers, t_start=t0, t_end=t1)
        _emit_compact_log(endpoint=endpoint, status_code=status, request_payload=req_payload, response_json=body,
                          headers=headers, t_start=t0, t_end=t1, usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}, cost=cost)
        if status >= 400: raise RuntimeError(f"Bedrock Converse error {status}")
        return body
    except ClientError as e:
        t1 = _now()
        meta = getattr(e, "response", {}).get("ResponseMetadata", {})
        status = meta.get("HTTPStatusCode", 500)
        headers = {**meta.get("HTTPHeaders", {}), "RequestId": meta.get("RequestId")}
        _store_full_call_json(endpoint=endpoint, method="POST", status_code=status or 500, request_payload=req_payload,
                              response_json=None, response_text_fallback=str(e), response_headers=headers, t_start=t0, t_end=t1)
        _emit_compact_log(endpoint=endpoint, status_code=status or 500, request_payload=req_payload, response_json=None,
                          headers=headers, t_start=t0, t_end=t1, usage=None, cost=None, error=str(e))
        raise

# -------------------- invoke_model (text; pass-through) --------------------
def bedrock_invoke_text(model_id: str, **provider_params) -> Dict[str, Any]:
    """
    Pass provider_params exactly as the model expects (e.g., Anthropic messages + anthropic_version).
    Example:
      bedrock_invoke_text(
        model_id=BEDROCK_MODEL_ID_TEXT,
        messages=[{"role":"user","content":[{"type":"text","text":"Hi"}]}],
        max_tokens=200, temperature=0.2, anthropic_version="bedrock-2023-05-31",
      )
    """
    client = _bedrock_client()
    model_id = _effective_model_id(model_id)
    endpoint = "/invoke_model"
    body = provider_params

    req_payload = {"modelId": model_id, "contentType": "application/json", "accept": "application/json", "body": json.dumps(body)}
    t0 = _now()
    try:
        resp = client.invoke_model(**req_payload)
        t1 = _now()
        meta = resp.get("ResponseMetadata", {}); status = meta.get("HTTPStatusCode", 200)
        headers = {**meta.get("HTTPHeaders", {}), "RequestId": meta.get("RequestId")}
        payload = json.loads(resp.get("body").read().decode("utf-8"))

        usage = payload.get("usage") or {}
        pt = usage.get("inputTokens", usage.get("prompt_tokens", 0))
        ct = usage.get("outputTokens", usage.get("completion_tokens", 0))
        tt = usage.get("totalTokens", usage.get("total_tokens", pt+ct))
        cost = _estimate_cost(model_id, prompt_tokens=pt, completion_tokens=ct)

        _store_full_call_json(endpoint=endpoint, method="POST", status_code=status, request_payload=req_payload,
                              response_json=payload, response_text_fallback=None, response_headers=headers, t_start=t0, t_end=t1)
        _emit_compact_log(endpoint=endpoint, status_code=status, request_payload=req_payload, response_json=payload,
                          headers=headers, t_start=t0, t_end=t1, usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}, cost=cost)
        if status >= 400: raise RuntimeError(f"Bedrock invoke_model error {status}")
        return payload
    except ClientError as e:
        t1 = _now()
        meta = getattr(e, "response", {}).get("ResponseMetadata", {})
        status = meta.get("HTTPStatusCode", 500)
        headers = {**meta.get("HTTPHeaders", {}), "RequestId": meta.get("RequestId")}
        _store_full_call_json(endpoint=endpoint, method="POST", status_code=status or 500, request_payload=req_payload,
                              response_json=None, response_text_fallback=str(e), response_headers=headers, t_start=t0, t_end=t1)
        _emit_compact_log(endpoint=endpoint, status_code=status or 500, request_payload=req_payload, response_json=None,
                          headers=headers, t_start=t0, t_end=t1, usage=None, cost=None, error=str(e))
        raise

# -------------------- Embeddings (Titan v2 / Cohere) --------------------
def bedrock_embeddings(texts: List[str], model_id: str) -> Dict[str, Any]:
    client = _bedrock_client()
    mid = _effective_model_id(model_id)
    endpoint = "/invoke_model"

    if "titan-embed" in mid:
        body = {"inputText": texts[0]}
    elif "cohere" in mid:
        body = {"texts": texts, "input_type": "search_document"}
    else:
        body = {"texts": texts}

    req_payload = {"modelId": mid, "contentType": "application/json", "accept": "application/json", "body": json.dumps(body)}
    t0 = _now()
    try:
        resp = client.invoke_model(**req_payload)
        t1 = _now()
        meta = resp.get("ResponseMetadata", {}); status = meta.get("HTTPStatusCode", 200)
        headers = {**meta.get("HTTPHeaders", {}), "RequestId": meta.get("RequestId")}
        payload = json.loads(resp.get("body").read().decode("utf-8"))

        _store_full_call_json(endpoint=endpoint, method="POST", status_code=status, request_payload=req_payload,
                              response_json=payload, response_text_fallback=None, response_headers=headers, t_start=t0, t_end=t1)
        _emit_compact_log(endpoint=endpoint, status_code=status, request_payload=req_payload, response_json=payload,
                          headers=headers, t_start=t0, t_end=t1, usage={}, cost=_estimate_cost(mid, embed_tokens=0))
        if status >= 400: raise RuntimeError(f"Bedrock embeddings error {status}")
        return payload
    except ClientError as e:
        t1 = _now()
        meta = getattr(e, "response", {}).get("ResponseMetadata", {})
        status = meta.get("HTTPStatusCode", 500)
        headers = {**meta.get("HTTPHeaders", {}), "RequestId": meta.get("RequestId")}
        _store_full_call_json(endpoint=endpoint, method="POST", status_code=status or 500, request_payload=req_payload,
                              response_json=None, response_text_fallback=str(e), response_headers=headers, t_start=t0, t_end=t1)
        _emit_compact_log(endpoint=endpoint, status_code=status or 500, request_payload=req_payload, response_json=None,
                          headers=headers, t_start=t0, t_end=t1, usage=None, cost=None, error=str(e))
        raise

# -------------------- Demo --------------------
if __name__ == "__main__":
    print(f"Region: {BEDROCK_REGION}")
    print(f"Text model: {BEDROCK_MODEL_ID_TEXT}")
    print(f"Embed model: {BEDROCK_MODEL_ID_EMBED}")
    if BEDROCK_INFERENCE_PROFILE_ARN or BEDROCK_INFERENCE_PROFILE_ID:
        print("Using Inference Profile as modelId:", BEDROCK_INFERENCE_PROFILE_ARN or BEDROCK_INFERENCE_PROFILE_ID)

    # 1) Converse (Claude Sonnet 4 on-demand OR profile if provided)
    messages = [{"role":"user","content":[{"text":"Give me a 1-sentence summary of RAG."}]}]
    inference_cfg = {"temperature": 0.3, "maxTokens": 256}  # don't set topP with temperature on Claude
    try:
        res = bedrock_converse(messages=messages, model_id=BEDROCK_MODEL_ID_TEXT, inference_config=inference_cfg)
        out_text = ""
        if isinstance(res, dict):
            msg = (res.get("output") or {}).get("message", {})
            parts = msg.get("content", [])
            out_text = "".join(p.get("text","") for p in parts if isinstance(p, dict) and p.get("text"))
        print("Converse output:", out_text or res)
    except Exception as e:
        print("Converse error:", e)

    # 2) invoke_model (Anthropic messages API)
    try:
        legacy = bedrock_invoke_text(
            model_id=BEDROCK_MODEL_ID_TEXT,
            messages=[{"role":"user","content":[{"type":"text","text":"List two benefits of JSON mode."}]}],
            max_tokens=200, temperature=0.2, anthropic_version="bedrock-2023-05-31",
        )
        print("Invoke (Anthropic) keys:", list(legacy.keys()))
    except Exception as e:
        print("invoke_model error:", e)

    # 3) Embeddings (Titan v2)
    try:
        emb = bedrock_embeddings(texts=["hello world"], model_id=BEDROCK_MODEL_ID_EMBED)
        print("Embeddings response keys:", list(emb.keys()))
    except Exception as e:
        print("Embeddings error:", e)