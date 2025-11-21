 # openai_logging_client.py
# -----------------------------------------------------------------------------
# A drop-in OpenAI client that:
# - Captures full request payloads (with a redact hook)
# - Captures full response JSON (or text fallback) + ALL response headers
# - Persists a pretty JSON file per call in the CURRENT WORKING DIRECTORY
# - Appends a JSONL log line per call for analytics
# - Measures latency (and TTFB for streaming if you wire it)
# - Estimates cost (edit PRICE_PER_1K for your account)
# - Supports: /responses, /chat/completions, /embeddings, /images/generations,
#             /audio/transcriptions (multipart), /files (multipart), /batches,
#             /fine_tuning/jobs
# -----------------------------------------------------------------------------

import os, time, uuid, json, logging, hashlib, pathlib
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, Generator, Tuple
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Optional for server-sent events (streaming); not required for basic usage
try:
    from sseclient import SSEClient  # pip install sseclient-py
except Exception:
    SSEClient = None

# --- Basics ------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-REPLACE_ME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

logging.basicConfig(level=logging.INFO)

# Files saved in the RUNNING DIRECTORY
RUN_DIR = pathlib.Path(".").resolve()
JSONL_PATH = RUN_DIR / "openai_calls.jsonl"  # rolling compact log

# --- Redaction hook (customize for PII/secret scrubbing) ---------------------
def redact(obj: Any) -> Any:
    try:
        s = json.dumps(obj)
        if OPENAI_API_KEY:
            s = s.replace(OPENAI_API_KEY, "****")
        return json.loads(s)
    except Exception:
        return obj

def write_jsonl(record: Dict[str, Any]) -> None:
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- Token price table (edit with your plan/prices) --------------------------
# Units: USD per 1K tokens
PRICE_PER_1K = {
    "gpt-4.1": {"input": 5.00, "output": 15.00},
    "gpt-4.1-mini": {"input": 0.30, "output": 1.20},
    "o4-mini": {"input": 1.10, "output": 4.40},
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    price = PRICE_PER_1K.get(model)
    if not price:
        return None
    return round(
        (prompt_tokens / 1000.0) * price["input"] +
        (completion_tokens / 1000.0) * price["output"], 6
    )

# --- Utilities ----------------------------------------------------------------
def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    base = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        base.update(extra)
    return base

def _ratelimit_subset(hdrs: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in hdrs.items() if k.lower().startswith("x-ratelimit-")}

def _normalize_headers(hdrs) -> dict:
    try:
        return {k: v for k, v in hdrs.items()}
    except Exception:
        return dict(hdrs or {})

def _now() -> float:
    return time.time()

def _hash_payload(payload: Dict[str, Any]) -> str:
    try:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    except Exception:
        return str(uuid.uuid4())

def _mk_filename(endpoint: str, status: int) -> pathlib.Path:
    # Save pretty JSON into the CURRENT directory, not a subfolder
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    base = endpoint.strip("/").replace("/", "_") or "root"
    fname = f"openai_call__{base}__{status}__{ts}.json"
    return RUN_DIR / fname

def _safe_json_dump(obj: dict, path: pathlib.Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

def _store_full_call_json(
    *,
    endpoint: str,
    method: str,
    status_code: int,
    request_payload: dict,
    response_json: dict | None,
    response_text_fallback: str | None,
    response_headers: dict,
    t_start: float,
    t_end: float
) -> pathlib.Path:
    record = {
        "http": {
            "method": method,
            "status_code": status_code,
            "headers": _normalize_headers(response_headers),
        },
        "request": {
            "endpoint": endpoint,
            "payload": redact(request_payload),
            "sent_at": datetime.utcfromtimestamp(t_start).isoformat() + "Z",
            "request_hash": _hash_payload(request_payload),
        },
        "response": {
            "body": response_json if response_json is not None else {"non_json_body": response_text_fallback},
            "id": (response_json or {}).get("id"),
            "created": (response_json or {}).get("created"),
            "model": (response_json or {}).get("model"),
            "system_fingerprint": (response_json or {}).get("system_fingerprint"),
        },
        "timing": {
            "started_at": t_start,
            "completed_at": t_end,
            "latency_ms": round((t_end - t_start) * 1000, 2),
        },
    }
    out_path = _mk_filename(endpoint, status_code)
    _safe_json_dump(record, out_path)
    return out_path

def _emit_log(
    *,
    endpoint: str,
    method: str,
    status_code: int,
    request_id: str,
    request_payload: Dict[str, Any],
    response_json: Optional[Dict[str, Any]],
    response_headers: Dict[str, str],
    t_start: float,
    t_first_byte: Optional[float],
    t_end: float,
    error_body: Optional[Any] = None,
):
    usage = (response_json or {}).get("usage") or {}
    model = (response_json or {}).get("model")
    finish_reason = None

    # Responses API shape
    if (response_json or {}).get("output"):
        out = response_json["output"]
        if isinstance(out, list) and out:
            finish_reason = out[0].get("finish_reason")

    # Chat Completions shape
    choices = (response_json or {}).get("choices")
    if isinstance(choices, list) and choices:
        finish_reason = choices[0].get("finish_reason") or finish_reason

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    cost = estimate_cost(model or "", prompt_tokens, completion_tokens)
    hdrs = {k.lower(): v for k, v in response_headers.items()}

    record = {
        "ts": datetime.utcfromtimestamp(t_end).isoformat() + "Z",
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "request_id_client": str(uuid.uuid4()),
        "request_id_server": hdrs.get("x-request-id"),
        "request_hash": _hash_payload(request_payload),
        "request_payload": redact(request_payload),
        "response_json": response_json,
        "headers": {
            "ratelimit": _ratelimit_subset(hdrs),
            "all": hdrs
        },
        "timing": {
            "started_at": t_start,
            "first_byte_at": t_first_byte,
            "completed_at": t_end,
            "ttfb_ms": None if t_first_byte is None else round((t_first_byte - t_start) * 1000, 2),
            "latency_ms": round((t_end - t_start) * 1000, 2),
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "details": usage,
        },
        "model": model,
        "system_fingerprint": (response_json or {}).get("system_fingerprint"),
        "finish_reason": finish_reason,
        "error": error_body,
        "cost_estimate_usd": cost,
    }
    write_jsonl(record)
    logging.info(f"[{endpoint}] {status_code} • {record['timing']['latency_ms']} ms • req:{record['request_hash']} • x-request-id:{record['headers']['all'].get('x-request-id')}")

# --- Client -------------------------------------------------------------------
class OpenAILoggingClient:
    def __init__(self, base_url: str = OPENAI_BASE_URL, api_key: str = OPENAI_API_KEY):
        self.base_url = base_url
        self.api_key = api_key

    # --------------- Core POST JSON ---------------
    def _post_json(self, path: str, payload: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        t0 = _now()
        resp = requests.post(url, headers=_headers(extra_headers), json=payload)
        t1 = _now()

        body = None
        text_fallback = None
        try:
            body = resp.json()
        except Exception:
            text_fallback = resp.text

        # Save full response (pretty JSON) in CURRENT directory
        stored = _store_full_call_json(
            endpoint=path,
            method="POST",
            status_code=resp.status_code,
            request_payload=payload,
            response_json=body,
            response_text_fallback=text_fallback,
            response_headers=resp.headers,
            t_start=t0,
            t_end=t1,
        )

        # Emit compact JSONL record
        _emit_log(
            endpoint=path, method="POST", status_code=resp.status_code,
            request_id=str(uuid.uuid4()), request_payload=payload,
            response_json=body, response_headers=resp.headers,
            t_start=t0, t_first_byte=None, t_end=t1,
            error_body=(None if resp.ok else (body or {"non_json_body": text_fallback}))
        )

        if not resp.ok:
            raise RuntimeError(f"OpenAI error {resp.status_code} (saved {stored.name}): {body or text_fallback}")
        return body

    # --------------- Streaming (optional) ---------------
    def _post_stream(self, path: str, payload: Dict[str, Any]) -> Tuple[Generator[str, None, Dict[str, Any]], str]:
        if SSEClient is None:
            raise RuntimeError("sseclient-py not installed. pip install sseclient-py")

        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = _headers({"Accept": "text/event-stream"})
        t0 = _now()
        resp = requests.post(url, headers=headers, json=payload, stream=True)
        ttfb = _now()

        if resp.status_code != 200:
            text = resp.text
            _emit_log(
                endpoint=path, method="POST", status_code=resp.status_code,
                request_id=str(uuid.uuid4()), request_payload=payload,
                response_json=None, response_headers=resp.headers,
                t_start=t0, t_first_byte=ttfb, t_end=_now(), error_body=text
            )
            raise RuntimeError(f"OpenAI streaming error {resp.status_code}: {text}")

        client = SSEClient(resp)
        collected = []
        chunk_times = []

        def gen() -> Generator[str, None, Dict[str, Any]]:
            for event in client.events():
                if event.event == "message":
                    chunk_times.append(_now())
                    data = event.data
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        collected.append(j)
                        if "choices" in j:  # chat.completions stream
                            delta = j["choices"][0].get("delta", {})
                            txt = delta.get("content") or ""
                            if txt:
                                yield txt
                        elif "output" in j:  # responses stream; adapt if needed
                            yield j.get("output_text") or ""
                    except json.JSONDecodeError:
                        pass
            return {"chunks": len(collected), "last_chunk_at": chunk_times[-1] if chunk_times else None}

        # We still log a non-stream summary at the end via caller if desired
        return gen(), url

    # --------------- Public API wrappers ---------------
    # Modern unified endpoint: /v1/responses
    def responses(self, **kwargs) -> Dict[str, Any]:
        # NOTE: Responses API uses text.format for JSON/Schema.
        # For plain text, omit 'text' entirely; e.g., text={"format":"json_object"}
        return self._post_json("/responses", kwargs)

    def responses_stream(self, **kwargs) -> Generator[str, None, None]:
        gen, _ = self._post_stream("/responses", {**kwargs, "stream": True})
        for chunk in gen:
            yield chunk

    # Classic chat endpoint
    def chat_completions(self, **kwargs) -> Dict[str, Any]:
        return self._post_json("/chat/completions", kwargs)

    def chat_completions_stream(self, **kwargs) -> Generator[str, None, None]:
        gen, _ = self._post_stream("/chat/completions", {**kwargs, "stream": True})
        for chunk in gen:
            yield chunk

    # Embeddings
    def embeddings(self, **kwargs) -> Dict[str, Any]:
        return self._post_json("/embeddings", kwargs)

    # Images (generate)
    def images_generate(self, **kwargs) -> Dict[str, Any]:
        return self._post_json("/images/generations", kwargs)

    # Audio — Transcriptions (multipart)
    def audio_transcriptions(self, file_path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/audio/transcriptions"
        t0 = _now()
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in kwargs.items()}
            resp = requests.post(url, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files=files, data=data)
        t1 = _now()

        body = None
        text_fallback = None
        try:
            body = resp.json()
        except Exception:
            text_fallback = resp.text

        _store_full_call_json(
            endpoint="/audio/transcriptions",
            method="POST",
            status_code=resp.status_code,
            request_payload={"multipart": {"fields": kwargs, "file": os.path.basename(file_path)}},
            response_json=body,
            response_text_fallback=text_fallback,
            response_headers=resp.headers,
            t_start=t0,
            t_end=t1,
        )

        _emit_log(
            endpoint="/audio/transcriptions", method="POST", status_code=resp.status_code,
            request_id=str(uuid.uuid4()), request_payload={"multipart": {"fields": kwargs, "file": os.path.basename(file_path)}},
            response_json=body, response_headers=resp.headers,
            t_start=t0, t_first_byte=None, t_end=t1,
            error_body=(None if resp.ok else (body or {"non_json_body": text_fallback}))
        )

        if not resp.ok:
            raise RuntimeError(f"OpenAI error {resp.status_code}: {body or text_fallback}")
        return body

    # Audio — Text-to-Speech (JSON)
    def audio_speech(self, **kwargs) -> Dict[str, Any]:
        return self._post_json("/audio/speech", kwargs)

    # Files (multipart)
    def files_upload(self, file_path: str, purpose: str = "fine-tune") -> Dict[str, Any]:
        url = f"{self.base_url}/files"
        t0 = _now()
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"purpose": purpose}
            resp = requests.post(url, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files=files, data=data)
        t1 = _now()

        body = None
        text_fallback = None
        try:
            body = resp.json()
        except Exception:
            text_fallback = resp.text

        _store_full_call_json(
            endpoint="/files",
            method="POST",
            status_code=resp.status_code,
            request_payload={"purpose": purpose, "file": os.path.basename(file_path)},
            response_json=body,
            response_text_fallback=text_fallback,
            response_headers=resp.headers,
            t_start=t0,
            t_end=t1,
        )

        _emit_log(
            endpoint="/files", method="POST", status_code=resp.status_code,
            request_id=str(uuid.uuid4()), request_payload={"purpose": purpose, "file": os.path.basename(file_path)},
            response_json=body, response_headers=resp.headers,
            t_start=t0, t_first_byte=None, t_end=t1,
            error_body=(None if resp.ok else (body or {"non_json_body": text_fallback}))
        )

        if not resp.ok:
            raise RuntimeError(f"OpenAI error {resp.status_code}: {body or text_fallback}")
        return body

    # Batches
    def batches(self, **kwargs) -> Dict[str, Any]:
        return self._post_json("/batches", kwargs)

    # Fine-tuning
    def fine_tuning_jobs(self, **kwargs) -> Dict[str, Any]:
        return self._post_json("/fine_tuning/jobs", kwargs)

# --- Helpers for Responses API formats ---------------------------------------
def text_plain() -> dict:
    # For normal text, you can omit 'text' entirely; this is here for symmetry
    return {}

def text_json() -> dict:
    # JSON mode without a schema (model returns a JSON object)
    return {"format": "json_object"}

def text_schema(name: str, schema_dict: dict, strict: bool = True) -> dict:
    # Structured Outputs (JSON Schema) for Responses API
    return {
        "format": {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema_dict,
                "strict": strict
            }
        }
    }

# --- Demo / Example usage -----------------------------------------------------
if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-REPLACE"):
        raise SystemExit("Set OPENAI_API_KEY first, e.g. export OPENAI_API_KEY=sk-...")

    client = OpenAILoggingClient()

    # 1) Responses API (plain text)
    out = client.responses(
        model="gpt-4.1",
        input=[{"role": "user", "content": "Give me a one-sentence summary of RAG."}],
        temperature=0.3
        # For JSON mode: text=text_json()
        # For schema: text=text_schema("rag_summary", {"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]})
    )
    print("RESPONSES:", out.get("output_text") or out)

    # 2) Chat Completions
    chat = client.chat_completions(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "List two benefits of JSON mode."},
        ],
        temperature=0.2,
        max_tokens=120
    )
    print("CHAT:", chat["choices"][0]["message"]["content"])

    # 3) Embeddings
    emb = client.embeddings(
        model="text-embedding-3-small",
        input=["hello world", "goodnight moon"]
    )
    print("EMBEDDINGS dims:", len(emb["data"][0]["embedding"]))

    # (Optional) 4) Streaming demo if sseclient-py installed:
    if SSEClient:
        print("STREAM (chat): ", end="", flush=True)
        for delta in client.chat_completions_stream(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Stream a haiku about observability."}],
            temperature=0.5,
        ):
            print(delta, end="", flush=True)
        print()
