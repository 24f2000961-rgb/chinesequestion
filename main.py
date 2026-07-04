import base64
import io
import math
import pickle
import re
import tempfile
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-api")

app = FastAPI()

# Lazy-loaded so the app can still boot even if the whisper model
# takes a while / fails to download on a constrained instance.
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # "base" keeps memory/CPU usage low enough for Render's free tier (512MB RAM).
        # Bump to "small" or "medium" for better Korean accuracy if you're on a paid plan.
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio(raw_bytes: bytes) -> Optional[str]:
    """Write bytes to a temp file (letting ffmpeg/soundfile sniff the real format)
    and run Whisper on it. Returns the transcript text, or None on failure."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as f:
            f.write(raw_bytes)
            f.flush()
            model = get_whisper_model()
            segments, info = model.transcribe(f.name, language="ko")
            text = " ".join(seg.text for seg in segments)
            logger.info("TRANSCRIPT: %s", text)
            return text
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        return None


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


# In-memory store so you can inspect what each request actually contained
# and what Whisper transcribed, via GET /debug/{audio_id}
DEBUG_STORE: Dict[str, Dict[str, Any]] = {}


def parse_transcript_to_dataframe(transcript: str) -> pd.DataFrame:
    """
    Best-effort parser for a spoken-aloud dataset transcript.
    This is a placeholder — refine once we see a real transcript via /debug.
    Attempts a generic strategy: find Korean "label: number" style mentions.
    """
    if not transcript:
        return pd.DataFrame()

    # Look for patterns like "키 160.5" / "키: 160.5" / "키는 160.5"
    pattern = re.compile(r"([가-힣A-Za-z]+)\s*(?:는|은|:)?\s*(-?\d+(?:\.\d+)?)")
    matches = pattern.findall(transcript)
    if not matches:
        return pd.DataFrame()

    # Group sequentially into rows: assume the same set of labels repeats per row
    labels_seen: List[str] = []
    rows: List[Dict[str, float]] = []
    current_row: Dict[str, float] = {}
    for label, value in matches:
        if label in current_row:
            rows.append(current_row)
            current_row = {}
        current_row[label] = float(value)
        if label not in labels_seen:
            labels_seen.append(label)
    if current_row:
        rows.append(current_row)

    return pd.DataFrame(rows)


def _clean(value: Any) -> Any:
    """Convert numpy/pandas scalar types into plain JSON-safe Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return value


def parse_payload_to_dataframe(raw_bytes: bytes) -> pd.DataFrame:
    """
    The "audio_base64" field does not actually contain audio.
    It's a base64-encoded dataset in one of several possible serialisations.
    Try each in turn until one works.
    """
    buffer = io.BytesIO(raw_bytes)

    # 1. Try numpy .npy / .npz
    try:
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            # Take the first array in the archive
            first_key = data.files[0]
            arr = data[first_key]
            if arr.dtype.names is not None:
                return pd.DataFrame(arr)
            return pd.DataFrame(arr)
        if isinstance(data, np.ndarray):
            if data.dtype.names is not None:
                return pd.DataFrame(data)
            return pd.DataFrame(data)
    except Exception:
        pass

    # 2. Try pandas / pickle (DataFrame, dict, or list pickled directly)
    try:
        buffer.seek(0)
        obj = pickle.load(buffer)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return pd.DataFrame(obj)
    except Exception:
        pass

    # 3. Try pandas' own read_pickle (handles some pandas-specific pickling quirks)
    try:
        buffer.seek(0)
        df = pd.read_pickle(buffer)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass

    # 4. Try JSON
    try:
        text = raw_bytes.decode("utf-8")
        return pd.read_json(io.StringIO(text))
    except Exception:
        pass

    # 5. Try CSV, several encodings
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            df = pd.read_csv(io.StringIO(text))
            if not df.empty:
                return df
        except Exception:
            continue

    # 6. Try Excel bytes
    try:
        buffer.seek(0)
        return pd.read_excel(buffer)
    except Exception:
        pass

    # Give up — empty frame
    return pd.DataFrame()


@app.get("/debug/{audio_id}")
async def debug(audio_id: str) -> Dict[str, Any]:
    return DEBUG_STORE.get(audio_id, {"message": "no data captured yet for this audio_id"})


@app.post("/process-audio")
async def process_audio(payload: AudioRequest) -> Dict[str, Any]:
    audio_bytes = base64.b64decode(payload.audio_base64)

    debug_info: Dict[str, Any] = {
        "raw_len": len(audio_bytes),
        "first_16_bytes_hex": audio_bytes[:16].hex(),
    }

    df = parse_payload_to_dataframe(audio_bytes)

    transcript = None
    if df.empty or len(df.columns) == 0:
        # Not a disguised tabular format — try treating it as real audio.
        transcript = transcribe_audio(audio_bytes)
        debug_info["transcript"] = transcript
        if transcript:
            df = parse_transcript_to_dataframe(transcript)

    debug_info["parsed_columns"] = df.columns.tolist() if not df.empty else []
    debug_info["parsed_rows"] = int(len(df))
    DEBUG_STORE[payload.audio_id] = debug_info

    if df.empty or len(df.columns) == 0:
        return {
            "rows": 0,
            "columns": [],
            "mean": {},
            "std": {},
            "variance": {},
            "min": {},
            "max": {},
            "median": {},
            "mode": {},
            "range": {},
            "allowed_values": {},
            "value_range": {},
            "correlation": [],
        }

    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols: List[str] = df.columns.tolist()

    mean_dict = _clean(df[numeric_cols].mean().to_dict())
    std_dict = _clean(df[numeric_cols].std().fillna(0).to_dict())
    variance_dict = _clean(df[numeric_cols].var().fillna(0).to_dict())
    min_dict = _clean(df[numeric_cols].min().to_dict())
    max_dict = _clean(df[numeric_cols].max().to_dict())
    median_dict = _clean(df[numeric_cols].median().to_dict())
    mode_dict = _clean(
        {col: df[col].mode(dropna=True).tolist() for col in numeric_cols}
    )
    range_dict = _clean(
        {col: float(max_dict[col]) - float(min_dict[col]) for col in numeric_cols}
    )
    allowed_values_dict = _clean(
        {col: df[col].dropna().unique().tolist() for col in all_cols}
    )
    value_range_dict = _clean(
        {col: [float(min_dict[col]), float(max_dict[col])] for col in numeric_cols}
    )
    if len(numeric_cols) >= 2:
        correlation = _clean(df[numeric_cols].corr().fillna(0).values.tolist())
    else:
        correlation = []

    return {
        "rows": int(len(df)),
        "columns": all_cols,
        "mean": mean_dict,
        "std": std_dict,
        "variance": variance_dict,
        "min": min_dict,
        "max": max_dict,
        "median": median_dict,
        "mode": mode_dict,
        "range": range_dict,
        "allowed_values": allowed_values_dict,
        "value_range": value_range_dict,
        "correlation": correlation,
    }


@app.get("/")
async def root():
    return {"status": "ok"}
