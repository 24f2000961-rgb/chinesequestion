import base64
import json
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

class StatsResponse(BaseModel):
    rows: int
    columns: List[str]
    mean: Dict[str, float]
    std: Dict[str, float]
    variance: Dict[str, float]
    min: Dict[str, float]
    max: Dict[str, float]
    median: Dict[str, float]
    mode: Dict[str, Any]
    range: Dict[str, float]
    allowed_values: Dict[str, List[Any]]
    value_range: Dict[str, List[float]]
    correlation: List[List[float]]

@app.post("/process-audio", response_model=StatsResponse)
async def process_audio(payload: AudioRequest):
    # Decode Base64 Data
    audio_bytes = base64.b64decode(payload.audio_base64)
    
    # --- UPDATED: Parse base64 string directly as a CSV dataset string ---
    try:
        csv_text = audio_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        # Fallback to an entirely empty DataFrame if it's not a text CSV
        df = pd.DataFrame()

    # Identify numeric columns dynamically
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() if not df.empty else []
    
    # Handle structural properties safely
    rows = len(df)
    columns = df.columns.tolist()
    
    # Compute aggregations or fall back to empty structures
    if not df.empty and len(numeric_cols) > 0:
        mean_dict = df[numeric_cols].mean().to_dict()
        std_dict = df[numeric_cols].std().fillna(0).to_dict()
        variance_dict = df[numeric_cols].var().fillna(0).to_dict()
        min_dict = df[numeric_cols].min().to_dict()
        max_dict = df[numeric_cols].max().to_dict()
        median_dict = df[numeric_cols].median().to_dict()
        
        mode_dict = {col: df[col].mode().dropna().tolist() for col in numeric_cols}
        range_dict = {col: float(max_dict[col] - min_dict[col]) for col in numeric_cols}
        allowed_vals_dict = {col: df[col].dropna().unique().tolist() for col in df.columns}
        value_range_dict = {col: [float(min_dict[col]), float(max_dict[col])] for col in numeric_cols}
        
        corr_matrix = df[numeric_cols].corr().fillna(0).values.tolist()
    else:
        # Strict fallbacks for an empty dataset or zero-column response
        mean_dict = {}
        std_dict = {}
        variance_dict = {}
        min_dict = {}
        max_dict = {}
        median_dict = {}
        mode_dict = {}
        range_dict = {}
        allowed_vals_dict = {}
        value_range_dict = {}
        corr_matrix = []

    return StatsResponse(
        rows=rows,
        columns=columns,
        mean=mean_dict,
        std=std_dict,
        variance=variance_dict,
        min=min_dict,
        max=max_dict,
        median=median_dict,
        mode=mode_dict,
        range=range_dict,
        allowed_values=allowed_vals_dict,
        value_range=value_range_dict,
        correlation=corr_matrix
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
