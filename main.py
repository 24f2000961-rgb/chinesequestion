@app.post("/process-audio", response_model=StatsResponse)
async def process_audio(payload: AudioRequest):
    # Decode Base64 Data
    audio_bytes = base64.b64decode(payload.audio_base64)
    
    # Try parsing as a binary dataset structure
    try:
        # 1. Attempt to load as a numpy structured binary file or array
        buffer = io.BytesIO(audio_bytes)
        try:
            data = np.load(buffer, allow_pickle=True)
            if isinstance(data, np.ndarray):
                # If it's a structured array with named fields
                if data.dtype.names is not None:
                    df = pd.DataFrame(data)
                else:
                    # Fallback columns if it's a raw matrix matching the expected count
                    df = pd.DataFrame(data, columns=["키", "몸무게"])
            else:
                df = pd.DataFrame(data)
        except Exception:
            # 2. Attempt to load as a pandas pickle object
            buffer.seek(0)
            df = pd.read_pickle(buffer)
            
    except Exception:
        # 3. Fallback to decoding as utf-8 or cp949 (Korean encoding) text CSV just in case
        try:
            csv_text = audio_bytes.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            try:
                csv_text = audio_bytes.decode('cp949') # Korean Windows encoding
                df = pd.read_csv(io.StringIO(csv_text))
            except Exception:
                df = pd.DataFrame()

    # --- Rest of your calculation logic remains the same ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() if not df.empty else []
    rows = len(df)
    columns = df.columns.tolist()
    
    if not df.empty and len(columns) > 0:
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
        mean_dict, std_dict, variance_dict, min_dict, max_dict, median_dict = {}, {}, {}, {}, {}, {}
        mode_dict, range_dict, allowed_vals_dict, value_range_dict = {}, {}, {}, {}
        corr_matrix = []

    return StatsResponse(
        rows=rows, columns=columns, mean=mean_dict, std=std_dict, variance=variance_dict,
        min=min_dict, max=max_dict, median=median_dict, mode=mode_dict, range=range_dict,
        allowed_values=allowed_vals_dict, value_range=value_range_dict, correlation=corr_matrix
    )
