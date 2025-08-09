from __future__ import annotations

import base64
import io
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from cleanse_tool import (
    read_dataset,     # expects an UploadFile-like object with .filename and .file
    profile_data,
    cleanse_dataset,
    validate_data,
)

# ---------- Models ----------

class DataFile(BaseModel):
    """Incoming base64-encoded file."""
    filename: str = Field(..., description="Name of the file, including extension.")
    content: str = Field(..., description="Base64-encoded file bytes.")

class CleanOptions(BaseModel):
    """Options for /clean."""
    cutoff: float = Field(
        0.9, ge=0.0, le=1.0,
        description="Similarity threshold for typo correction (0.0–1.0). Higher = stricter."
    )
    fill_missing: bool = Field(False, description="Fill missing values after cleaning.")

# ---------- App ----------

app = FastAPI(title="Cleanse API", description="AI-inspired data cleaning service")

# CORS so your website frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO: replace "*" with ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------

def decode_datafile(datafile: DataFile) -> pd.DataFrame:
    """Decode a base64 DataFile -> DataFrame, using an UploadFile-like object."""
    try:
        file_bytes = base64.b64decode(datafile.content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 content: {exc}")

    # Minimal UploadFile-like object expected by read_dataset()
    pseudo_upload = type(
        "PseudoUploadFile",
        (),
        {"filename": datafile.filename, "file": io.BytesIO(file_bytes)},
    )()

    try:
        df = read_dataset(pseudo_upload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return df

# ---------- Routes ----------

@app.get("/")
def root():
    return {"ok": True, "message": "Cleanse API running. Use /docs, /profile, /clean, /validate, /health."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/profile")
async def profile_endpoint(data: DataFile) -> JSONResponse:
    df = decode_datafile(data)
    return JSONResponse(content=profile_data(df))

@app.post("/clean")
async def clean_endpoint(data: DataFile, options: Optional[CleanOptions] = None) -> JSONResponse:
    df = decode_datafile(data)
    opts = options or CleanOptions()
    cleaned = cleanse_dataset(
        df,
        correct_typo_cutoff=opts.cutoff,
        fill_missing=opts.fill_missing,
    )

    # Return cleaned CSV as base64
    buf = io.StringIO()
    cleaned.to_csv(buf, index=False)
    csv_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("ascii")
    new_filename = data.filename.rsplit(".", 1)[0] + "_cleaned.csv"

    payload = {
        "filename": new_filename,
        "content": csv_b64,
        "profile": profile_data(cleaned),
        "validation": validate_data(cleaned),
    }
    return JSONResponse(content=payload)

@app.post("/validate")
async def validate_endpoint(data: DataFile) -> JSONResponse:
    df = decode_datafile(data)
    return JSONResponse(content=validate_data(df))
