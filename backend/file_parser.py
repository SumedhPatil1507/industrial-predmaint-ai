"""Universal file parser – CSV, Excel, JSON, Parquet, TSV."""
import pandas as pd
import io
import json
from pathlib import Path


SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".feather"}


def parse_upload(filename: str, content: bytes) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED}")

    buf = io.BytesIO(content)

    if ext == ".csv":
        return pd.read_csv(buf)
    elif ext == ".tsv":
        return pd.read_csv(buf, sep="\t")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(buf)
    elif ext == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data]) if not any(isinstance(v, list) for v in data.values()) \
                else pd.DataFrame(data)
        raise ValueError("JSON must be array or object-of-arrays")
    elif ext == ".parquet":
        return pd.read_parquet(buf)
    elif ext == ".feather":
        return pd.read_feather(buf)

    raise ValueError(f"Cannot parse {ext}")


def validate_columns(df: pd.DataFrame, required: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


def infer_schema(df: pd.DataFrame) -> dict:
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "sample": df.head(3).to_dict(orient="records"),
    }
