import logging
from safe_exec import run_sandboxed
import os
import pandas as pd
import numpy as np
import re
import json
import io
import base64
import time
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

from typing import Tuple
from openai import OpenAI

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from geopy.distance import geodesic
import tldextract

import dateparser
from dateparser.search import search_dates

from joblib import Parallel, delayed

import contextlib
import urllib3
urllib3.disable_warnings()

try:
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    ENGLISH_STOPWORDS = set()

#nltk.download("stopwords")
#english_stops = set(stopwords.words("english"))

ml_friendly_types = [
    "Numerical",
    "Boolean",
    "Categorical",
    "Ordinal",
    "Text",
    "Datetime",
    "GPS Coordinates",
    "Percentage",
    "Currency",
    "Duration / Timedelta",
    "Image URL"
]

try:
    import filetype      
except ImportError:
    filetype = None        

_RE_IMAGE_URL  = re.compile(r'https?://[^\s]+\.(?:jpe?g|png|gif)(?:\?.*)?$', re.I)
_RE_BASE64_IMG = re.compile(r'^data:image\/[^;]+;base64,\s*', re.I)   # allow whitespace
_RE_VIDEO_URL  = re.compile(r'(youtu\.be/|youtube\.com/(watch\?v=|embed/)|\.(mp4|mov|avi|webm)(\?.*)?$)', re.I)
_RE_DOC_URL    = re.compile(r'https?://[^\s]+\.(?:pdf|docx?|xlsx?|csv)(?:\?.*)?$', re.I)
_RE_URL        = re.compile(r'https?://', re.I)
_RE_FILE_PATH  = re.compile(r'^[a-zA-Z]:\\|^(\/[^\/ ]+)+\/[^\/ ]+\.\w+$')
_RE_GPS        = re.compile(r'^-?\d{1,3}\.\d+[,;\s]\s*-?\d{1,3}\.\d+$')
_RE_EMAIL      = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
_RE_PHONE      = re.compile(r'^\+?\d[\d\s\-]{7,}$')
_RE_PERCENT    = re.compile(r'^\d+(\.\d+)?%$')
_RE_HEX_COLOR  = re.compile(r'^#(?:[A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$')
_RE_RGB_COLOR  = re.compile(r'^rgb\(')

BOOL_SET = {"true", "false", "0", "1"}

def _is_binary_image(val) -> bool:
    """Detect raw byte streams OR base-64 encoded images."""
    if filetype is None or pd.isna(val):
        return False
    try:
        if isinstance(val, (bytes, bytearray, memoryview)):
            return filetype.is_image(val)
        if isinstance(val, str) and _RE_BASE64_IMG.match(val):
            _, b64 = val.split(',', 1)
            return filetype.is_image(base64.b64decode(b64[:80]))
    except Exception:
        pass
    return False

def _date_success_ratio(series: pd.Series, sample_n: int = 60) -> float:
    non_null = series.dropna()
    # Guard against an empty or all-null series — nothing to parse.
    if series.empty or non_null.empty:
        return 0.0

    sample   = non_null.sample(min(sample_n, len(non_null)), random_state=42)
    fast     = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
    fast_ok  = fast.notna()

    slow_needed = sample[~fast_ok]
    if slow_needed.empty:
        return 1.0

    slow_ok = slow_needed.apply(lambda x: dateparser.parse(str(x)) is not None
                                or bool(search_dates(str(x))))
    return (fast_ok.sum() + slow_ok.sum()) / len(sample)


def fallback_infer_type_with_llm(col_name, sample_values):
    prompt = f"""
Column Name: {col_name}
Sample Values: {sample_values}

What is the most appropriate data type? 
Choose one from the following: 
["Image Bytes", "Image URL", "Video URL", "Document URL", "General URL", "File Path", 
"GPS Coordinates", "Email Address", "Phone Number", "Percentage", "Currency", "Color Code", 
"JSON / Nested", "Numerical", "Datetime", "Boolean", "Identifier / ID", "Categorical", 
"Ordinal", "Duration / Timedelta", "Mixed / Ambiguous", "Text"]

Reply with only one of the above types.
"""
    try:
        result = call_llm(prompt).strip()
        return result or "Text"
    except Exception as e:
        print(f"LLM Fallback Error: {e}")
        return "Text"


def infer_column_type(col: pd.Series,
                      thresh: float = 0.5,
                      sample_size: int = 100) -> str:
    """
    Fast, dtype-aware heuristic that covers all types you had:
    - Image Bytes (base64/raw) | Image/Video/Document/General URL | File Path
    - GPS | Email | Phone | Percentage | Currency | Color Code | JSON/Nested
    - Numerical | Datetime (robust) | Boolean (incl. string booleans)
    - Identifier/ID | Categorical | Ordinal | Duration | Mixed/Ambiguous | Text
    Heavy checks are sampled; numeric/datetime/bool dtypes exit early.
    """

    if pd.api.types.is_bool_dtype(col):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(col):
        return "Numerical"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "Datetime"

    non_null = col.dropna()
    if non_null.empty:
        return "Null-heavy"

    raw_sample = non_null.sample(min(sample_size, len(non_null)),  
                                random_state=42)

    sample = raw_sample.astype(str).str.strip()                   
    avg_len = sample.str.len().mean()
    has_digits = sample.str.contains(r"\d").mean()  

    is_plausible_date = (avg_len < 40) and (has_digits > 0.3)  


    null_ratio   = col.isna().mean()
    if null_ratio > 0.80:
        return "Null-heavy"

    nunique = col.nunique(dropna=True)
    if nunique <= 1:
        return "Constant / Low Variance"

    # 1) Image bytes / base-64 
    if sample.str.contains("data:image", regex=False).any() or sample.apply(lambda x: isinstance(x, (bytes, bytearray))).any():
        if sample.apply(_is_binary_image).mean() > thresh:
            return "Image Bytes"

    # 2) URL / path-like detectors
    if sample.str.match(_RE_IMAGE_URL).mean() > thresh:  return "Image URL"
    if sample.str.match(_RE_VIDEO_URL).mean() > thresh:  return "Video URL"
    if sample.str.match(_RE_DOC_URL).mean()   > thresh:  return "Document URL"
    if sample.str.match(_RE_URL).mean()       > thresh:  return "General URL"
    if sample.str.match(_RE_FILE_PATH).mean() > thresh:  return "File Path"

    # 3) Structured text detectors
    if sample.str.match(_RE_GPS).mean()     > thresh:  return "GPS Coordinates"
    if sample.str.match(_RE_EMAIL).mean()   > thresh:  return "Email Address"
    if sample.str.match(_RE_PHONE).mean()   > thresh:  return "Phone Number"
    if sample.str.match(_RE_PERCENT).mean() > thresh:  return "Percentage"
    if sample.str.contains(r'(?:USD|EUR|GBP|INR|[$€£₹])\s?\d', case=False).mean() > thresh:
        return "Currency"

    # Color codes
    if (sample.str.match(_RE_HEX_COLOR) | sample.str.match(_RE_RGB_COLOR)).mean() > thresh:
        return "Color Code"

    # JSON / Nested ( objects and arrays)
    if (sample.str.startswith("{") | sample.str.startswith("[")).mean() > thresh:
        return "JSON / Nested"

    # 4) Datetime
    if is_plausible_date and _date_success_ratio(col) > 0.70:
        return "Datetime"

    # 5) Boolean-from-strings (object dtype)
    if nunique <= 2 and sample.str.lower().isin(BOOL_SET).mean() > 0.90:
        return "Boolean"

    # 6) Identifier / ID 
    if nunique == len(non_null):
        return "Identifier / ID"

    # 7) Categorical
    unique_ratio = nunique / max(len(col), 1)
    if unique_ratio < 0.05:
        return "Categorical"

    # 8) Ordinal 
    known_ordinals = {"low", "medium", "high", "rare", "common", "excellent", "poor"}
    if sample.str.lower().isin(known_ordinals).mean() > thresh:
        return "Ordinal"

    # 9) Duration HH:MM[:SS]
    if sample.str.match(r'^\d+:\d{2}(?::\d{2})?$').mean() > thresh:
        return "Duration / Timedelta"

    # 10) Mixed / Ambiguous (heterogeneous Python types in raw sample)
    type_set = {type(x).__name__ for x in raw_sample}
    if len(type_set) > 1:
        return "Mixed / Ambiguous"

    # Fallback – try LLM once more before returning Text
    inferred_by_llm = fallback_infer_type_with_llm(col.name, sample.tolist())

    if inferred_by_llm in {
        "Image Bytes", "Image URL", "Video URL", "Document URL", "General URL", "File Path",
        "GPS Coordinates", "Email Address", "Phone Number", "Percentage", "Currency", "Color Code",
        "JSON / Nested", "Numerical", "Datetime", "Boolean", "Identifier / ID", "Categorical",
        "Ordinal", "Duration / Timedelta", "Mixed / Ambiguous", "Text"
    }:
        return inferred_by_llm
    else:
        return "Text"



def get_cleaning_and_enrichment_suggestions(df: pd.DataFrame) -> dict:
    column_details = []

    for col in df.columns:
        col_data = df[col]
        col_type = infer_column_type(col_data)
        sample_values = col_data.dropna().astype(str).tolist()[:3]

        column_details.append({
            "column_name": col,
            "column_type": col_type,
            "sample_values": sample_values
        })

    prompt = (
        "You are a data analysis assistant.\n"
        "For each of the following columns, suggest:\n"
        "1. A practical cleaning/transformation suggestion (for ML pipelines).\n"
        "2. A useful enrichment/derived feature suggestion.\n\n"
        "Respond in valid JSON ONLY with this format:\n"
        "{\n"
        "  \"cleaning\": {\n"
        "    \"column1\": \"...\",\n"
        "    \"column2\": \"...\"\n"
        "  },\n"
        "  \"enrichment\": {\n"
        "    \"column1\": \"...\",\n"
        "    \"column2\": \"...\"\n"
        "  }\n"
        "}\n\n"
        "Here are the columns:\n"
    )

    for detail in column_details:
        prompt += (
            f"- Column Name: {detail['column_name']}\n"
            f"  Type: {detail['column_type']}\n"
            f"  Sample Values: {detail['sample_values']}\n\n"
        )

    response = call_llm(prompt)

    response = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(response)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        print("⚠️ Failed to parse LLM response:", e)
        print("🔍 Raw response was:", response[:500])
        return {}


def quick_pipeline_score(col_type, miss_pct, uniq_pct, series, top_vals):
    desc = f"{miss_pct:.1f}% missing | {uniq_pct:.1f}% unique"

    if col_type == "Numerical":
        if miss_pct < 5: return f"⭐⭐⭐⭐⭐ — Numeric ({desc})"
        return f"⭐⭐⭐ — Numeric issues ({desc})"

    if col_type in ("Categorical", "Ordinal"):
        example = ', '.join([f"{k} ({v})" for k, v in top_vals.items()])
        if uniq_pct < 5:  return f"⭐⭐⭐⭐ — Low-card ({desc} | {example})"
        return f"⭐⭐⭐ — Categorical ({desc} | {example})"

    if col_type == "Datetime":
        return f"⭐⭐⭐ — Datetime ({desc})"
    if col_type == "Boolean":
        return f"⭐⭐⭐⭐ — Boolean ({desc})"
    if miss_pct > 70:
        return f"⭐ — Too many missing ({miss_pct:.1f}%)"
    return f"⭐ — Other ({desc})"

def get_viz_capability(col_type: str, col_name: str = "") -> str:
    """
    Returns the visualization usability indicator with chart type if supported.
    """
    plot_map = {
        "Numerical": "Box Plot",
        "Categorical": "Bar Chart",
        "Boolean": "Bar Chart",
        "Ordinal": "Histogram",
        "Text": "Word Cloud",
        "Datetime": "Time Series Line Chart",
        "GPS Coordinates": "Map",
        "Percentage": "Histogram",
        "Currency": "Histogram",
        "Color Code": "Swatches",
        "Email Address": "Top Values",
        "Phone Number": "Top Values",
        "Image URL": "Image Viewer",
        "Video URL": "Video Preview",
        "General URL": "LLM Summary",
        "Document URL": "Download Viewer",
        "File Path": "Download Viewer",
    }

    if col_type == "General URL" and col_name.lower() in ["website", "site", "webpage"]:
        return "✅ (LLM Summary)"

    if col_type in plot_map:
        return f"✅ ({plot_map[col_type]})"

    return "❌"


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing_pct = df.isna().mean() * 100
    nunique     = df.nunique(dropna=True)
    total_rows  = len(df)

    def process_column(col_name: str) -> dict:
        series     = df[col_name]
        col_start  = time.perf_counter()          

        t0 = time.perf_counter()
        col_type = infer_column_type(series)
        infer_ms = (time.perf_counter() - t0) * 1_000

        miss      = missing_pct[col_name]
        uniq      = nunique[col_name]
        uniq_pct  = (uniq / total_rows) * 100 if total_rows else 0
        usable_ml = col_type in ml_friendly_types
        usable_vz = get_viz_capability(col_type, col_name)

        if col_type in ("Categorical", "Ordinal"):
            top_vals = series.dropna().head(500).value_counts().head(2).to_dict()
        else:
            top_vals = {}

        t0 = time.perf_counter()
        ml_ready = quick_pipeline_score(col_type, miss, uniq_pct, series, top_vals)
        score_ms = (time.perf_counter() - t0) * 1_000

        total_ms = (time.perf_counter() - col_start) * 1_000
        logger.debug(
            "[TIMING] %-20s infer=%6.1f ms | score=%6.1f ms | total=%6.1f ms",
            col_name, infer_ms, score_ms, total_ms,
        )

        return {
            "Column": col_name,
            "Inferred Type": col_type,
            "Usable for ML": "✅" if usable_ml else "❌",
            "ML Readiness": ml_ready,
            "Usable for Visualization": usable_vz,
            "Elapsed ms": round(total_ms, 1)
        }

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_column)(c) for c in df.columns
    )
    return pd.DataFrame(results)


def _column_outlier_pct(series: pd.Series, iqr_multiplier: float = 1.5):
    """IQR-based outlier rate for a single numeric column, mirroring the
    fence logic in pipeline_logic.remove_outliers. Returns None when the
    column has no spread to compute a fence from."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    Q1 = non_null.quantile(0.25)
    Q3 = non_null.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return 0.0
    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR
    outliers = non_null[(non_null < lower) | (non_null > upper)]
    return float(len(outliers) / len(non_null) * 100)


def generate_quality_report(df: pd.DataFrame, inferred_types_df: pd.DataFrame) -> dict:
    """Aggregate a dataset-level data-quality report from data already
    gathered during inference (``analyze_dataframe``).

    Reuses:
      - the per-column ``Inferred Type`` from ``inferred_types_df`` (so
        null-heavy / constant / mixed-ambiguous columns are already
        detected by ``infer_column_type``),
      - the same IQR-fence logic ``pipeline_logic.remove_outliers`` uses,
        for a per-numeric-column outlier rate.

    Returns a plain dict (JSON-serializable, aside from numpy scalar
    rounding which is cast to native float/int) with:
      - ``overall_score``: 0-100 overall data-quality score.
      - ``total_rows`` / ``total_columns``.
      - ``missing_pct_overall``: mean % missing across all cells.
      - ``duplicate_rows`` / ``duplicate_pct``: fully-duplicated rows.
      - ``columns``: per-column stats + score (list of dicts).
      - ``flagged_columns``: subset of ``columns`` with at least one
        detected issue, for a "what needs attention" view.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_pct_overall = float(df.isna().mean().mean() * 100) if total_cols else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = float(duplicate_rows / total_rows * 100) if total_rows else 0.0

    numeric_cols = set(df.select_dtypes(include=np.number).columns)

    column_rows = []
    flagged_columns = []
    per_col_scores = []

    for _, row in inferred_types_df.iterrows():
        col = row["Column"]
        col_type = row["Inferred Type"]
        if col not in df.columns:
            continue

        miss_pct = float(df[col].isna().mean() * 100)
        uniq_pct = float(df[col].nunique(dropna=True) / total_rows * 100) if total_rows else 0.0
        outlier_pct = _column_outlier_pct(df[col]) if col in numeric_cols else None

        score = 100.0
        issues = []

        if col_type == "Null-heavy" or miss_pct > 50:
            score -= 50
            issues.append("high missingness")
        elif miss_pct > 5:
            score -= 15
            issues.append("some missing values")

        if col_type == "Constant / Low Variance":
            score -= 40
            issues.append("constant / low variance")

        if col_type == "Mixed / Ambiguous":
            score -= 20
            issues.append("mixed / ambiguous types")

        if outlier_pct is not None and outlier_pct > 5:
            score -= min(20.0, outlier_pct)
            issues.append(f"{outlier_pct:.1f}% outlier rows")

        score = max(0.0, min(100.0, score))
        per_col_scores.append(score)

        entry = {
            "Column": col,
            "Inferred Type": col_type,
            "% Missing": round(miss_pct, 1),
            "% Unique": round(uniq_pct, 1),
            "% Outliers": round(outlier_pct, 1) if outlier_pct is not None else None,
            "Score": round(score, 1),
        }
        column_rows.append(entry)
        if issues:
            flagged_columns.append({**entry, "Issues": ", ".join(issues)})

    overall_score = float(np.mean(per_col_scores)) if per_col_scores else 100.0
    # Dataset-level penalty for duplicate rows, on top of the per-column average.
    overall_score = max(0.0, overall_score - min(20.0, duplicate_pct))

    return {
        "overall_score": round(overall_score, 1),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_pct_overall": round(missing_pct_overall, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_pct, 2),
        "columns": column_rows,
        "flagged_columns": flagged_columns,
    }


def custom_cleaning_via_llm(user_instruction: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Calls the LLM with a strict prompt, parses returned code from JSON, executes it on df.

    Args:
        user_instruction (str): User's natural language cleaning instruction.
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        Tuple[pd.DataFrame, str]: Cleaned DataFrame and executed code.
    """
    formatted_df = df.to_csv(index=False)
    prompt = f"""
You are a Python data cleaning assistant.

Your task is to generate Python code that modifies the `df` DataFrame *in-place* based on the user's instruction. You have access to the full dataset below in CSV format. 

Make sure the code:

USER INSTRUCTION:
{user_instruction}

FULL DATAFRAME (CSV FORMAT):
{formatted_df}

PROMPT LENGTH (characters): {len(user_instruction) + len(formatted_df)}
"""

    try:
        llm_response = call_llm(prompt)
        llm_response = llm_response.strip()

        if not llm_response:
            raise ValueError("LLM returned an empty response.")

        cleaned_response = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", llm_response, flags=re.IGNORECASE | re.DOTALL).strip()

        code_str = ""
        try:
            code_data = json.loads(cleaned_response)
            if isinstance(code_data, dict):
                code_str = str(code_data.get("code", "")).strip()
            elif isinstance(code_data, str):
                code_str = code_data.strip()
        except json.JSONDecodeError:
            code_str = cleaned_response

        if not code_str:
            raise ValueError("LLM response did not contain executable code.")

        cleaned_df = run_sandboxed("custom_cleaning", code_str, df.copy())

        return cleaned_df, code_str

    except Exception as e:
        raise RuntimeError(f"Failed to apply LLM cleaning: {e}")

def call_llm(prompt: str, temperature=0.3, max_tokens=2000) -> str:
    import streamlit as st
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except:
        api_key = ""

    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in secrets.toml or environment variables")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    model_name = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.6")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model didn't return a response (it may have run out of output budget) - try a shorter prompt")
    return content.strip()

def execute_plot_code(code: str, df: pd.DataFrame):
    """
    Executes LLM-generated plot code in an isolated subprocess (see
    ``safe_exec.run_sandboxed``) and returns the rendered chart as PNG bytes.
    """
    try:
        return run_sandboxed("plot", code, df)
    except Exception as e:
        raise RuntimeError(f"Error executing visualization code: {e}")
