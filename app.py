"""
app.py — Smart CSV Toolkit
===========================
Main Streamlit application with three tabs:

    Tab 1 – CSV Cleaner:        Upload a CSV, select pipeline steps (with
                                configurable parameters), request AI-generated
                                cleaning suggestions, and download the result.

    Tab 2 – Metadata Inspector: Infer column types, visualise each column,
                                run a custom LLM cleaning instruction, and
                                explore an interactive decision-tree for
                                advanced cleaning actions.

    Tab 3 – Memory Browser:     Search or browse past LLM cleaning
                                suggestions stored in the ChromaDB vector
                                store (RAG memory).

Below the tabs there is also a panel for querying a custom-trained local LLM
via REST API, and a feedback widget.
"""

import os
import socket
import ipaddress
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import re
import io
import uuid
import json
import tempfile
import contextlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
import requests
import dtale
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from PIL import Image
from bs4 import BeautifulSoup
from wordcloud import WordCloud

from openai import OpenAI
from streamlit_agraph import agraph, Config
import streamlit.components.v1 as components

from pipeline_logic import run_pipeline
from metadata_inference import analyze_dataframe, custom_cleaning_via_llm, execute_plot_code
from cleaningDecisionTree import render_pyvis_tree, render_agraph_tree
from pyvis.network import Network
from DB.log_to_db import log_session, log_file, log_event
from datetime import datetime
import time
import urllib3
from vector_store.store import query_suggestions, add_suggestion, get_all_suggestions, count_suggestions

urllib3.disable_warnings()

########################################################################################
############################CSV Cleaning with AI Suggestions############################
########################################################################################

########################################################################################
######################################Upload limits######################################
########################################################################################

# Application-level guardrails, well below Streamlit's default 200MB
# per-file uploader cap, to avoid exhausting memory on the deployed instance.
MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS = 500_000


def check_upload_size(size_bytes: int, label: str = "file") -> bool:
    """
    Return True if size_bytes is within MAX_UPLOAD_BYTES, otherwise show an
    st.error and return False.
    """
    if size_bytes is not None and size_bytes > MAX_UPLOAD_BYTES:
        st.error(
            f"❌ {label} is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"The maximum allowed size is {MAX_UPLOAD_MB} MB."
        )
        return False
    return True


def check_row_count(df: pd.DataFrame, label: str = "file") -> bool:
    """
    Return True if df has at most MAX_ROWS rows, otherwise show an st.error
    and return False.
    """
    if len(df) > MAX_ROWS:
        st.error(
            f"❌ {label} has too many rows ({len(df):,}). "
            f"The maximum allowed is {MAX_ROWS:,} rows."
        )
        return False
    return True


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with any string cell that starts with '=', '+', '-',
    '@', tab, or carriage return prefixed with a single quote, so spreadsheet
    apps (Excel/LibreOffice/Google Sheets) treat it as literal text instead
    of a formula. Mitigates CSV/DDE formula injection on export.
    """
    def esc(v):
        if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
            return "'" + v
        return v
    return df.applymap(esc)


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Send a prompt to the OpenRouter-hosted model and return the text response.

    The API key is read from ``st.secrets["OPENROUTER_API_KEY"]`` or the
    ``OPENROUTER_API_KEY`` environment variable; if it is absent
    the function raises immediately so the caller can surface a clear error to
    the user rather than a cryptic auth failure. The model used is
    configurable via the ``OPENROUTER_MODEL`` environment variable and
    defaults to ``z-ai/glm-4.6``.

    Args:
        prompt: Full prompt string to send to the model.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
            Defaults to 0.3 for consistent data-cleaning outputs.
        max_tokens: Maximum number of output tokens.  Defaults to 700.

    Returns:
        Stripped text response from the model.

    Raises:
        ValueError: If ``OPENROUTER_API_KEY`` is not configured.
    """
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
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
    # Reasoning models (GLM-4.6 included) can spend the whole max_tokens
    # budget on hidden reasoning and return content: None for longer/more
    # complex prompts instead of the expected text.
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model didn't return a response (it may have run out of output budget) - try a shorter prompt")
    return content.strip()


def get_cleaning_code_from_llm(instruction: str, df: pd.DataFrame) -> str:
    """Ask the LLM to generate raw Python code that cleans ``df`` in-place.

    Sends the natural-language instruction together with a 2-row sample of the
    DataFrame so the model has concrete column/type context.  The returned
    string is raw Python — no markdown fences or explanations.

    Args:
        instruction: User-written cleaning instruction, e.g.
            ``"Drop rows where 'Age' is negative"``.
        df: The DataFrame to be cleaned (only the first 2 rows are sent).

    Returns:
        Raw Python code string that modifies a variable named ``df``.
    """
    example_data = df.head(2).to_dict(orient="records")
    prompt = f"""
    You are a Python data cleaning assistant.

    INSTRUCTION: Convert the following user instruction into Python code that modifies the `df` DataFrame in-place.

    RULES:
    - DO NOT wrap your answer in markdown or backticks
    - DO NOT return explanation or comments
    - ONLY return raw Python code

    Instruction:
    \"\"\"{instruction}\"\"\"

    Sample Data:
    {json.dumps(example_data, indent=2)}
    """
    return call_llm(prompt, temperature=0.2, max_tokens=500)


def fetch_llm_suggestions(df: pd.DataFrame, config: dict) -> list:
    """Request 5 actionable cleaning suggestions for the given DataFrame.

    Extracts the names and descriptions of already-configured pipeline steps
    from ``config`` and explicitly excludes them from the suggestions to avoid
    redundancy.

    Args:
        df: The uploaded DataFrame (first 2 rows are sent to the LLM).
        config: The parsed ``table_steps.json`` configuration dict.

    Returns:
        A list of 5 suggestion strings.
    """
    # Collect all step names/descriptions already in the pipeline config so
    # the LLM doesn't suggest things we already support.
    excluded_keywords = []
    for section, step_list in config.get("processing", {}).items():
        for step in step_list:
            excluded_keywords.append(step["name"].lower())
            excluded_keywords.append(step["description"].lower())

    example_data = df.head(2).to_dict(orient="records")

    prompt = f"""
        You are a smart data cleaning assistant.

        Your job is to suggest 5 **new**, **non-redundant**, and **directly executable** data cleaning actions based on the uploaded CSV file.

        IMPORTANT:
        - ONLY suggest actions that can be applied using the available data — no vague, manual, or unverifiable suggestions.
        - Do NOT suggest anything already covered in: {excluded_keywords}
        - Do NOT include ideas like "flag suspicious data" or "verify with external sources"

        Format:
        - Return exactly 5 short, actionable suggestions
        - Put each suggestion on its own line
        - Do NOT include numbering, JSON, markdown, explanations, or extra text

        Example output:
        Convert 'DOB' column to datetime format
        Drop columns with more than 50% missing values
        Normalize numeric columns to a 0-1 range
        Trim whitespace in string columns
        One-hot encode categorical columns

        Here is a preview of the data (first 2 rows):
        {json.dumps(example_data, indent=2)}

        Return ONLY the 5 suggestions.
    """

    llm_output = call_llm(prompt)
    cleaned_output = re.sub(r"^```(?:json)?\s*|\s*```$", "", llm_output.strip(), flags=re.IGNORECASE | re.DOTALL)

    try:
        parsed_output = json.loads(cleaned_output)
        if isinstance(parsed_output, list):
            suggestions = [str(item).strip() for item in parsed_output if str(item).strip()]
            if suggestions:
                return suggestions[:5]
    except json.JSONDecodeError:
        pass

    suggestions = []
    for line in cleaned_output.splitlines():
        suggestion = re.sub(r"^[-*•]\s*", "", line.strip())
        suggestion = re.sub(r"^\d+[.)]\s*", "", suggestion)
        suggestion = suggestion.strip().strip('"').strip("'")
        if suggestion:
            suggestions.append(suggestion)

    if not suggestions:
        raise ValueError("LLM did not return parseable suggestions")

    return suggestions[:5]


with open("table_steps.json", "r") as f:
    config = json.load(f)

st.set_page_config(page_title="Smart CSV Toolkit", layout="wide")
st.title("Smart CSV Toolkit")
#tab1, tab2 = st.tabs(["🧼 CSV Cleaner", "🧠 Metadata Inspector"])
tab1, tab2, tab3 = st.tabs(["🧼 CSV Cleaner", "🧠 Metadata Inspector", "📚 Memory Browser"])

with tab1:
    st.header("CSV Cleaning with AI Suggestions")
    st.header("Step 1: Upload CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        if not check_upload_size(getattr(uploaded_file, "size", None), label="Uploaded file"):
            st.stop()
        df = pd.read_csv(uploaded_file)
        if not check_row_count(df, label="Uploaded file"):
            st.stop()

        # Reset stale results from a previously uploaded file. Without this,
        # replacing the uploaded file left st.session_state.cleaned_df (and
        # the LLM suggestion state) pointing at the OLD file's cleaned
        # output — the preview above showed the new file, but "Download
        # Cleaned CSV" would still serve the previous file's result until
        # "Run Cleaning Pipeline" was clicked again.
        file_fingerprint = (uploaded_file.name, getattr(uploaded_file, "size", None))
        if st.session_state.get("_tab1_file_fingerprint") != file_fingerprint:
            st.session_state["_tab1_file_fingerprint"] = file_fingerprint
            st.session_state.pop("cleaned_df", None)
            st.session_state.llm_suggestions = []
            st.session_state.suggested_once = False
            st.session_state.clicked_llm = False
            st.session_state.selected_suggestion = ""
            st.session_state.custom_suggestion = ""

        st.subheader("Preview of Uploaded Data")
        num_rows = st.slider("Rows to display", min_value=5, max_value=len(df), value=10, key="cleaner_preview_rows")
        st.dataframe(df.head(num_rows), use_container_width=True)

        st.header("Step 2: Select and Configure Processing Steps")
        steps = []
        sections = config.get("processing", {})

        # df is guaranteed to exist here — it was assigned from pd.read_csv above.
        categorical_columns = [
            col for col in df.columns
            if df[col].dtype == "object" or df[col].nunique() < 20
        ]

        for section, step_list in sections.items():
            st.markdown(f"###  {section.replace('_', ' ').title()}")
            for step in step_list:
                step_key = f"step_{step['name']}"
                if st.checkbox(f"{step['name']} – {step['description']}", key=step_key):
                    params = {}
                    for opt in step.get("options", []):
                        param_key = f"{step['name']}_{opt['name']}"
                        dtype = opt.get("data_type")
                        default = opt.get("value")
                        choices = opt.get("options", [])

                        if opt["name"] == "target_column" and dtype == "select":
                            if categorical_columns:
                                params[opt["name"]] = st.selectbox(
                                    label=f"Select target column (categorical) for {step['name']}",
                                    options=categorical_columns,
                                    key=param_key
                                )
                            else:
                                st.warning("No categorical columns found in the dataset to use as target.")
                                params[opt["name"]] = None

                        elif dtype == "int":
                            if choices:
                                params[opt["name"]] = st.selectbox(param_key, choices, index=choices.index(default), key=param_key)
                            else:
                                params[opt["name"]] = st.number_input(param_key, value=int(default), key=param_key)

                        elif dtype == "float":
                            if choices:
                                params[opt["name"]] = st.selectbox(param_key, choices, index=choices.index(default), key=param_key)
                            else:
                                params[opt["name"]] = st.number_input(param_key, value=float(default), key=param_key)

                        elif dtype == "str":
                            if choices:
                                params[opt["name"]] = st.selectbox(param_key, choices, index=choices.index(default), key=param_key)
                            else:
                                params[opt["name"]] = st.text_input(param_key, value=default, key=param_key)

                    steps.append({"name": step["name"], "params": params})


        st.header("AI Cleaning Suggestions using LLM")

        if "llm_suggestions" not in st.session_state:
            st.session_state.llm_suggestions = []
        if "suggested_once" not in st.session_state:
            st.session_state.suggested_once = False
        if "clicked_llm" not in st.session_state:
            st.session_state.clicked_llm = False
        if "selected_suggestion" not in st.session_state:
            st.session_state.selected_suggestion = ""
        if "custom_suggestion" not in st.session_state:
            st.session_state.custom_suggestion = ""

        button_label = "Regenerate Suggestions" if st.session_state.suggested_once else "Get Smart LLM Suggestions"
        if st.button(button_label) and uploaded_file:
            try:
                st.session_state.llm_suggestions = fetch_llm_suggestions(df, config)
                st.session_state.clicked_llm = True
                st.session_state.suggested_once = True
                st.session_state.selected_suggestion = ""
                st.session_state.custom_suggestion = ""
            except Exception as e:
                st.error(str(f"Failed to fetch suggestions: {e}"))

        selected_radio = None
        if st.session_state.clicked_llm and st.session_state.llm_suggestions:
            st.subheader("Suggested Improvements")
            suggestion_texts = st.session_state.llm_suggestions

            if suggestion_texts:
                selected_radio = st.radio(
                    "Select one suggestion to apply:",
                    suggestion_texts,
                    key="radio_selection"
                )

        if selected_radio and not st.session_state.custom_suggestion:
            st.session_state.selected_suggestion = selected_radio

        st.markdown("### Custom LLM Operation")
        custom_input = st.text_input(
            "Describe your own cleaning step (e.g., 'Drop column City')",
            value=st.session_state.custom_suggestion,
            key="custom_input"
        )

        if custom_input.strip() and custom_input != st.session_state.custom_suggestion:
            st.session_state.custom_suggestion = custom_input
            st.session_state.selected_suggestion = ""

        final_llm_operation = st.session_state.custom_suggestion or st.session_state.selected_suggestion

        st.markdown(f"**Final Cleaning Instruction:** `{final_llm_operation}`")


        with st.container():
            st.header("Step 3: Run and Download")

            if st.button("Run Cleaning Pipeline"):
                final_instruction = st.session_state.custom_suggestion or st.session_state.selected_suggestion
                if final_instruction:
                    try:
                        code = get_cleaning_code_from_llm(final_instruction, df)
                        st.warning("⚠️ This is the Python code automatically generated and executed by the LLM to perform the data cleaning. Please review it carefully to ensure correctness.")
                        st.code(code, language="python")
                        steps.append({
                            "name": "LLM Data Cleaning",
                            "params": {
                                "code": code
                            }
                        })
                    except Exception as e:
                        st.error(str(f"Failed to get code from LLM: {e}"))
                        st.stop()

                try:
                    cleaned_df = run_pipeline(df.copy(), steps)
                    st.session_state.cleaned_df = cleaned_df 
                    st.success("✅ Cleaning complete!")

                except (ValueError, RuntimeError) as e:
                    st.warning(f"Some steps were skipped or failed:\n\n{e}")
                    st.stop()

            if "cleaned_df" in st.session_state:
                cleaned_df = st.session_state.cleaned_df

                with st.expander("Preview of Cleaned Data", expanded=True):
                    num_rows_output = st.slider(
                        "Rows to display (cleaned data)",
                        min_value=5,
                        max_value=len(cleaned_df),
                        value=10,
                        key="cleaned_rows"
                    )
                    visible_rows = cleaned_df.head(num_rows_output)
                    row_height = 35
                    min_height = 150
                    max_height = 300
                    st.dataframe(visible_rows, use_container_width=True, height=min(max(len(visible_rows) * row_height, min_height), max_height))

                csv = sanitize_for_csv(cleaned_df).to_csv(index=False).encode("utf-8")
                st.download_button("Download Cleaned CSV", data=csv, file_name="cleaned_output.csv", mime="text/csv")

    

##################################################################################################
##################################Metadata Inference & Usability##################################
##################################################################################################

def is_safe_url(url: str) -> bool:
    """
    Validate that a URL is safe to fetch server-side: scheme must be
    http/https, and the resolved hostname must not point at a private,
    loopback, link-local, or otherwise reserved IP range. This mitigates
    SSRF via URLs embedded in uploaded CSV cell content (e.g. cloud
    metadata endpoints or internal services).
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve all addresses for the hostname and reject if any of them
        # are private/reserved, to guard against DNS rebinding as well.
        addr_infos = socket.getaddrinfo(hostname, None)
        if not addr_infos:
            return False
        for info in addr_infos:
            ip_str = info[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False
        return True
    except Exception:
        return False


def render_image_urls(urls):
    st.markdown("### 🖼️ Image Previews")
    for url in urls:
        try:
            if not is_safe_url(url):
                st.warning(f"Blocked unsafe or non-public URL: {url}")
                continue
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=url, use_column_width=True)
        except:
            st.warning(f"Could not load image: {url}")

def render_file_links(urls):
    st.markdown("### 📎 File Links")
    for url in urls:
        st.markdown(f"[📥 Download File]({url})", unsafe_allow_html=True)

def render_video_urls(urls):
    st.markdown("### 🎬 Video Previews")
    for url in urls:
        embed_url = None

        if "youtube.com/embed/" in url:
            embed_url = url

        elif "youtube.com/watch?v=" in url:
            video_id = url.split("watch?v=")[-1].split("&")[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        elif url.lower().endswith((".mp4", ".webm", ".mov")):
            embed_url = url

        if embed_url:
            st.video(embed_url)
            st.caption(f"[🔗 Watch on YouTube]({url})")
        else:
            st.warning(f"⚠️ Unsupported video format: {url}")

def fetch_and_summarize_url(url: str, instruction: str, llm_func) -> str:
    try:
        if not is_safe_url(url):
            return f"❌ Blocked unsafe or non-public URL: {url}"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)[:1500]

        full_prompt = (
            f"You are a helpful assistant. The user wants to perform the following task on this web page:\n\n"
            f"URL: {url}\n\n"
            f"Extracted Content:\n{text}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Please analyze or summarize accordingly."
        )

        return llm_func(full_prompt)
    except Exception as e:
        return f"❌ Failed to fetch or process the URL: {e}"


def render_column_visualization(df: pd.DataFrame, col: str, inferred_type: str):
    with st.expander(f"Visualization for `{col}` ({inferred_type})", expanded=True):
        try:
            if inferred_type == "Categorical":
                top_k = df[col].value_counts().nlargest(10).reset_index()
                top_k.columns = [col, 'count']
                fig = px.bar(top_k, x=col, y='count')
                st.plotly_chart(fig)

            elif inferred_type == "Text":
                text_data = " ".join(df[col].dropna().astype(str)).lower()
                if text_data.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("Text column is empty or invalid for wordcloud.")

            elif inferred_type == "Numerical":
                fig = px.box(df, y=col)
                st.plotly_chart(fig)
                if df.select_dtypes(include="number").shape[1] > 1:
                    st.markdown("**Correlation Heatmap**")
                    corr = df.select_dtypes(include="number").corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

            elif inferred_type == "Datetime":
                # Parse into a local Series for charting only — do NOT assign
                # back into df[col]. df is the caller's live DataFrame (e.g.
                # st.session_state.final_df), so writing here would silently
                # convert the column's dtype (and coerce unparseable values
                # to NaT, i.e. drop data) as a side effect of just viewing a
                # chart, persisting even if the user never asked to clean it.
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
                timeline = parsed_dates.dt.date.value_counts().sort_index().reset_index()
                timeline.columns = [col, 'Count']
                fig = px.line(timeline, x=col, y='Count')
                st.plotly_chart(fig)

            elif inferred_type == "GPS Coordinates":
                # Same rule as above: split into a standalone DataFrame
                # instead of writing 'lat'/'lon' columns into the caller's
                # df. Split on the same delimiter set the type-inference
                # regex accepts (comma, semicolon, or whitespace) rather
                # than comma only, so semicolon/space-separated coordinates
                # don't fail here.
                coords = df[col].dropna().astype(str).str.split(r'[,;\s]+', expand=True)
                if coords.shape[1] >= 2:
                    coords = coords.iloc[:, :2].astype(float)
                    coords.columns = ['lat', 'lon']
                    st.map(coords)
                else:
                    st.warning("Could not split GPS coordinates into lat/lon.")

            elif inferred_type in ["Boolean", "Ordinal", "Percentage", "Currency"]:
                data = df[col].dropna()
                if inferred_type in ["Currency", "Percentage"]:
                    data = data.replace(r'[^\d\.]', '', regex=True).astype(float)
                fig = px.histogram(data, x=col)
                st.plotly_chart(fig)

            elif inferred_type == "Color Code":
                unique_colors = df[col].dropna().unique()[:20]
                st.markdown("**Color Swatches**")
                st.write("".join([
                    f"<div style='display:inline-block;width:50px;height:20px;background-color:{color};margin:2px;border:1px solid #000'></div>"
                    for color in unique_colors]), unsafe_allow_html=True)

            elif inferred_type in ["Email Address", "Phone Number"]:
                top_vals = df[col].value_counts().nlargest(10).reset_index()
                top_vals.columns = [col, 'count']
                fig = px.bar(top_vals, x=col, y='count')
                st.plotly_chart(fig)

            elif inferred_type == "Identifier / ID":
                st.info("Column identified as unique ID – typically not useful for visualization.")

            elif inferred_type in ["Null-heavy", "Constant / Low Variance"]:
                st.warning(f"Column classified as `{inferred_type}` – not suitable for meaningful visualization.")

            elif inferred_type in ["Image URL", "Video URL", "Document URL", "General URL", "File Path"]:
                urls = df[col].dropna().unique().tolist()

                if inferred_type == "Image URL":
                    with st.container():
                        st.markdown("<div style='max-height:400px; overflow-y:auto;'>", unsafe_allow_html=True)
                        render_image_urls(urls)
                        st.markdown("</div>", unsafe_allow_html=True)

                elif inferred_type == "Video URL":
                    urls = df[col].dropna().unique().tolist()
                    render_video_urls(urls)

                elif inferred_type == "General URL":
                    st.markdown("### 🌐 General URL Processor")

                    selected_url = st.selectbox("Select a URL to summarize", urls, key=f"{col}_url_select")
                    user_prompt = st.text_area("💬 What would you like to know or extract?", placeholder="e.g., Summarize the article, extract key points...", key=f"{col}_prompt_input")

                    if st.button("Analyze Webpage", key=f"{col}_analyze_button"):
                        if selected_url and user_prompt.strip():
                            with st.spinner("⏳ Fetching and analyzing..."):
                                result = fetch_and_summarize_url(selected_url, user_prompt, call_llm)

                            st.success("Done")
                            st.markdown("### Result:")
                            st.write(result)
                        else:
                            st.warning("⚠️ Please select a URL and provide an instruction.")


                elif inferred_type in ["Document URL", "File Path"]:
                    with st.container():
                        st.markdown("<div style='max-height:400px; overflow-y:auto;'>", unsafe_allow_html=True)
                        render_file_links(urls)
                        st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.write(urls) 

        except Exception as e:
            st.error(str(f"Error visualizing `{col}`: {str(e)}"))

def multi_csv_merge_ui(max_files: int = 5):
    """
    Streamlit helper that
      • lets the user upload up to `max_files` CSVs,
      • if 1 file   → shows a "Submit" button and stores it as final_df,
      • if >1 file  → lets the user configure joins and shows a
                      "Submit and Merge Files" button,
      • clears st.session_state.final_df when the uploader is emptied.
    """

    uploaded_files = st.file_uploader(
        f"Upload up to {max_files} CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_csv"
    )

    if not uploaded_files:
        st.session_state.pop("final_df", None)
        st.session_state.pop("final_df_version", None)
        return

    if len(uploaded_files) > max_files:
        st.warning(f"Please upload at most {max_files} files.")
        return

    # Per-file and aggregate size guard, since up to `max_files` files can be
    # uploaded and read into memory in a single rerun.
    sizes = [getattr(f, "size", 0) or 0 for f in uploaded_files]
    for f, size in zip(uploaded_files, sizes):
        if not check_upload_size(size, label=f"File '{f.name}'"):
            return
    total_size = sum(sizes)
    if total_size > MAX_UPLOAD_BYTES:
        st.error(
            f"❌ Combined size of uploaded files is too large "
            f"({total_size / (1024 * 1024):.1f} MB). The maximum allowed "
            f"combined size is {MAX_UPLOAD_MB} MB."
        )
        return

    dataframes  = [pd.read_csv(f) for f in uploaded_files]
    file_names  = [f.name for f in uploaded_files]

    total_rows = 0
    for name, d in zip(file_names, dataframes):
        if not check_row_count(d, label=f"File '{name}'"):
            return
        total_rows += len(d)
    if total_rows > MAX_ROWS:
        st.error(
            f"❌ Combined row count of uploaded files is too large "
            f"({total_rows:,}). The maximum allowed combined total is "
            f"{MAX_ROWS:,} rows."
        )
        return

    if len(dataframes) == 1:
        if st.button("Submit"):
            st.session_state.final_df = dataframes[0]
            # Bump the version marker so tab2 knows to reset its derived
            # working state (cleaned copy, metadata, decision tree, etc.)
            # instead of carrying over stale state from a previously
            # uploaded file.
            st.session_state.final_df_version = uuid4().hex
            st.success("File loaded!")
        return

    st.subheader("Join configuration")
    join_config = []

    for i in range(len(dataframes) - 1):
        st.markdown(f"### Join {i+1}: `{file_names[i]}` ⨝ `{file_names[i+1]}`")

        left_col = st.selectbox(
            f"Column from **{file_names[i]}**",
            dataframes[i].columns.tolist(),
            key=f"left_col_{i}"
        )

        right_col = st.selectbox(
            f"Column from **{file_names[i+1]}**",
            dataframes[i+1].columns.tolist(),
            key=f"right_col_{i}"
        )

        join_type = st.selectbox(
            "Join type",
            ["inner", "outer", "left", "right"],
            key=f"join_type_{i}"
        )

        join_config.append({
            "left_df": i,
            "right_df": i + 1,
            "left_on": left_col,
            "right_on": right_col,
            "how": join_type
        })

    if st.button("Submit and Merge Files"):
        merged = dataframes[0]
        for cfg in join_config:
            merged = pd.merge(
                merged,
                dataframes[cfg["right_df"]],
                how=cfg["how"],
                left_on=cfg["left_on"],
                right_on=cfg["right_on"]
            )
        st.session_state.final_df = merged
        st.session_state.final_df_version = uuid4().hex
        st.success("Files successfully merged!")


###############################################################################
# Tab 2 – Metadata Inference & Usability
###############################################################################

if "expanded_columns" not in st.session_state:
    st.session_state.expanded_columns = set()

def _push_history(prev_df, limit=10):
    """Snapshot the working DataFrame so an LLM cleaning step can be undone."""
    hist = st.session_state.setdefault("df_history", [])
    hist.append(prev_df.copy())
    del hist[:-limit]


with tab2:
    st.header("Metadata Inference & Usability")
    multi_csv_merge_ui()

    if "final_df" in st.session_state:
        df = st.session_state.final_df

        # ⏺️ Reset all state derived from the previously uploaded/merged
        # file whenever a new one replaces it (final_df_version changes).
        # Without this, uploading a new file over an old one left
        # st.session_state.df (the Custom-Cleaning-via-LLM working copy),
        # metadata_df, and the decision-tree state pointing at the OLD
        # file's data — so every subsequent tab2 action silently kept
        # operating on stale data even though the preview showed the new
        # file, and column lookups (e.g. metadata_df["Column"]) could
        # KeyError against the new file's different columns.
        current_version = st.session_state.get("final_df_version")
        if st.session_state.get("_final_df_seen_version") != current_version:
            st.session_state["_final_df_seen_version"] = current_version
            st.session_state.df = df.copy()
            st.session_state.df_history = []
            st.session_state.pop("metadata_df", None)
            st.session_state["show_tree"] = False
            st.session_state["executed_actions"] = set()
            st.session_state["agraph_tree_data"] = None
            st.session_state["agraph_leaf_nodes"] = []
            st.session_state["agraph_col"] = ""
            st.session_state["agraph_parent_map"] = {}
            st.session_state["last_executed_click"] = ""
            st.session_state["last_executed_code"] = ""

        # ⏺️ Session setup for logging
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        log_session(st.session_state.session_id)

        # ⏺️ Save uploaded file to audit directory — once per distinct upload.
        # Streamlit reruns this whole script on every widget interaction, so
        # without this guard the same file got written to disk again on every
        # slider drag / checkbox toggle / keystroke, filling the disk within
        # minutes of routine use (issue #10).
        if st.session_state.get("_audit_saved_version") != current_version:
            st.session_state["_audit_saved_version"] = current_version

            AUDIT_DIR = Path(__file__).resolve().parent / "DB" / "auditCSVFiles"
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            filename = f"uploaded_{timestamp}_{st.session_state.session_id[:8]}.csv"
            file_path = AUDIT_DIR / filename

            df.to_csv(file_path, index=False)

            # ⏺️ Log file save and event
            log_file(st.session_state.session_id, filename, str(file_path))
            log_event(st.session_state.session_id, "file_upload", f"Saved CSV to {file_path}")

        st.subheader("Preview of Uploaded Data")
        num_rows = st.slider("Rows to display", min_value=5, max_value=len(df), value=10, key="metadata_preview_rows")
        st.dataframe(df.head(num_rows), use_container_width=True)

        if st.button("🔍 Run Inference on Columns"):
            with st.spinner("Inferring column types and suggestions..."):
                st.session_state.metadata_df = analyze_dataframe(df)
                log_event(st.session_state.session_id, "inference_triggered", "User ran column inference")

        if "metadata_df" in st.session_state:
            metadata_df = st.session_state.metadata_df

            st.subheader("Column Type Inference")
            st.dataframe(metadata_df)

            st.subheader("Basic Suggested Visualizations")
            st.markdown("### Choose columns to visualize")
            
            instance = dtale.show(df, open_browser=False)
            d_url = instance._main_url 

            st.markdown(f"🔗 [Open D-Tale Visualization]({d_url})")

            selected_cols = st.multiselect(
                "Select columns to show visualizations for",
                options=list(metadata_df["Column"]),
                default=list(metadata_df["Column"])[:0],  
                key="selected_columns"
            )

            for col in selected_cols:
                inferred_type = metadata_df[metadata_df["Column"] == col]["Inferred Type"].values[0]
                render_column_visualization(df, col, inferred_type)
                log_event(st.session_state.session_id, "column_visualized", f"{col} ({inferred_type})")

################################### Custom Visualization Via LLM ###################################
            st.subheader("🧪 Custom Visualization via LLM")

            user_viz_prompt = st.text_area(
                "Describe your custom visualization (e.g., 'Plot a bar chart of category counts')",
                height=100,
                key="custom_viz_prompt"
            )

            if st.button("✨ Generate and Render Plot with LLM"):
                if not user_viz_prompt.strip():
                    st.warning("Please enter a visualization instruction.")
                else:
                    log_event(st.session_state.session_id, "custom_viz_prompt", user_viz_prompt)

                    prompt = f"""
            You are a Python data visualization assistant.

            TASK: Convert the following user instruction into Python code that uses matplotlib or seaborn to generate a plot from a pandas DataFrame named `df`.

            RULES:
            - DO NOT wrap the code in markdown or backticks.
            - DO NOT include explanations or comments.
            - The code should work directly with `df` and create the plot.
            - Use seaborn or matplotlib as needed.
            - The DataFrame has the following columns: {', '.join(df.columns)}.

            USER INSTRUCTION: {user_viz_prompt}
            """
                    with st.spinner("Calling LLM to generate plot..."):
                        try:
                            response = call_llm(prompt)
                            fig = execute_plot_code(response, df)
                            with st.container():
                                st.markdown("### 📊 Generated Plot")
                                st.pyplot(fig)
                            log_event(st.session_state.session_id, "custom_viz_success", user_viz_prompt)
                        except Exception as e:
                            st.error(str(e))
                            log_event(st.session_state.session_id, "custom_viz_error", str(e))




################################### Custom Cleaning Via LLM ###################################

            # add_suggestion / query_suggestions are already imported at the top of the file
            if "df" not in st.session_state:
                st.session_state.df = df.copy() 

            st.markdown("## 🛠️ Custom Cleaning via LLM")

            _hist = st.session_state.get("df_history", [])
            if st.button(f"↩️ Undo last cleaning step ({len(_hist)} available)", disabled=not _hist):
                st.session_state.df = _hist.pop()
                st.session_state["executed_actions"] = set()
                st.rerun()

            use_memory = st.checkbox("📌 Use past cleaning memory (RAG)", value=True)

            # 🔹 Show general past suggestions immediately
            if use_memory:
                related = query_suggestions("cleaning", st.session_state.session_id, n_results=5)  # dummy query to fetch some memory
                if related and "documents" in related and related["documents"][0]:
                    st.markdown("### 📌 Related Past Suggestions (general)")
                    for idx, doc in enumerate(related["documents"][0], start=1):
                        st.write(f"{idx}. {doc}")
                else:
                    st.info("⚠️ No past memory found yet.")

            with st.form(key="custom_cleaning_form"):
                user_instruction = st.text_area("Enter your custom cleaning instruction")
                submitted = st.form_submit_button("Submit to LLM")

                if submitted and user_instruction:
                    log_event(st.session_state.session_id, "custom_cleaning_prompt", user_instruction)

                    # 🔹 Step 1: Retrieve related past suggestions for this query
                    if use_memory:
                        related = query_suggestions(user_instruction, st.session_state.session_id, n_results=3)
                        if related and "documents" in related and related["documents"][0]:
                            st.markdown("### 📌 Related Past Suggestions (for your query)")
                            for idx, doc in enumerate(related["documents"][0], start=1):
                                st.write(f"{idx}. {doc}")
                        else:
                            st.info("⚠️ No past memory found yet for this query.")

                    with st.spinner("Calling LLM and applying changes..."):
                        try:
                            cleaned_df, executed_code = custom_cleaning_via_llm(user_instruction, st.session_state.df)
                            _push_history(st.session_state.df)
                            st.session_state.df = cleaned_df

                            st.success("✅ Cleaning applied successfully!")
                            st.markdown("### Executed Code")
                            st.code(executed_code, language="python")

                            # 🔹 Step 2: Save suggestion + executed code into memory
                            metadata = {
                                "id": str(uuid.uuid4()),
                                "session_id": st.session_state.session_id,
                                "instruction": user_instruction,
                                "code": executed_code
                            }
                            add_suggestion(user_instruction, metadata)

                            log_event(st.session_state.session_id, "custom_cleaning_success", executed_code)

                        except Exception as err:
                            st.error("❌ Failed to apply cleaning.")
                            st.markdown("#### Error Details")
                            st.error(str(err))

                            log_event(st.session_state.session_id, "custom_cleaning_error", str(err))

            st.subheader("📄 Current Working CSV")
            st.dataframe(st.session_state.df)


################################### Decision Tree Visualization ##################################

            if "show_tree" not in st.session_state:
                st.session_state["show_tree"] = False
            if "executed_actions" not in st.session_state:
                st.session_state["executed_actions"] = set()
            if "agraph_tree_data" not in st.session_state:
                st.session_state["agraph_tree_data"] = None
            if "agraph_leaf_nodes" not in st.session_state:
                st.session_state["agraph_leaf_nodes"] = []
            if "agraph_col" not in st.session_state:
                st.session_state["agraph_col"] = ""
            if "last_executed_click" not in st.session_state:
                st.session_state["last_executed_click"] = ""
            if "last_executed_code" not in st.session_state:
                st.session_state["last_executed_code"] = ""

            if st.button("Show Interactive Graph for Advanced Cleaning"):
                st.session_state["show_tree"] = not st.session_state["show_tree"]

            if st.session_state["show_tree"]:
                st.subheader("Interactive AGraph")

                working_df = st.session_state.df
                selected_col = st.selectbox("Select column for decision tree", working_df.columns)

                if selected_col:
                    st.write("Selected column:", selected_col)
                    log_event(st.session_state.session_id, "tree_column_selected", selected_col)   
                    
                    if (
                        st.session_state["agraph_tree_data"] is None
                        or st.session_state["agraph_col"] != selected_col
                    ):
                        response = render_agraph_tree(working_df, call_llm, selected_col)

                        if response["status"] == "success":
                            st.session_state["agraph_tree_data"] = {
                                "nodes": response["nodes"],
                                "edges": response["edges"]
                            }
                            st.session_state["agraph_leaf_nodes"] = response["leaves"]
                            st.session_state["agraph_col"] = selected_col
                            # Build child->parent lookup from the tree edges
                            # so a leaf click below can walk back up to its
                            # ancestor branch. This was previously never
                            # populated, so every click's LLM instruction
                            # silently lost the branch context (e.g. "Start
                            # -> Handle Missing -> Drop Rows") and only ever
                            # sent the leaf's own label.
                            st.session_state["agraph_parent_map"] = {
                                edge.target: edge.source for edge in response["edges"]
                            }
                            log_event(st.session_state.session_id, "agraph_tree_generated", f"Tree for {selected_col}")

                        else:
                            st.error(str(f"❌ {response['message']}"))
                            log_event(st.session_state.session_id, "agraph_tree_error", response["message"])
                            st.stop()

                    config = Config(
                        width=800,
                        height=400,
                        directed=True,
                        nodeHighlightBehavior=True,
                        highlightColor="#F7A7A6",
                        collapsible=True,
                        physics=True,
                    )

                    tree_data = st.session_state["agraph_tree_data"]
                    result = agraph(
                        nodes=tree_data["nodes"],
                        edges=tree_data["edges"],
                        config=config
                    )

                    if isinstance(result, str) and result != st.session_state["last_executed_click"]:
                        clicked_node = result
                        st.session_state["last_executed_click"] = clicked_node

                        st.write(f"Clicked node: `{clicked_node}`")

                        if clicked_node in st.session_state["agraph_leaf_nodes"]:
                            if clicked_node in st.session_state["executed_actions"]:
                                st.info(f"ℹ️ `{clicked_node}` already executed.")
                            else:
                                try:
                                    path = [clicked_node]
                                    current = clicked_node
                                    parent_map = st.session_state.get("agraph_parent_map", {})
                                    while True:
                                        parent = parent_map.get(current)
                                        if parent is None or parent.startswith("Column: "):
                                            break
                                        path.insert(0, parent)
                                        current = parent

                                    context_path = " → ".join(path)
                                    instruction = f"For the column '{selected_col}', apply the following action sequence:\n{context_path}"

                                    cleaned_df, code_str = custom_cleaning_via_llm(instruction, st.session_state.df)

                                    _push_history(st.session_state.df)
                                    st.session_state.df = cleaned_df
                                    st.session_state["executed_actions"].add(clicked_node)
                                    st.session_state["last_executed_code"] = code_str  

                                    st.success(f"Cleaning action applied for `{clicked_node}`")
                                    st.code(code_str or st.session_state["last_executed_code"], language="python")

                                    log_event(
                                                                st.session_state.session_id,
                                                                "agraph_node_cleaning_success",
                                                                f"Node: {clicked_node}, Col: {selected_col}, Path: {context_path}"
                                                            )
                                    st.subheader(" Updated Working DataFrame")

                                except Exception as e:
                                    st.error(str(f"❌ Error while applying cleaning: {e}"))
                                    log_event(
                                        st.session_state.session_id,
                                        "agraph_node_cleaning_error",
                                        f"{clicked_node} – {str(e)}"
                                    )

                    num_rows = st.slider(
                        "Rows to display",
                        min_value=5,
                        max_value=len(st.session_state.df),
                        value=10,
                        key="updated_df_row_slider"
                    )
                    st.dataframe(st.session_state.df.head(num_rows), use_container_width=True)
                    

###############################################################################
# Custom LLM Panel (outside tabs — always visible)
# Uses the same OpenRouter-hosted model as the rest of the app (call_llm()) -
# this used to call a self-hosted model at LLM.config.API_URL
# (localhost:9000), which was never reachable from the deployed container and
# always failed with a connection-refused error in production.
###############################################################################

st.title("⚡ Custom Trained LLM via API")

user_input = st.text_area(
    "Enter your query for the LLM:",
    height=150,
    placeholder="E.g., Infer type for column 'Date of Birth' and suggest cleaning steps...",
    key="llm_input_text"
)

if st.button("Submit", key="llm_submit_button"):
    if user_input.strip():
        with st.spinner("Querying the LLM..."):
            try:
                start_time = time.time()
                result = call_llm(user_input, temperature=0.3, max_tokens=700)
                end_time = time.time()

                st.markdown("**LLM Response:**")
                st.text_area("Output", result, height=250, key="llm_output_area")
                st.success(f"Time taken: {end_time - start_time:.2f} seconds")
            except Exception as e:
                st.error(str(f"Error contacting LLM API: {e}"))
    else:
        st.warning("Please enter a prompt before submitting.")




##############################################################################################
############Feedback ########################################################################
st.markdown("---")
st.subheader("🔁 Was this response helpful?")

col1, col2 = st.columns(2)
with col1:
    thumbs_up = st.button("👍 Yes", key="feedback_thumbs_up")
with col2:
    thumbs_down = st.button("👎 No", key="feedback_thumbs_down")

feedback_text = st.text_area(
    "💬 Any suggestions or comments?",
    key="feedback_textbox",
    height=100
)

# uuid is already imported at the top of the file
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    log_session(st.session_state["session_id"])

if st.button("Submit Feedback", key="feedback_submit_button"):
    feedback_value = "Thumbs Up" if thumbs_up else "Thumbs Down" if thumbs_down else "Neutral"
    feedback_detail = f"{feedback_value} | {feedback_text.strip()}" if feedback_text.strip() else feedback_value
    log_event(session_id=st.session_state["session_id"], event_type="feedback", event_detail=feedback_detail)
    st.success("✅ Feedback submitted successfully!")


with tab3:
    st.header("📚 Memory Browser")

    search_query = st.text_input("🔍 Search past suggestions", placeholder="e.g., datetime, drop column...")

    if st.button("Search Memory"):
        if search_query.strip():
            results = query_suggestions(search_query, st.session_state["session_id"], n_results=10)
            if results and "documents" in results and results["documents"][0]:
                st.success(f"Found {len(results['documents'][0])} suggestions")
                for idx, doc in enumerate(results["documents"][0], start=1):
                    meta = results["metadatas"][0][idx-1]
                    st.markdown(f"**{idx}. {doc}**")
                    st.caption(f"Session: {meta.get('session_id', '-')}, Code: `{meta.get('code', '')}`")
            else:
                st.info("⚠️ No suggestions found for this query.")

    if st.button("Show All Memory"):
        count = count_suggestions(st.session_state["session_id"])
        st.write(f"Total suggestions stored for your session: {count}")
        if count > 0:
            all_data = get_all_suggestions(st.session_state["session_id"])
            for idx, doc in enumerate(all_data["documents"], start=1):
                meta = all_data["metadatas"][idx-1]
                st.markdown(f"**{idx}. {doc}**")
                st.caption(f"Session: {meta.get('session_id', '-')}, Code: `{meta.get('code', '')}`")
