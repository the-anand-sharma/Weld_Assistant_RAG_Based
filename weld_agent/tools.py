"""Tools for the Weld Quality Agent.

Each tool is a plain Python function. ADK reads the type hints and the
docstring to tell Gemini what the tool does and which arguments it needs.
Every tool returns a dict with a "status" key so the model can tell a real
result from an error.
"""

import io
import json
from pathlib import Path

import pandas as pd
from google.adk.tools import ToolContext

DATA_DIR = Path(__file__).parent / "data"
WPS_FILE = DATA_DIR / "weld_data.txt"
LIMITS_FILE = DATA_DIR / "wps_limits.json"
SAMPLE_CSV = DATA_DIR / "sample_weld_log.csv"

# The FAISS index is built the first time search_wps is called, then reused.
_vectorstore = None


def _get_vectorstore():
    """Build the FAISS index over the WPS document (once)."""
    global _vectorstore
    if _vectorstore is None:
        # Imported here because loading the embedding model takes a few seconds.
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings

        text = WPS_FILE.read_text(encoding="utf-8")
        sections = text.split("\n\n")
        # First line of the file: "WELDING PROCEDURE SPECIFICATION - WPS-2026-001"
        wps_id = sections[0].split(" - ")[-1].strip()

        # One chunk per section (JOINT DESIGN, BASE METAL, ...). The WPS id is
        # added to every chunk so a query like "WPS-2026-001 preheat" matches.
        docs = []
        for section in sections[1:]:
            heading = section.splitlines()[0]
            docs.append(
                Document(
                    page_content=f"{wps_id}\n{section}",
                    metadata={"wps_id": wps_id, "source": f"{WPS_FILE.name} > {heading}"},
                )
            )

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = FAISS.from_documents(docs, embeddings)
    return _vectorstore


def search_wps(query: str) -> dict:
    """Searches the welding procedure specification (WPS) documents.

    Use this for any question about what the WPS says: joint design, base and
    filler metal, preheat, welding parameters, PWHT, inspection results, defect
    acceptance limits, the defect log or welder qualification.

    Args:
        query: What to look for, e.g. "preheat temperature for WPS-2026-001".

    Returns:
        The 3 most relevant WPS sections, each with its text and source reference.
    """
    matches = _get_vectorstore().similarity_search(query, k=3)
    results = []
    for doc in matches:
        results.append(
            {
                "wps_id": doc.metadata["wps_id"],
                "source": doc.metadata["source"],
                "text": doc.page_content,
            }
        )
    return {"status": "success", "results": results}


def _load_pass_limits(wps_id: str, weld_pass: str) -> dict:
    """Returns {"limits": ...} for one pass of one WPS, or {"error": ...}."""
    all_limits = json.loads(LIMITS_FILE.read_text(encoding="utf-8"))
    if wps_id not in all_limits:
        return {"error": f"No limits on file for '{wps_id}'. Known WPS: {list(all_limits)}"}
    passes = all_limits[wps_id]
    weld_pass = weld_pass.lower()
    if weld_pass not in passes:
        return {"error": f"'{wps_id}' has no pass '{weld_pass}'. Known passes: {list(passes)}"}
    return {"limits": passes[weld_pass]}


def check_weld_parameters(
    wps_id: str, weld_pass: str, current: float, voltage: float, travel_speed: float
) -> dict:
    """Checks actual welding parameters against the WPS limits for one pass.

    Args:
        wps_id: The WPS number, e.g. "WPS-2026-001".
        weld_pass: Which pass was welded: "root", "hot", "fill" or "cap".
        current: Actual welding current in amperes.
        voltage: Actual arc voltage in volts.
        travel_speed: Actual travel speed in cm/min.

    Returns:
        PASS or FAIL for each parameter with the allowed range, and an overall result.
    """
    found = _load_pass_limits(wps_id, weld_pass)
    if "error" in found:
        return {"status": "error", "message": found["error"]}
    limits = found["limits"]

    actual_values = {
        "current_A": current,
        "voltage_V": voltage,
        "travel_speed_cm_min": travel_speed,
    }
    checks = {}
    for name, actual in actual_values.items():
        low, high = limits[name]["min"], limits[name]["max"]
        checks[name] = {
            "actual": actual,
            "allowed_min": low,
            "allowed_max": high,
            "result": "PASS" if low <= actual <= high else "FAIL",
        }

    all_passed = all(check["result"] == "PASS" for check in checks.values())
    return {
        "status": "success",
        "wps_id": wps_id,
        "weld_pass": weld_pass.lower(),
        "overall": "PASS" if all_passed else "FAIL",
        "checks": checks,
    }


def _latest_uploaded_csv(tool_context: ToolContext) -> bytes | None:
    """Returns the bytes of the newest CSV the user attached in this chat.

    Uploaded files arrive as "inline data" parts inside the user's messages,
    which ADK keeps in the session's event history.
    """
    for event in reversed(tool_context.session.events):
        if event.content is None or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.inline_data and "csv" in (part.inline_data.mime_type or ""):
                return part.inline_data.data
    return None


def analyze_weld_csv(
    file_name: str, wps_id: str, weld_pass: str, tool_context: ToolContext
) -> dict:
    """Analyzes a welding sensor log (CSV) against the WPS limits for one pass.

    The CSV needs the columns timestamp_s, current_A and voltage_V.

    Args:
        file_name: "uploaded" for the CSV the user attached in this chat, or
            "sample_weld_log.csv" for the built-in sample log.
        wps_id: The WPS number to check against, e.g. "WPS-2026-001".
        weld_pass: Which pass the log is from: "root", "hot", "fill" or "cap".

    Returns:
        Mean/min/max/std for current and voltage, and how many samples were
        outside the WPS range and when.
    """
    found = _load_pass_limits(wps_id, weld_pass)
    if "error" in found:
        return {"status": "error", "message": found["error"]}
    limits = found["limits"]

    if file_name == "uploaded":
        csv_bytes = _latest_uploaded_csv(tool_context)
        if csv_bytes is None:
            return {"status": "error", "message": "No CSV file has been uploaded in this chat."}
        df = pd.read_csv(io.BytesIO(csv_bytes))
    elif file_name == SAMPLE_CSV.name:
        df = pd.read_csv(SAMPLE_CSV)
    else:
        return {"status": "error", "message": f"File '{file_name}' not found."}

    needed = ["timestamp_s", "current_A", "voltage_V"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        return {"status": "error", "message": f"CSV is missing columns: {missing}"}

    signals = {}
    for name in ["current_A", "voltage_V"]:
        low, high = limits[name]["min"], limits[name]["max"]
        values = df[name]
        out_of_range = df[(values < low) | (values > high)]
        signals[name] = {
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
            "std": round(float(values.std()), 2),
            "allowed_min": low,
            "allowed_max": high,
            "samples_out_of_range": len(out_of_range),
            "percent_out_of_range": round(100 * len(out_of_range) / len(df), 1),
            "out_of_range_times_s": out_of_range["timestamp_s"].tolist()[:20],
        }

    return {
        "status": "success",
        "file": file_name,
        "wps_id": wps_id,
        "weld_pass": weld_pass.lower(),
        "total_samples": len(df),
        "duration_s": round(float(df["timestamp_s"].max() - df["timestamp_s"].min()), 1),
        "signals": signals,
    }
