
import io
import json
import pandas as pd
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
import openpyxl

from config import (
    AZURE_BLOB_CONNECTION_STRING,
    AZURE_BLOB_CONTAINER,
    BLOB_CONT_PEND_FILE,
    BLOB_MAINT_LOG_FILE,
    BLOB_ROUTES_FILES,
    BLOB_STN_LDUNLD_TIME_FILE,
    BLOB_TRN_SCH_TRIPS_FILE,
    BLOB_TRN_DATA_FILE,
    AZURE_OPENAI_CHAT_API_KEY,
    AZURE_OPENAI_CHAT_ENDPOINT,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)
from data_checks import (
    check_maintenance_overdue,
    check_route_stack_mismatch,
    check_missing_routes,
    check_train_capacity,
    check_station_capacity,
    compute_data_health_score,
)

# ── Azure OpenAI client ───────────────────────────────────────────────────────
llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_version="2024-02-01",
)


# ── Blob reader ───────────────────────────────────────────────────────────────
def _read_excel_from_blob(blob_name: str) -> pd.DataFrame:
    """Download an Excel blob and return as DataFrame. Raises clear error on failure."""
    try:
        service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
        container_client = service.get_container_client(AZURE_BLOB_CONTAINER)
        blob = container_client.get_blob_client(blob_name)
        data = blob.download_blob().readall()
        return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        # List available blobs to help diagnose
        try:
            container_client = service.get_container_client(AZURE_BLOB_CONTAINER)
            available = [b.name for b in container_client.list_blobs()]
        except:
            available = ["(could not list blobs)"]
        raise FileNotFoundError(
            f"\n❌ Blob not found: '{blob_name}'"
            f"\n📦 Container: '{AZURE_BLOB_CONTAINER}'"
            f"\n✅ Available blobs: {available}"
            f"\n🔧 Original error: {e}"
        )
def _load_all_tables(progress_callback=None) -> dict[str, pd.DataFrame]:
    def _log(msg):
        if progress_callback:
            progress_callback(msg)
    file_map = {
        "containers":  BLOB_CONT_PEND_FILE,
        "maintenance": BLOB_MAINT_LOG_FILE,
        "routes":      BLOB_ROUTES_FILES,
        "stations":    BLOB_STN_LDUNLD_TIME_FILE,
        "trips":       BLOB_TRN_SCH_TRIPS_FILE,
        "trains":      BLOB_TRN_DATA_FILE,
    }
    tables = {}
    for key, fname in file_map.items():
        _log(f"📥 Loading '{fname}' from Azure Blob...")
        tables[key] = _read_excel_from_blob(fname)
        _log(f"✅ Loaded '{fname}' — {len(tables[key])} rows")
    return tables


# ── Main entry point ──────────────────────────────────────────────────────────
def run_agent1(progress_callback=None) -> dict:
    """
    Entry point for Agent 1.
    Returns validated context dict consumed directly by Agent 2.
    """

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # ── Step 1: Load all tables ───────────────────────────────────────────────
    tables = _load_all_tables(progress_callback)

    containers  = tables["containers"]
    trains      = tables["trains"]
    routes      = tables["routes"]
    stations    = tables["stations"]
    trips       = tables["trips"]
    maintenance = tables["maintenance"]

    # ── Step 2: Run all validation checks ────────────────────────────────────
    _log("🔍 Checking maintenance overdue...")
    a1 = check_maintenance_overdue(trains)

    _log("🔍 Checking stack-type mismatches...")
    a2 = check_route_stack_mismatch(containers, routes)

    _log("🔍 Checking missing routes...")
    a3 = check_missing_routes(containers, routes)

    _log("🔍 Checking train capacities...")
    a4 = check_train_capacity(trips, containers, trains)

    _log("🔍 Checking station capacities...")
    a5 = check_station_capacity(trips, containers, stations, routes)

    anomalies = a1 + a2 + a3 + a4 + a5

    # ── Step 3: Data health score ─────────────────────────────────────────────
    total_entities = len(containers) + len(trains) + len(trips)
    health_score   = compute_data_health_score(anomalies, total_entities)

    # ── Step 4: LLM narrative via Azure OpenAI ────────────────────────────────
    _log("🤖 Agent 1 generating validation narrative...")

    anomaly_text = json.dumps(anomalies, indent=2) if anomalies else "No anomalies found."

    prompt = f"""
You are a rail operations data auditor for a port.
You have just completed automated checks on the rail scheduling data.

Data Health Score: {health_score} / 1.0
Total anomalies found: {len(anomalies)}

Anomaly details:
{anomaly_text}

Write a concise, plain-English validation report (max 200 words) for the port operations manager.
Structure it as:
1. Overall data health (one line).
2. Critical issues requiring immediate action (bullet points).
3. Warnings that need monitoring (bullet points).
4. What data is clean and ready for optimization.

Be direct and action-oriented. No fluff.
"""

    llm_summary = ""
    try:
        response = llm_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        llm_summary = response.choices[0].message.content.strip()
    except Exception as e:
        llm_summary = f"LLM summary unavailable: {str(e)}"

    # ── Step 5: Build clean/flagged splits for Agent 2 ────────────────────────
    flagged_container_ids = set(
        a["entity"] for a in anomalies
        if a["type"] in ("stack_mismatch", "missing_route")
    )
    flagged_train_ids = set(
        a["entity"] for a in anomalies
        if a["type"] == "maintenance_overdue"
    )

    clean_containers   = containers[~containers["container_id"].isin(flagged_container_ids)].copy()
    flagged_containers = containers[containers["container_id"].isin(flagged_container_ids)].copy()

    # ── Step 6: Return validated context ─────────────────────────────────────
    _log("✅ Agent 1 complete. Passing validated context to Agent 2...")

    return {
        # Full raw tables — Agent 2 needs all of these for the solver
        "tables": {
            "containers":  containers,
            "trains":      trains,
            "routes":      routes,
            "stations":    stations,
            "trips":       trips,
            "maintenance": maintenance,
        },
        # Clean/flagged splits
        "clean_containers":   clean_containers,
        "flagged_containers": flagged_containers,
        "flagged_train_ids":  list(flagged_train_ids),
        # Anomaly report
        "anomalies":          anomalies,
        "health_score":       health_score,
        "llm_summary":        llm_summary,
        # Counts
        "total_pending":      int(
            (containers["status"].str.strip().str.lower() == "pending").sum()
        ),
        "total_anomalies":    len(anomalies),
    }
