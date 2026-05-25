
import json
import pandas as pd
from openai import AzureOpenAI


from optimizer_engine import build_schedule
from config import (
    AZURE_OPENAI_CHAT_API_KEY,
    AZURE_OPENAI_CHAT_ENDPOINT,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)

# ── Azure OpenAI client ───────────────────────────────────────────────────────
llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_version="2024-02-01",
)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_agent2(
    agent1_context: dict,
    scenario: str = "baseline",
    knocked_out_train: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Parameters
    ----------
    agent1_context    : output dict from run_agent1()
    scenario          : "baseline" | "train_knockout" | "priority_only"
    knocked_out_train : train_id string, only used when scenario == "train_knockout"
    progress_callback : optional callable(str) for live UI log updates

    Returns
    -------
    Full action report dict
    """

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    tables      = agent1_context["tables"]
    containers  = tables["containers"].copy()
    trains      = tables["trains"].copy()
    routes      = tables["routes"].copy()
    trips       = tables["trips"].copy()
    maintenance = tables["maintenance"].copy()

    # ── Step 1: Apply scenario modifications ──────────────────────────────────
    _log(f"⚙️  Applying scenario: {scenario}...")

    if scenario == "train_knockout" and knocked_out_train:
        trains.loc[
            trains["train_id"] == knocked_out_train, "status"
        ] = "In Maintenance"
        trains.loc[
            trains["train_id"] == knocked_out_train, "days_since_maintenance"
        ] = 99
        _log(f"🚨 Simulating breakdown — {knocked_out_train} marked unavailable.")

    elif scenario == "priority_only":
        containers = containers[
            containers["priority"].str.strip().str.lower() == "high"
        ].copy()
        _log("🎯 Priority-only mode: running High priority containers only.")

    # ── Step 2: CP-SAT solve ──────────────────────────────────────────────────
    _log("🔢 Running CP-SAT optimization solver...")

    solver_result = build_schedule(
        containers=containers,
        trains=trains,
        routes=routes,
        trips=trips,
        maintenance=maintenance,
        planning_days=7,
    )

    # ── Step 3: Enrich assignments with human-readable details ────────────────
    _log("📋 Enriching assignment results...")

    route_details     = routes.set_index("route_id").to_dict("index")
    train_details     = trains.set_index("train_id").to_dict("index")
    container_details = tables["containers"].set_index("container_id").to_dict("index")

    enriched_assignments = []
    for a in solver_result["assignments"]:
        route = route_details.get(a["route_id"], {})
        train = train_details.get(a["train_id"], {})
        cdata = container_details.get(a["container_id"], {})
        enriched_assignments.append({
            "container_id":    a["container_id"],
            "size":            cdata.get("size", "—"),
            "priority":        a["priority"],
            "origin":          cdata.get("origin_yard", "—"),
            "destination":     cdata.get("destination", "—"),
            "train_id":        a["train_id"],
            "train_name":      train.get("name", "—"),
            "route_id":        a["route_id"],
            "stack_type":      route.get("stack_type", "—"),
            "travel_time_hrs": route.get("travel_time_hrs", "—"),
            "teu":             a["teu"],
        })

    # ── Step 4: Resolve Agent 1 exceptions ───────────────────────────────────
    _log("🔧 Resolving exceptions flagged by Agent 1...")

    exception_resolutions = _resolve_exceptions(
        anomalies=agent1_context["anomalies"],
        trains=trains,
        routes=routes,
    )

    # ── Step 5: Plain-text action summary ────────────────────────────────────
    _log("📝 Building action summary...")

    action_summary = _build_action_summary(
        assignments=enriched_assignments,
        unassigned=solver_result["unassigned"],
        blocked_trains=solver_result["blocked_trains"],
        solver_status=solver_result["solver_status"],
        total_pending=agent1_context["total_pending"],
        scenario=scenario,
        knocked_out_train=knocked_out_train,
    )

    # ── Step 6: LLM explanation via Azure OpenAI ──────────────────────────────
    _log("🤖 Agent 2 generating plain-language explanation...")

    llm_explanation = _llm_explain(
        assignments=enriched_assignments,
        unassigned=solver_result["unassigned"],
        blocked_trains=solver_result["blocked_trains"],
        solver_status=solver_result["solver_status"],
        anomalies=agent1_context["anomalies"],
        scenario=scenario,
        knocked_out_train=knocked_out_train,
        health_score=agent1_context["health_score"],
    )

    _log("✅ Agent 2 complete. Schedule ready.")

    return {
        "scenario":              scenario,
        "knocked_out_train":     knocked_out_train,
        "solver_status":         solver_result["solver_status"],
        "assignments":           enriched_assignments,
        "unassigned":            solver_result["unassigned"],
        "blocked_trains":        solver_result["blocked_trains"],
        "available_trains":      solver_result["available_trains"],
        "pendency_cleared":      solver_result["pendency_cleared"],
        "total_teu_cleared":     solver_result["total_teu_cleared"],
        "total_pending":         agent1_context["total_pending"],
        "exception_resolutions": exception_resolutions,
        "action_summary":        action_summary,
        "llm_explanation":       llm_explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: exception resolution
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_exceptions(
    anomalies: list[dict],
    trains: pd.DataFrame,
    routes: pd.DataFrame,
) -> list[dict]:

    resolutions        = []
    operational_trains = trains[
        trains["status"].str.strip() == "Operational"
    ]["train_id"].tolist()

    for a in anomalies:
        resolution = {
            "anomaly_id":     a["anomaly_id"],
            "type":           a["type"],
            "entity":         a["entity"],
            "severity":       a["severity"],
            "original_issue": a["detail"],
        }

        if a["type"] == "maintenance_overdue":
            resolution["resolution"] = (
                f"Train {a['entity']} excluded from schedule automatically. "
                f"Maintenance window: next 5 days. "
                f"Redistribute to: {', '.join(operational_trains) or 'None available'}."
            )
            resolution["status"] = "Auto-Resolved"

        elif a["type"] == "stack_mismatch":
            alt_routes = routes[
                routes["stack_type"] == "Double Stack"
            ]["route_id"].tolist()
            resolution["resolution"] = (
                f"Container {a['entity']} (40ft) blocked on Single Stack route. "
                f"Available Double Stack routes: {alt_routes}. "
                f"Manual rerouting needed if no matching O-D exists."
            )
            resolution["status"] = "Needs Manual Review"

        elif a["type"] == "missing_route":
            resolution["resolution"] = (
                f"Container {a['entity']} has no configured route. "
                f"Held in yard. Escalated to ops team for route configuration."
            )
            resolution["status"] = "Escalated"

        elif a["type"] == "capacity_exceeded":
            resolution["resolution"] = (
                f"Trip {a['entity']} overloaded. Solver excluded excess containers. "
                f"Verify final assignment manually."
            )
            resolution["status"] = "Auto-Resolved"

        elif a["type"] == "station_capacity_exceeded":
            resolution["resolution"] = (
                f"Station {a['entity']} overloaded on flagged date. "
                f"Stagger lower-priority arrivals by 1 day."
            )
            resolution["status"] = "Needs Manual Review"

        else:
            resolution["resolution"] = "Review manually."
            resolution["status"]     = "Open"

        resolutions.append(resolution)

    return resolutions


# ─────────────────────────────────────────────────────────────────────────────
# Helper: plain-text action summary
# ─────────────────────────────────────────────────────────────────────────────
def _build_action_summary(
    assignments: list[dict],
    unassigned: list[dict],
    blocked_trains: list[str],
    solver_status: str,
    total_pending: int,
    scenario: str,
    knocked_out_train: str | None,
) -> str:

    scenario_label = {
        "baseline":       "Baseline — Full Optimization",
        "train_knockout": f"Simulation — {knocked_out_train} Breakdown",
        "priority_only":  "High Priority Containers Only",
    }.get(scenario, scenario)

    lines = [
        "=" * 54,
        "   ACTION SCHEDULE — RAIL ROUTE OPTIMIZER",
        f"   Mode    : {scenario_label}",
        f"   Solver  : {solver_status}",
        "=" * 54,
        "",
        "📦 ASSIGNMENTS",
    ]

    if assignments:
        for a in assignments:
            lines.append(
                f"  ✅ {a['container_id']} ({a['size']}, {a['priority']}) "
                f"→ {a['train_name']} [{a['train_id']}] via {a['route_id']} "
                f"| {a['origin']} → {a['destination']} "
                f"| {a['teu']} TEU | {a['travel_time_hrs']} hrs travel"
            )
    else:
        lines.append("  ❌ No containers could be assigned.")

    lines += ["", "⚠️  UNASSIGNED CONTAINERS"]
    if unassigned:
        for u in unassigned:
            lines.append(f"  🔴 {u['container_id']} — {u['reason']}")
    else:
        lines.append("  ✅ All pending containers assigned.")

    lines += ["", "🔧 BLOCKED TRAINS"]
    if blocked_trains:
        for t in blocked_trains:
            lines.append(f"  🚫 {t} — excluded (maintenance / overdue)")
    else:
        lines.append("  ✅ All trains operational.")

    teu_cleared = sum(a["teu"] for a in assignments)
    remaining   = total_pending - len(assignments)

    lines += [
        "",
        "📊 SUMMARY",
        f"  Total Pending      : {total_pending}",
        f"  Pendency Cleared   : {len(assignments)}",
        f"  Remaining Pendency : {remaining}",
        f"  Total TEU Cleared  : {teu_cleared}",
        "=" * 54,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: LLM plain-language explanation
# ─────────────────────────────────────────────────────────────────────────────
def _llm_explain(
    assignments: list[dict],
    unassigned: list[dict],
    blocked_trains: list[str],
    solver_status: str,
    anomalies: list[dict],
    scenario: str,
    knocked_out_train: str | None,
    health_score: float,
) -> str:

    payload = {
        "solver_status":     solver_status,
        "scenario":          scenario,
        "knocked_out_train": knocked_out_train,
        "health_score":      health_score,
        "assignments":       assignments,
        "unassigned":        unassigned,
        "blocked_trains":    blocked_trains,
        "anomalies_count":   len(anomalies),
    }

    prompt = f"""
You are an AI operations advisor for a port rail scheduling system.
The CP-SAT optimizer has just produced the following result:

{json.dumps(payload, indent=2)}

Write a clear, concise explanation (max 250 words) for the port operations manager covering:
1. What the optimizer decided and why (plain English).
2. Which containers were cleared and which were held — with the specific reason for each held container.
3. Which trains are blocked and why.
4. Any risks or watch-outs the manager should act on today.
5. One recommended next step.

Tone: professional, direct. Use short paragraphs — no bullet soup.
"""

    try:
        response = llm_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM explanation unavailable: {str(e)}"
