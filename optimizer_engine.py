
import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta


PRIORITY_SCORE  = {"High": 3, "Medium": 2, "Low": 1}
SIZE_TO_TEU     = {"20ft": 1, "40ft": 2}
STACK_COMPAT    = {"Double Stack": ["20ft", "40ft"], "Single Stack": ["20ft"]}
TURNAROUND_HRS  = 12
MAINT_CYCLE     = 30
MAINT_BLOCK     = 5


def build_schedule(
    containers: pd.DataFrame,
    trains: pd.DataFrame,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    maintenance: pd.DataFrame,
    planning_days: int = 7,
) -> dict:
    """
    Returns:
        {
          "assignments": [ {container_id, train_id, route_id, trip_slot, teu} ],
          "unassigned":  [ {container_id, reason} ],
          "solver_status": str,
          "total_teu_cleared": int,
          "pendency_cleared": int,
        }
    """

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # ── Available trains (not in maintenance, not overdue) ────────────────────
    maint_blocked = set(
        maintenance[maintenance["status"] == "Upcoming"]["train_id"].tolist()
    )
    overdue = set(
        trains[trains["days_since_maintenance"] >= MAINT_CYCLE]["train_id"].tolist()
    )
    unavailable = set(
        trains[trains["status"] == "In Maintenance"]["train_id"].tolist()
    )
    blocked_trains = maint_blocked | overdue | unavailable

    available_trains = trains[~trains["train_id"].isin(blocked_trains)].copy()

    # ── Pending containers only ───────────────────────────────────────────────
    pending = containers[containers["status"] == "Pending"].copy()

    # ── Build (container, route, train) feasibility triples ──────────────────
    route_map = {}
    for _, r in routes.iterrows():
        route_map[(r["origin"], r["destination"])] = r

    assignments = {}   # (c_id, t_id, r_id) → BoolVar
    feasible    = []

    for _, c in pending.iterrows():
        c_id = c["container_id"]
        key  = (c["origin_yard"], c["destination"])
        route = route_map.get(key)
        if route is None:
            continue  # no route → unassigned

        # Stack compatibility check
        if c["size"] not in STACK_COMPAT.get(route["stack_type"], []):
            continue

        for _, t in available_trains.iterrows():
            t_id = t["train_id"]
            r_id = route["route_id"]
            var  = model.NewBoolVar(f"assign_{c_id}_{t_id}_{r_id}")
            assignments[(c_id, t_id, r_id)] = var
            feasible.append((c_id, t_id, r_id, c["size"], c["priority"], t["capacity_TEU"]))

    # ── Constraint: each container assigned at most once ──────────────────────
    for _, c in pending.iterrows():
        c_id = c["container_id"]
        related = [v for (ci, ti, ri), v in assignments.items() if ci == c_id]
        if related:
            model.Add(sum(related) <= 1)

    # ── Constraint: train capacity (TEU) ──────────────────────────────────────
    train_capacity = available_trains.set_index("train_id")["capacity_TEU"].to_dict()
    for t_id, cap in train_capacity.items():
        train_vars = [
            (assignments[(ci, ti, ri)], SIZE_TO_TEU.get(size, 1))
            for (ci, ti, ri), _ in assignments.items()
            if ti == t_id
            for _, _, _, size, _, _ in [f for f in feasible if f[0] == ci and f[1] == ti]
        ]
        # Rebuild cleanly
        t_terms = []
        for (ci, ti, ri), var in assignments.items():
            if ti != t_id:
                continue
            size = pending[pending["container_id"] == ci]["size"].values
            teu  = SIZE_TO_TEU.get(size[0], 1) if len(size) else 1
            t_terms.append(var * teu)
        if t_terms:
            model.Add(sum(t_terms) <= cap)

    # ── Objective: maximise weighted pendency removal ─────────────────────────
    objective_terms = []
    for (c_id, t_id, r_id), var in assignments.items():
        priority = pending[pending["container_id"] == c_id]["priority"].values
        score    = PRIORITY_SCORE.get(priority[0], 1) if len(priority) else 1
        objective_terms.append(var * score)

    model.Maximize(sum(objective_terms))

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    # ── Extract results ───────────────────────────────────────────────────────
    assigned_containers = set()
    result_assignments  = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (c_id, t_id, r_id), var in assignments.items():
            if solver.Value(var) == 1:
                size = pending[pending["container_id"] == c_id]["size"].values[0]
                result_assignments.append({
                    "container_id": c_id,
                    "train_id":     t_id,
                    "route_id":     r_id,
                    "teu":          SIZE_TO_TEU.get(size, 1),
                    "priority":     pending[pending["container_id"] == c_id]["priority"].values[0],
                })
                assigned_containers.add(c_id)

    # ── Unassigned + reasons ──────────────────────────────────────────────────
    unassigned = []
    for _, c in pending.iterrows():
        if c["container_id"] not in assigned_containers:
            key   = (c["origin_yard"], c["destination"])
            route = route_map.get(key)
            if route is None:
                reason = "No route exists for this origin→destination."
            elif c["size"] not in STACK_COMPAT.get(route["stack_type"], []):
                reason = f"Container size {c['size']} incompatible with {route['stack_type']} route."
            elif not available_trains.empty:
                reason = "Could not fit within available train capacities."
            else:
                reason = "No trains available (maintenance/overdue blocks)."
            unassigned.append({"container_id": c["container_id"], "reason": reason})

    status_map = {
        cp_model.OPTIMAL:   "OPTIMAL",
        cp_model.FEASIBLE:  "FEASIBLE",
        cp_model.INFEASIBLE:"INFEASIBLE",
        cp_model.UNKNOWN:   "UNKNOWN",
    }

    return {
        "assignments":       result_assignments,
        "unassigned":        unassigned,
        "solver_status":     status_map.get(status, "UNKNOWN"),
        "blocked_trains":    list(blocked_trains),
        "available_trains":  available_trains["train_id"].tolist(),
        "pendency_cleared":  len(result_assignments),
        "total_teu_cleared": sum(a["teu"] for a in result_assignments),
    }
