
import pandas as pd

PRIORITY_SCORE         = {"High": 3, "Medium": 2, "Low": 1}
MAINTENANCE_CYCLE_DAYS = 30
MAINTENANCE_BLOCK_DAYS = 5
TURNAROUND_HRS         = 12


def check_maintenance_overdue(trains: pd.DataFrame) -> list[dict]:
    anomalies = []
    for _, t in trains.iterrows():
        if (
            t["days_since_maintenance"] >= MAINTENANCE_CYCLE_DAYS
            and t["status"] not in ("In Maintenance",)  
        ):
            anomalies.append({
                "anomaly_id": f"A-MAINT-{t['train_id']}",
                "type":       "maintenance_overdue",
                "entity":     t["train_id"],
                "detail":     (
                    f"{t['name']} has {t['days_since_maintenance']} days since last "
                    f"maintenance (limit: {MAINTENANCE_CYCLE_DAYS})."
                ),
                "severity":   "High",
                "action":     f"Schedule {t['name']} for maintenance immediately. Remove from active trips.",
            })
    return anomalies


def check_route_stack_mismatch(containers: pd.DataFrame, routes: pd.DataFrame) -> list[dict]:
    """40ft containers cannot travel on Single Stack routes."""
    anomalies = []
    route_map = routes.set_index(["origin", "destination"]).to_dict("index")

    for _, c in containers[containers["size"] == "40ft"].iterrows():
        key   = (c["origin_yard"], c["destination"])
        route = route_map.get(key)
        if route and route["stack_type"] == "Single Stack":
            anomalies.append({
                "anomaly_id": f"A-STACK-{c['container_id']}",
                "type":       "stack_mismatch",
                "entity":     c["container_id"],
                "detail":     (
                    f"{c['container_id']} (40ft) assigned to Single Stack route "
                    f"{c['origin_yard']} → {c['destination']}."
                ),
                "severity":   "Medium",
                "action":     "Find alternate Double Stack route or hold container.",
            })
    return anomalies


def check_missing_routes(containers: pd.DataFrame, routes: pd.DataFrame) -> list[dict]:
    anomalies   = []
    valid_pairs = set(zip(routes["origin"], routes["destination"]))

    for _, c in containers.iterrows():
        if (c["origin_yard"], c["destination"]) not in valid_pairs:
            anomalies.append({
                "anomaly_id": f"A-NOROUTE-{c['container_id']}",
                "type":       "missing_route",
                "entity":     c["container_id"],
                "detail":     f"No route found for {c['origin_yard']} → {c['destination']}.",
                "severity":   "High",
                "action":     "Manually assign route or flag for ops review.",
            })
    return anomalies


def check_train_capacity(
    trips: pd.DataFrame,
    containers: pd.DataFrame,
    trains: pd.DataFrame,
) -> list[dict]:
    """20ft = 1 TEU, 40ft = 2 TEU. Verify each trip stays within train capacity."""
    anomalies     = []
    size_to_teu   = {"20ft": 1, "40ft": 2}
    container_map = containers.set_index("container_id")["size"].to_dict()
    train_cap     = trains.set_index("train_id")["capacity_TEU"].to_dict()

    for _, trip in trips.iterrows():
        assigned = [
            c.strip()
            for c in str(trip["containers_assigned"]).split(",")
            if c.strip()
        ]
        teu_load = sum(
            size_to_teu.get(container_map.get(c, "20ft"), 1) for c in assigned
        )
        cap = train_cap.get(trip["train_id"], 999)
        if teu_load > cap:
            anomalies.append({
                "anomaly_id": f"A-CAP-{trip['trip_id']}",
                "type":       "capacity_exceeded",
                "entity":     trip["trip_id"],
                "detail":     (
                    f"Trip {trip['trip_id']} load {teu_load} TEU exceeds "
                    f"{trip['train_id']} capacity {cap} TEU."
                ),
                "severity":   "High",
                "action":     "Remove containers from trip until within capacity.",
            })
    return anomalies


def check_station_capacity(
    trips: pd.DataFrame,
    containers: pd.DataFrame,
    stations: pd.DataFrame,
    routes: pd.DataFrame,
) -> list[dict]:
    """Aggregate TEU per destination per day vs station daily capacity."""
    anomalies     = []
    size_to_teu   = {"20ft": 1, "40ft": 2}
    container_map = containers.set_index("container_id")["size"].to_dict()
    station_cap   = stations.set_index("station_name")["handling_capacity_TEU_per_day"].to_dict()
    route_dest    = routes.set_index("route_id")["destination"].to_dict()

    daily_load: dict[tuple, int] = {}
    for _, trip in trips.iterrows():
        dest     = route_dest.get(trip["route_id"], "UNKNOWN")
        date     = str(trip["arrival_datetime"])[:10]
        assigned = [
            c.strip()
            for c in str(trip["containers_assigned"]).split(",")
            if c.strip()
        ]
        teu_load = sum(
            size_to_teu.get(container_map.get(c, "20ft"), 1) for c in assigned
        )
        key = (dest, date)
        daily_load[key] = daily_load.get(key, 0) + teu_load

    for (dest, date), load in daily_load.items():
        cap = station_cap.get(dest, 999)
        if load > cap:
            anomalies.append({
                "anomaly_id": f"A-STCAP-{dest}-{date}",
                "type":       "station_capacity_exceeded",
                "entity":     dest,
                "detail":     (
                    f"{dest} receives {load} TEU on {date}, "
                    f"exceeding daily capacity of {cap} TEU."
                ),
                "severity":   "Medium",
                "action":     f"Stagger arrivals at {dest} or delay lower-priority trips.",
            })
    return anomalies


def compute_data_health_score(all_anomalies: list[dict], total_entities: int) -> float:
    severity_weight = {"High": 3, "Medium": 2, "Low": 1}
    total_weight    = sum(severity_weight.get(a["severity"], 1) for a in all_anomalies)
    max_possible    = total_entities * 3
    score = max(0.0, 1.0 - (total_weight / max(max_possible, 1)))
    return round(score, 2)
