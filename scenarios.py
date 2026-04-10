import random
import numpy as np
import math

random.seed(42)
np.random.seed(42)


def get_scenario_config(scenario_name):
    if scenario_name == 'easy':
        return easy_scenario()
    elif scenario_name == 'medium':
        return medium_scenario()
    elif scenario_name == 'hard':
        return hard_scenario()
    return easy_scenario()


def easy_scenario():
    return {
        "days": 15,
        "products": [
            {"id": "AUX-HIGH-01", "name": "Standard Charger", "type": "high", "demand_base": 20, "demand_fluctuation": 2, "storage_cost": 0.1, "warehouse_capacity": 500, "product_price": 25.0},
            {"id": "AUX-LOW-02", "name": "Basic Mousepad", "type": "low", "demand_base": 5, "demand_fluctuation": 1, "storage_cost": 0.1, "warehouse_capacity": 500, "product_price": 15.0},
            {"id": "AUX-VAR-03", "name": "USB Cable Hub", "type": "fluctuating", "demand_base": 12, "demand_fluctuation": 5, "storage_cost": 0.1, "warehouse_capacity": 500, "product_price": 35.0}
        ]
    }


def medium_scenario():
    return {
        "days": 30,
        "products": [
            {"id": "AUX-HIGH-01", "name": "Essential Hoodie", "type": "high", "demand_base": 40, "demand_fluctuation": 10, "storage_cost": 0.5, "warehouse_capacity": 1000, "product_price": 45.0},
            {"id": "AUX-LOW-02", "name": "Dumbbells (5kg)", "type": "low", "demand_base": 10, "demand_fluctuation": 2, "storage_cost": 1.0, "warehouse_capacity": 500, "product_price": 60.0},
            {"id": "AUX-VAR-03", "name": "Water Lantern", "type": "fluctuating", "demand_base": 25, "demand_fluctuation": 20, "storage_cost": 0.5, "warehouse_capacity": 800, "product_price": 90.0}
        ]
    }


def hard_scenario():
    return {
        "days": 50,
        "event_chance": 0.3,
        "event_multiplier": 6,
        "products": [
            {"id": "AUX-HIGH-01", "name": "Kindle Reader", "type": "high", "demand_base": 80, "demand_fluctuation": 30, "storage_cost": 1.0, "warehouse_capacity": 2000, "product_price": 120.0},
            {"id": "AUX-LOW-02", "name": "Luxury Perfume", "type": "low", "demand_base": 15, "demand_fluctuation": 5, "storage_cost": 2.0, "warehouse_capacity": 500, "product_price": 250.0},
            {"id": "AUX-VAR-03", "name": "Mega Sale Day Lights", "type": "fluctuating", "demand_base": 50, "demand_fluctuation": 80, "storage_cost": 1.0, "warehouse_capacity": 1500, "product_price": 150.0}
        ]
    }


SCENARIO_STATS = {
    "easy":   {"baseline": 800,   "optimal": 2200,  "days": 15},
    "medium": {"baseline": 3500,  "optimal": 12000, "days": 30},
    "hard":   {"baseline": 12000, "optimal": 65000, "days": 50}
}


def _safe_float(val, default=0.5):
    """Convert any value to float safely."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _extract_metrics(data):
    """
    Safely extract profit_metrics from WHATEVER the validator passes in.
    Handles: profit_metrics dict, full info dict, nested structures, None, etc.
    """
    if data is None or not isinstance(data, dict):
        return {"profit": 0.0, "cost_efficiency": 0.5, "stockout_control": 0.5, "decision_quality": 0.5}

    # Case 1: already a profit_metrics dict (has 'profit' key directly)
    if "profit" in data:
        return {
            "profit":          _safe_float(data.get("profit", 0.0), 0.0),
            "cost_efficiency": _safe_float(data.get("cost_efficiency", 0.5), 0.5),
            "stockout_control":_safe_float(data.get("stockout_control", 0.5), 0.5),
            "decision_quality":_safe_float(data.get("decision_quality", 0.5), 0.5),
        }

    # Case 2: full info dict — dig into cumulative_stats.profit_metrics
    cs = data.get("cumulative_stats", {})
    if isinstance(cs, dict):
        pm = cs.get("profit_metrics", {})
        if isinstance(pm, dict) and "profit" in pm:
            return {
                "profit":          _safe_float(pm.get("profit", 0.0), 0.0),
                "cost_efficiency": _safe_float(pm.get("cost_efficiency", 0.5), 0.5),
                "stockout_control":_safe_float(pm.get("stockout_control", 0.5), 0.5),
                "decision_quality":_safe_float(pm.get("decision_quality", 0.5), 0.5),
            }

        # Case 3: build from cumulative_stats raw fields
        revenue  = _safe_float(cs.get("revenue", 0.0), 0.0)
        holding  = _safe_float(cs.get("holding_cost", 0.0), 0.0)
        stockout = _safe_float(cs.get("stockout_penalty", 0.0), 0.0)
        profit   = revenue - holding - stockout
        eff      = _safe_float(cs.get("efficiency_score", 0.5), 0.5)
        return {
            "profit":          profit,
            "cost_efficiency": eff,
            "stockout_control":0.5,
            "decision_quality":0.5,
        }

    # Fallback
    return {"profit": 0.0, "cost_efficiency": 0.5, "stockout_control": 0.5, "decision_quality": 0.5}


def compute_weighted_score(profit_metrics, scenario):
    stats    = SCENARIO_STATS.get(scenario, SCENARIO_STATS["easy"])
    baseline = stats["baseline"]
    optimal  = stats["optimal"]
    profit   = _safe_float(profit_metrics.get("profit", 0.0), 0.0)

    span = optimal - baseline
    raw_profit_score = (profit - baseline) / span if span != 0 else 0.5
    raw_profit_score = _safe_float(raw_profit_score, 0.5)

    profit_score = max(0.011, min(0.989, raw_profit_score))
    cost_eff     = max(0.011, min(0.989, _safe_float(profit_metrics.get("cost_efficiency",  0.5), 0.5)))
    stock_ctrl   = max(0.011, min(0.989, _safe_float(profit_metrics.get("stockout_control", 0.5), 0.5)))
    dec_qual     = max(0.011, min(0.989, _safe_float(profit_metrics.get("decision_quality", 0.5), 0.5)))

    final_score = (
        0.40 * profit_score +
        0.25 * cost_eff     +
        0.20 * stock_ctrl   +
        0.15 * dec_qual
    )

    final_score = _safe_float(final_score, 0.5)
    return max(0.011, min(0.989, final_score))


def grade_easy(data):
    metrics = _extract_metrics(data)
    score   = compute_weighted_score(metrics, "easy")
    return max(0.011, min(0.989, float(score)))


def grade_medium(data):
    metrics = _extract_metrics(data)
    score   = compute_weighted_score(metrics, "medium")
    return max(0.011, min(0.989, float(score)))


def grade_hard(data):
    metrics = _extract_metrics(data)
    score   = compute_weighted_score(metrics, "hard")
    return max(0.011, min(0.989, float(score)))
