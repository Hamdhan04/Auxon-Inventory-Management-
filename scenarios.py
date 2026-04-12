import random
import numpy as np
import math

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _strict(x, lo=0.001, hi=0.999):
    """Clamp x to (lo, hi) — strictly interior to (0, 1)."""
    try:
        x = float(x)
    except Exception:
        x = 0.5
    if math.isnan(x) or math.isinf(x):
        return 0.5
    return max(lo, min(hi, x))


# ──────────────────────────────────────────────────────────────────────────────
# Scenario configs
# ──────────────────────────────────────────────────────────────────────────────

def get_scenario_config(scenario_name):
    if scenario_name == 'easy':
        return easy_scenario()
    elif scenario_name == 'medium':
        return medium_scenario()
    elif scenario_name == 'hard':
        return hard_scenario()
    return easy_scenario()

def easy_scenario():
    """3 products in a stable environment (Auxon Basics)."""
    return {
        "days": 15,
        "products": [
            {"id": "AUX-HIGH-01", "name": "Standard Charger", "type": "high", "demand_base": 20, "demand_fluctuation": 2, "storage_cost": 0.1, "warehouse_capacity": 500, "product_price": 25.0},
            {"id": "AUX-LOW-02", "name": "Basic Mousepad", "type": "low", "demand_base": 5, "demand_fluctuation": 1, "storage_cost": 0.1, "warehouse_capacity": 500, "product_price": 15.0},
            {"id": "AUX-VAR-03", "name": "USB Cable Hub", "type": "fluctuating", "demand_base": 12, "demand_fluctuation": 5, "storage_cost": 0.1, "warehouse_capacity": 500, "product_price": 35.0}
        ]
    }

def medium_scenario():
    """3 products in a fluctuating environment (Auxon Seasonal)."""
    return {
        "days": 30,
        "products": [
            {"id": "AUX-HIGH-01", "name": "Essential Hoodie", "type": "high", "demand_base": 40, "demand_fluctuation": 10, "storage_cost": 0.5, "warehouse_capacity": 1000, "product_price": 45.0},
            {"id": "AUX-LOW-02", "name": "Dumbbells (5kg)", "type": "low", "demand_base": 10, "demand_fluctuation": 2, "storage_cost": 1.0, "warehouse_capacity": 500, "product_price": 60.0},
            {"id": "AUX-VAR-03", "name": "Water Lantern", "type": "fluctuating", "demand_base": 25, "demand_fluctuation": 20, "storage_cost": 0.5, "warehouse_capacity": 800, "product_price": 90.0}
        ]
    }

def hard_scenario():
    """3 products during Mega Sale Day / Festival spikes."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Scenario performance thresholds
# ──────────────────────────────────────────────────────────────────────────────

SCENARIO_STATS = {
    "easy":   {"baseline": 800,   "optimal": 2200,  "days": 15},
    "medium": {"baseline": 3500,  "optimal": 12000, "days": 30},
    "hard":   {"baseline": 12000, "optimal": 65000, "days": 50},
}


# ──────────────────────────────────────────────────────────────────────────────
# Weighted scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_weighted_score(profit_metrics, scenario):
    """
    final_score = 0.40 * profit_score
                + 0.25 * cost_efficiency
                + 0.20 * stockout_control
                + 0.15 * decision_quality

    Every sub-score and the final score are guaranteed to be strictly in (0.001, 0.999).
    """
    stats = SCENARIO_STATS.get(scenario, SCENARIO_STATS["easy"])

    baseline = float(stats["baseline"])
    optimal  = float(stats["optimal"])
    span     = optimal - baseline

    # Profit sub-score — guard against divide-by-zero
    profit = float(profit_metrics.get('profit', 0) or 0)
    if span > 0:
        raw_profit_score = (profit - baseline) / span
    else:
        raw_profit_score = 0.5
    profit_score = _strict(raw_profit_score)

    # Other sub-scores — use 0.5 fallback, NOT 0
    cost_eff   = _strict(profit_metrics.get('cost_efficiency')  or 0.5)
    stock_ctrl = _strict(profit_metrics.get('stockout_control') or 0.5)
    dec_qual   = _strict(profit_metrics.get('decision_quality') or 0.5)

    final_score = (
        0.40 * profit_score +
        0.25 * cost_eff +
        0.20 * stock_ctrl +
        0.15 * dec_qual
    )

    # Final NaN / inf guard + strict clamp
    if math.isnan(final_score) or math.isinf(final_score):
        final_score = 0.5

    return round(_strict(final_score), 6)


# ──────────────────────────────────────────────────────────────────────────────
# Metric extractor — handles all OpenEnv episode formats
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_METRICS = {
    "profit": 0.0,
    "cost_efficiency": 0.5,
    "stockout_control": 0.5,
    "decision_quality": 0.5,
}

def extract_metrics(*args, **kwargs):
    """
    Extract profit_metrics from whatever OpenEnv passes to the grader.

    Supported formats (tried in order):
      1. Plain dict with 'profit' key (direct profit_metrics dict)
      2. Episode object with .steps list  →  steps[-1].info['cumulative_stats']['profit_metrics']
      3. Dict                              →  dict['cumulative_stats']['profit_metrics']
      4. List of step objects              →  list[-1].info['cumulative_stats']['profit_metrics']
      5. Single Step object               →  step.info['cumulative_stats']['profit_metrics']
    Falls back to _FALLBACK_METRICS with neutral 0.5 sub-scores so the weighted
    score is never exactly 0.0 or 1.0.
    """
    obj = args[0] if args else kwargs.get('profit_metrics') or kwargs.get('episode')

    if obj is None:
        return dict(_FALLBACK_METRICS)

    # ── Format 1: already a profit_metrics dict ────────────────────────────
    if isinstance(obj, dict) and 'profit' in obj:
        return obj

    # ── Format 2: Episode object with .steps ──────────────────────────────
    try:
        steps = obj.steps
        if steps:
            info = steps[-1].info
            pm = info['cumulative_stats']['profit_metrics']
            if isinstance(pm, dict):
                return pm
    except Exception:
        pass

    # ── Format 3: Info dict  ───────────────────────────────────────────────
    try:
        pm = obj['cumulative_stats']['profit_metrics']
        if isinstance(pm, dict):
            return pm
    except Exception:
        pass

    # ── Format 4: List of step objects ────────────────────────────────────
    try:
        if isinstance(obj, (list, tuple)) and obj:
            info = obj[-1].info
            pm = info['cumulative_stats']['profit_metrics']
            if isinstance(pm, dict):
                return pm
    except Exception:
        pass

    # ── Format 5: Single step object with .info ───────────────────────────
    try:
        info = obj.info
        pm = info['cumulative_stats']['profit_metrics']
        if isinstance(pm, dict):
            return pm
    except Exception:
        pass

    # ── Format 6: List of (obs, action, reward, done, info) tuples ────────
    try:
        if isinstance(obj, (list, tuple)) and obj:
            last = obj[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 5:
                info = last[4]
                pm = info['cumulative_stats']['profit_metrics']
                if isinstance(pm, dict):
                    return pm
    except Exception:
        pass

    return dict(_FALLBACK_METRICS)


# ──────────────────────────────────────────────────────────────────────────────
# Public graders (called by OpenEnv via openenv.yaml)
# ──────────────────────────────────────────────────────────────────────────────

def grade_easy(*args, **kwargs):
    """Grade the easy (Auxon Basics) task. Returns a float strictly in (0, 1)."""
    try:
        metrics = extract_metrics(*args, **kwargs)
        score   = compute_weighted_score(metrics, "easy")
        return _strict(score)
    except Exception:
        return 0.5


def grade_medium(*args, **kwargs):
    """Grade the medium (Seasonal Demand) task. Returns a float strictly in (0, 1)."""
    try:
        metrics = extract_metrics(*args, **kwargs)
        score   = compute_weighted_score(metrics, "medium")
        return _strict(score)
    except Exception:
        return 0.5


def grade_hard(*args, **kwargs):
    """Grade the hard (Mega Sale Day) task. Returns a float strictly in (0, 1)."""
    try:
        metrics = extract_metrics(*args, **kwargs)
        score   = compute_weighted_score(metrics, "hard")
        return _strict(score)
    except Exception:
        return 0.5
