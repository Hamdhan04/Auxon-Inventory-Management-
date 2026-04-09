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

# SCENARIO PERFORMANCE THRESHOLDS
# Baseline: Random policy approx
# Optimal: Target profit for the scenario
SCENARIO_STATS = {
    "easy": {"baseline": 800, "optimal": 2200, "days": 15},
    "medium": {"baseline": 3500, "optimal": 12000, "days": 30},
    "hard": {"baseline": 12000, "optimal": 65000, "days": 50}
}

def compute_weighted_score(profit_metrics, scenario):
    """
    Computes final_score = 0.4 * profit_score + 0.25 * cost_efficiency + 0.2 * stockout_control + 0.15 * decision_quality
    Ensure no default 1.0 used.
    """
    stats = SCENARIO_STATS.get(scenario, SCENARIO_STATS["easy"])
    
    # 1. Profit Score (40%)
    baseline = stats["baseline"]
    optimal = stats["optimal"]
    profit = profit_metrics.get('profit', 0)
    raw_profit_score = (profit - baseline) / (optimal - baseline) if optimal > baseline else 0.0
    profit_score = max(0.001, min(0.999, raw_profit_score))  # strictly (0, 1)

    cost_eff = max(0.001, min(0.999, profit_metrics.get('cost_efficiency', 0)))
    stock_ctrl = max(0.001, min(0.999, profit_metrics.get('stockout_control', 0)))
    dec_qual = max(0.001, min(0.999, profit_metrics.get('decision_quality', 0)))
    
    # Weighted calculation
    final_score = (
        0.4 * profit_score +
        0.25 * cost_eff +
        0.2 * stock_ctrl +
        0.15 * dec_qual
    )
    
    # Final check: Score must be strictly between 0 and 1 (exclusive).
    if math.isnan(final_score) or math.isinf(final_score):
        final_score = 0.5

    final_score = max(0.01, min(0.99, final_score))
    return round(final_score, 4)

def grade_easy(profit_metrics):
    return compute_weighted_score(profit_metrics, "easy")

def grade_medium(profit_metrics):
    return compute_weighted_score(profit_metrics, "medium")

def grade_hard(profit_metrics):
    return compute_weighted_score(profit_metrics, "hard")
