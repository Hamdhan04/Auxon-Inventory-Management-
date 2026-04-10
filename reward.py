import math


def safe_score(x):
    """Clamp any float to strictly (0.01, 0.99) — never 0.0 or 1.0."""
    try:
        x = float(x)
    except Exception:
        x = 0.5
    if math.isnan(x) or math.isinf(x):
        return 0.5
    return max(0.01, min(0.99, x))


def normalize_reward(raw_reward, baseline=-500.0, optimal=2200.0):
    """
    Map a raw reward value into (0, 1) exclusive using a sigmoid-like linear normalization.
    baseline: worst expected total_reward value (e.g. heavy stockout run)
    optimal:  best expected total_reward value (e.g. perfect restock run)
    """
    span = optimal - baseline
    if span == 0:
        return 0.5
    normalized = (raw_reward - baseline) / span
    # Clamp strictly to (0.01, 0.99)
    return safe_score(normalized)


def compute_reward(state, action, demand_met):
    """
    Compute total reward and its components for Auxon Inventory Optimization.

    Args:
        state (dict): Current dictionary state.
        action (dict): Action taken.
        demand_met (int): Units sold in the current step.

    Returns:
        dict: Detailed reward components including a normalized score strictly in (0, 1).
    """
    revenue = demand_met * state['product_price']

    # Restock cost
    restock_cost = 0
    if action['action_type'] == 'restock':
        restock_cost = action.get('quantity', 0) * 25.0

    storage_cost = state['current_stock'] * state['storage_cost_per_unit']

    # Stockout penalty
    demand = state.get('demand_rate', 0)
    stockout_penalty = max(0, demand - demand_met) * (state['product_price'] * 2.25)

    # Overstock penalty
    overstock_threshold = state.get('warehouse_capacity', 1000) * 0.8
    overstock_penalty = max(0, state['current_stock'] - overstock_threshold) * 2.6

    # Transfer cost
    transfer_cost = 0
    if action['action_type'] == 'transfer_warehouse':
        transfer_cost = action.get('quantity', 0) * 1.5

    total_reward = (
        revenue
        - restock_cost
        - storage_cost
        - stockout_penalty
        - overstock_penalty
        - transfer_cost
    )

    # ✅ CRITICAL: normalize raw reward to strictly (0, 1) for grader
    score = normalize_reward(total_reward)

    return {
        "value": float(total_reward),
        "score": score,              # ← Use THIS as your task score, never raw "value"
        "revenue": float(revenue),
        "restock_cost": float(restock_cost),
        "storage_cost": float(storage_cost),
        "stockout_penalty": float(stockout_penalty),
        "overstock_penalty": float(overstock_penalty),
        "transfer_cost": float(transfer_cost),
    }
