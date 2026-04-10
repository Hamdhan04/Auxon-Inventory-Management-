import math

def compute_reward(state, action, demand_met):
    """
    Compute total reward and its components for Auxon Inventory Optimization.
    Ensures final reward is strictly within (0, 1).
    """
    
    # -----------------------------
    # 1. Revenue
    # -----------------------------
    revenue = demand_met * state['product_price']
    
    # -----------------------------
    # 2. Costs
    # -----------------------------
    restock_cost = 0
    if action['action_type'] == 'restock':
        restock_cost = action.get('quantity', 0) * 25.0
        
    storage_cost = state['current_stock'] * state['storage_cost_per_unit']
    
    # -----------------------------
    # 3. Penalties
    # -----------------------------
    demand = state.get('demand_rate', 0)
    
    stockout_penalty = max(0, demand - demand_met) * (state['product_price'] * 2.25)
    
    overstock_threshold = state.get('warehouse_capacity', 1000) * 0.8
    overstock_penalty = max(0, state['current_stock'] - overstock_threshold) * 2.6
    
    transfer_cost = 0
    if action['action_type'] == 'transfer_warehouse':
        transfer_cost = action.get('quantity', 0) * 1.5
    
    # -----------------------------
    # 4. Raw Reward
    # -----------------------------
    total_reward = (
        revenue
        - restock_cost
        - storage_cost
        - stockout_penalty
        - overstock_penalty
        - transfer_cost
    )
    
    # -----------------------------
    # 5. SAFE NORMALIZATION (CRITICAL FIX)
    # -----------------------------
    def normalize_reward(value):
        # Smooth scaling using tanh (handles large +/- values safely)
        normalized = (math.tanh(value / 1000) + 1) / 2
        
        # Ensure strictly between (0,1)
        if math.isnan(normalized) or math.isinf(normalized):
            normalized = 0.5
        
        return max(0.01, min(0.99, normalized))
    
    normalized_reward = normalize_reward(total_reward)
    
    # -----------------------------
    # 6. Return
    # -----------------------------
    return {
        "value": float(normalized_reward),   # ✅ REQUIRED for evaluation
        "raw_value": float(total_reward),    # 🔍 optional (debugging)
        "revenue": float(revenue),
        "restock_cost": float(restock_cost),
        "storage_cost": float(storage_cost),
        "stockout_penalty": float(stockout_penalty),
        "overstock_penalty": float(overstock_penalty),
        "transfer_cost": float(transfer_cost)
    }
