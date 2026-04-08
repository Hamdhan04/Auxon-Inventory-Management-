def compute_reward(state, action, demand_met):
    """
    Compute total reward and its components for Auxon Inventory Optimization.
    
    Args:
        state (dict): Current dictionary state.
        action (dict): Action taken.
        demand_met (int): Units sold in the current step.
    
    Returns:
        dict: Detailed reward components.
    """
    revenue = demand_met * state['product_price']
    
    # Costs
    restock_cost = 0
    if action['action_type'] == 'restock':
        # Assume cost of restock is 50% of base price
        restock_cost = action.get('quantity', 0) * 25.0
        
    storage_cost = state['current_stock'] * state['storage_cost_per_unit']
    
    # Penalties (Significantly increased for realism)
    demand = state.get('demand_rate', 0)
    # Lost sales + Reputation penalty (2.25x of price per unit of unmet demand)
    stockout_penalty = max(0, demand - demand_met) * (state['product_price'] * 2.25)
    
    # Overstock penalty (2.6x storage cost for any units above 80% capacity)
    overstock_threshold = state.get('warehouse_capacity', 1000) * 0.8
    overstock_penalty = max(0, state['current_stock'] - overstock_threshold) * 2.6
    
    transfer_cost = 0
    if action['action_type'] == 'transfer_warehouse':
        transfer_cost = action.get('quantity', 0) * 1.5  # $1.5 per unit transferred
    
    total_reward = revenue - restock_cost - storage_cost - stockout_penalty - overstock_penalty - transfer_cost
    
    return {
        "value": float(total_reward),
        "revenue": float(revenue),
        "restock_cost": float(restock_cost),
        "storage_cost": float(storage_cost),
        "stockout_penalty": float(stockout_penalty),
        "overstock_penalty": float(overstock_penalty),
        "transfer_cost": float(transfer_cost)
    }
