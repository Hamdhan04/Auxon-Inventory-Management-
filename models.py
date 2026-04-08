from pydantic import BaseModel
from typing import Optional, List

class Observation(BaseModel):
    product_id: str
    current_stock: int
    demand_rate: float
    product_price: float
    storage_cost_per_unit: float
    warehouse_capacity_remaining: int
    days_left: int
    unmet_demand: int
    holding_cost: float

class Action(BaseModel):
    product_id: str  # ID of the product this action applies to
    action_type: str  # restock, reduce_price, increase_price, transfer_warehouse, do_nothing
    quantity: int = 0
    percentage: float = 0.0

class Reward(BaseModel):
    value: float
    revenue: float
    restock_cost: float
    storage_cost: float
    stockout_penalty: float
    overstock_penalty: float
    transfer_cost: float
