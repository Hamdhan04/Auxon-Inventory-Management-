from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

from environment import InventoryEnv
from models import Action

app = FastAPI(title="Auxon Inventory Optimization - Multi-Product")

# Global environment instance
env = InventoryEnv(scenario_name="medium")

# Models for API
class StepRequest(BaseModel):
    product_id: str
    action_type: str
    quantity: int = 0
    percentage: float = 0.0

class ResetRequest(BaseModel):
    scenario: str = "medium"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/state")
async def get_state():
    state = env.state()
    # Convert Pydantic models to dict for JSON serialization
    serialized_state = state
    return {
        "products": serialized_state,
        "total_profit": env.total_profit,
        "scenario": env.scenario_name,
        "days_left": env.days_left
    }

@app.post("/step")
async def step_env(request: StepRequest):
    action = Action(
        product_id=request.product_id,
        action_type=request.action_type,
        quantity=request.quantity,
        percentage=request.percentage
    )
    next_obs, reward, done, info = env.step(action)
    
    serialized_next_obs = next_obs
    return {
        "next_state": serialized_next_obs,
        "reward": float(reward),
        "reward_breakdown": info.get("step_reward_breakdown"),
        "reasoning": f"User-initiated {request.action_type} for {request.product_id}",
        "done": done,
        "info": info
    }

@app.post("/agent-step")
async def agent_step():
    """Simple heuristic agent taking actions for a randomly chosen product each day."""
    state = env.state()
    product_ids = list(state.keys())
    # Pick a product that needs help most (lowest stock ratio)
    target_p_id = min(product_ids, key=lambda p_id: state[p_id]["current_stock"] / state[p_id]["demand_rate"] if state[p_id]["demand_rate"] > 0 else 999)
    obs = state[target_p_id]
    
    # Simple Heuristic with Reasoning
    reasoning = ""
    if obs['current_stock'] < obs['demand_rate'] * 2:
        action_type = "restock"
        quantity = int(obs['demand_rate'] * 5)
        percentage = 0.0
        reasoning = f"Demand ({obs['demand_rate']:.0f}) > Stock ({obs['current_stock']}), risk of stockout. Replenishing."
    elif obs['current_stock'] > 800:
        action_type = "reduce_price"
        quantity = 0
        percentage = 0.1
        reasoning = f"Stock ({obs['current_stock']}) is high (>800). Reducing price by 10% to stimulate demand."
    else:
        action_type = "do_nothing"
        quantity = 0
        percentage = 0.0
        reasoning = f"Stock level ({obs['current_stock']}) is healthy relative to demand ({obs['demand_rate']:.0f})."
        
    action = Action(product_id=target_p_id, action_type=action_type, quantity=quantity, percentage=percentage)
    next_obs, reward, done, info = env.step(action)
    
    serialized_next_obs = next_obs
    return {
        "action_taken": action.model_dump(),
        "next_state": serialized_next_obs,
        "reward": float(reward),
        "reward_breakdown": info.get("step_reward_breakdown"),
        "reasoning": reasoning,
        "done": done,
        "info": info
    }

@app.post("/reset")
async def reset_env():
    global env
    env = InventoryEnv(scenario_name="medium")
    obs = env.reset()
    serialized_obs = obs
    return {
        "observation": serialized_obs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
