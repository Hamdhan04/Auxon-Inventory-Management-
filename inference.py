import os
import random
import numpy as np
from openai import OpenAI

random.seed(42)
np.random.seed(42)
from environment import InventoryEnv
from models import Action

def run_inference():
    # 2. Read environment variables (with provided values as defaults)
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # 3. Initialize client
    # We initialize it here. If the key is missing, it might raise an error 
    # which we'll handle in the try-except block during the step loop.
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        client = None

    # Initialize environment
    scenario = os.getenv("SCENARIO", "medium")
    seed = int(os.getenv("SEED", 42))
    env = InventoryEnv(scenario_name=scenario)
    obs_batch = env.reset(seed=seed)

    # 📤 OUTPUT FORMAT (STRICT): Print [START]
    print("[START]")

    done = False
    step_count = 1
    max_steps = 50 # Max steps: 30–50

    while not done and step_count <= max_steps:
        # Calculate totals for fallback logic
        total_stock = sum(obs.current_stock for obs in obs_batch.values())
        total_demand = sum(obs.demand_rate for obs in obs_batch.values())

        action_type = "do_nothing" # Default

        # 🧠 ACTION SELECTION LOGIC: Hybrid approach
        try:
            if client is None:
                raise ValueError("Client not initialized")
                
            # 1. Try LLM call
            # Prompt: "Decide optimal inventory action: restock or do_nothing"
            response = client.chat.completions.create(
                model=MODEL_NAME if MODEL_NAME else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Decide optimal inventory action: restock or do_nothing"}],
                max_tokens=20
            )
            llm_text = response.choices[0].message.content.lower()

            # 2. Parse response: if contains "restock" → restock else → do_nothing
            if "restock" in llm_text:
                action_type = "restock"
            else:
                action_type = "do_nothing"
        except Exception:
            # 3. If LLM fails (no key or API error) → fallback heuristic
            # if total_stock < total_demand: restock else: do_nothing
            if total_stock < total_demand:
                action_type = "restock"
            else:
                action_type = "do_nothing"

        # 📤 OUTPUT FORMAT (STRICT): Print step log in required format
        print(f"[STEP] step={step_count} action={action_type}")

        # Execute action in environment
        # For multi-product, we pick the product with the lowest stock-to-demand ratio
        target_p_id = list(obs_batch.keys())[0]
        min_ratio = float('inf')
        for p_id, obs in obs_batch.items():
            ratio = obs.current_stock / obs.demand_rate if obs.demand_rate > 0 else 999
            if ratio < min_ratio:
                min_ratio = ratio
                target_p_id = p_id
        
        # Determine quantity for the restock action
        quantity = 0
        if action_type == "restock":
            # Simple heuristic: Restock 5 days worth of demand for the target product
            tgt_obs = obs_batch[target_p_id]
            quantity = max(100, int(tgt_obs.demand_rate * 5))
            
        action_obj = Action(
            product_id=target_p_id,
            action_type=action_type,
            quantity=quantity
        )

        # Call env.step(action)
        obs_batch, reward, env_done, info = env.step(action_obj)
        
        if env_done:
            done = True
            
        step_count += 1

    # 📤 OUTPUT FORMAT (STRICT): Print [END]
    print("[END]")

if __name__ == "__main__":
    run_inference()
