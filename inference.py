import os
import random
import numpy as np
import math
from openai import OpenAI
from environment import InventoryEnv
from models import Action

# 1. Environment Variable Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    pass

def run_scenario(client, scenario_id, seed=42):
    env_name = "auxon-inventory-v1"
    
    print(f"[START] task={scenario_id} env={env_name} model={MODEL_NAME}")
    
    success = False
    step_count = 0
    rewards_history = []
    last_error = "null"
    
    try:
        env = InventoryEnv(scenario_name=scenario_id)
        obs_batch = env.reset(seed=seed)
        
        max_steps = env.config.get('days', 15)
        done = False
        
        while not done and step_count < max_steps:
            step_count += 1
            
            action_type = "do_nothing"
            try:
                if client:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": "Restock or do_nothing?"}],
                        max_tokens=10
                    )
                    ans = response.choices[0].message.content.lower()
                    action_type = "restock" if "restock" in ans else "do_nothing"
                else:
                    total_stock = sum(obs["current_stock"] for obs in obs_batch.values())
                    total_demand = sum(obs["demand_rate"] for obs in obs_batch.values())
                    if total_stock < total_demand:
                        action_type = "restock"
            except Exception as e:
                last_error = str(e).replace("\n", " ")
                action_type = "do_nothing"

            target_p_id = list(obs_batch.keys())[0]
            quantity = 0
            if action_type == "restock":
                quantity = max(100, int(obs_batch[target_p_id]["demand_rate"] * 5))
            
            action_obj = Action(product_id=target_p_id, action_type=action_type, quantity=quantity)
            
            obs_batch, step_reward, env_done, info = env.step(action_obj)
            
            is_last_step = (step_count == max_steps) or env_done
            
            if is_last_step:
                current_normalized_reward = info.get("cumulative_stats", {}).get("efficiency_score", 0.5)
                if math.isnan(float(current_normalized_reward)) or math.isinf(float(current_normalized_reward)):
                    current_normalized_reward = 0.5
                current_normalized_reward = max(0.011, min(0.989, float(current_normalized_reward)))
            else:
                step_profit = info.get("total_profit", 0)
                baseline = info.get("cumulative_stats", {}).get("baseline_profit", 800)
                optimal = info.get("cumulative_stats", {}).get("optimal_profit", 2200)
                raw = (step_profit - baseline) / (optimal - baseline + 1e-6)
                if math.isnan(raw) or math.isinf(raw):
                    raw = 0.5
                current_normalized_reward = max(0.011, min(0.989, float(raw)))
            
            rewards_history.append(current_normalized_reward)
            
            done_str = "true" if env_done else "false"
            print(f"[STEP] step={step_count} action={action_type} reward={current_normalized_reward:.2f} done={done_str} error={last_error}")

            if env_done:
                done = True
        
        success = True
        
    except Exception as e:
        last_error = str(e).replace("\n", " ")
        success = False
    
    formatted_rewards = ",".join([f"{r:.2f}" for r in rewards_history])
    
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={step_count} rewards={formatted_rewards}")

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable is missing.")
        return

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except:
        client = None

    seed = int(os.getenv("SEED", 42))
    random.seed(seed)
    np.random.seed(seed)

    for scenario in ["easy", "medium", "hard"]:
        run_scenario(client, scenario, seed=seed)

if __name__ == "__main__":
    main()
