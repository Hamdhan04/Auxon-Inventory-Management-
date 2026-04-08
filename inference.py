import os
import random
import numpy as np
from openai import OpenAI

random.seed(42)
np.random.seed(42)
from environment import InventoryEnv
from models import Action


def run_scenario(client, scenario, seed=42):
    """Run a single scenario to completion and return the final efficiency score."""
    env = InventoryEnv(scenario_name=scenario)
    obs_batch = env.reset(seed=seed)

    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

    print(f"[START] scenario={scenario}")

    done = False
    step_count = 1
    max_steps = env.config['days']  # Run for the full scenario duration
    info = {}

    while not done and step_count <= max_steps:
        total_stock = sum(obs["current_stock"] for obs in obs_batch.values())
        total_demand = sum(obs["demand_rate"] for obs in obs_batch.values())

        action_type = "do_nothing"  # Default

        # 🧠 ACTION SELECTION: Try LLM, fall back to heuristic
        try:
            if client is None:
                raise ValueError("Client not initialized")

            response = client.chat.completions.create(
                model=MODEL_NAME if MODEL_NAME else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Decide optimal inventory action: restock or do_nothing"}],
                max_tokens=20
            )
            llm_text = response.choices[0].message.content.lower()
            if "restock" in llm_text:
                action_type = "restock"
            else:
                action_type = "do_nothing"
        except Exception:
            # Fallback heuristic: restock if stock is running low
            if total_stock < total_demand:
                action_type = "restock"
            else:
                action_type = "do_nothing"

        print(f"[STEP] step={step_count} action={action_type}")

        # Pick the product with the lowest stock-to-demand ratio
        target_p_id = list(obs_batch.keys())[0]
        min_ratio = float('inf')
        for p_id, obs in obs_batch.items():
            ratio = obs["current_stock"] / obs["demand_rate"] if obs["demand_rate"] > 0 else 999
            if ratio < min_ratio:
                min_ratio = ratio
                target_p_id = p_id

        quantity = 0
        if action_type == "restock":
            tgt_obs = obs_batch[target_p_id]
            quantity = max(100, int(tgt_obs["demand_rate"] * 5))

        action_obj = Action(
            product_id=target_p_id,
            action_type=action_type,
            quantity=quantity
        )

        obs_batch, reward, env_done, info = env.step(action_obj)

        if env_done:
            done = True

        step_count += 1

    # Extract grader score — always strictly within (0, 1)
    efficiency_score = info.get("cumulative_stats", {}).get("efficiency_score", 0.5)

    print(f"[SCORE] scenario={scenario} score={efficiency_score:.4f}")
    print(f"[END] scenario={scenario}")

    return efficiency_score


def run_inference():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    HF_TOKEN = os.getenv("HF_TOKEN")

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        client = None

    seed = int(os.getenv("SEED", 42))

    # ✅ Run ALL 3 scenarios — platform requires at least 3 tasks with graders
    for scenario in ["easy", "medium", "hard"]:
        run_scenario(client, scenario, seed=seed)


if __name__ == "__main__":
    run_inference()
