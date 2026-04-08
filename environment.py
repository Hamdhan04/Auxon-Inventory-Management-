import random
import numpy as np
from typing import Optional

random.seed(42)
np.random.seed(42)

from models import Observation, Action, Reward
from scenarios import get_scenario_config
from reward import compute_reward


class InventoryEnv:
    """
    Auxon Inventory Optimization - Multi-Product RL Environment
    """

    def __init__(self, scenario_name="easy"):
        self.scenario_name = scenario_name
        self.config = get_scenario_config(scenario_name)
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, seed: Optional[int] = None):
        """Initialize state with reproducibility"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)

        self.days_left = self.config['days']

        # 🔥 FIX: reset all metrics cleanly
        self.total_profit = 0.0
        self.cumulative_revenue = 0.0
        self.cumulative_holding_cost = 0.0
        self.cumulative_stockout_penalty = 0.0
        self.cumulative_stockouts = 0
        self.cumulative_overstock_units = 0

        self.products = {}

        for p_conf in self.config['products']:
            self.products[p_conf['id']] = {
                "conf": p_conf,
                "stock": 200,
                "price": p_conf['product_price'],
                "unmet_demand": 0,
                "history": []
            }

        return self._get_full_observation()

    def _get_product_observation(self, p_id):
        p = self.products[p_id]
        conf = p['conf']

        base = conf['demand_base']
        fluctuation = conf['demand_fluctuation']

        base_price = conf['product_price']
        price_ratio = p['price'] / base_price
        price_effect = (price_ratio - 1.0) * 2.0

        noise = self.rng.integers(-30, 61)
        fluct_offset = self.rng.integers(-fluctuation, fluctuation + 1) if fluctuation > 0 else 0

        demand_rate = max(0, base + noise + fluct_offset * (1.0 - price_effect))

        if self.config.get('event_chance') and self.rng.random() < self.config['event_chance']:
            demand_rate *= self.config.get('event_multiplier', 8)

        return Observation(
            product_id=p_id,
            current_stock=int(p['stock']),
            demand_rate=float(demand_rate),
            product_price=float(p['price']),
            storage_cost_per_unit=float(conf['storage_cost']),
            warehouse_capacity_remaining=int(conf['warehouse_capacity'] - p['stock']),
            days_left=int(self.days_left),
            unmet_demand=int(p['unmet_demand']),
            holding_cost=float(p['stock'] * conf['storage_cost'])
        )

    def _get_full_observation(self):
        # 🔥 FIX: return dict (important for FastAPI + OpenEnv)
        return {p_id: self._get_product_observation(p_id).dict() for p_id in self.products}

    def step(self, action: Action):
        if isinstance(action, dict):
            action = Action(**action)

        target_p_id = action.product_id

        # 1. Apply action
        if target_p_id in self.products:
            p = self.products[target_p_id]

            if action.action_type == 'restock':
                p['stock'] = min(p['conf']['warehouse_capacity'], p['stock'] + action.quantity)

            elif action.action_type == 'reduce_price':
                p['price'] = max(p['conf']['product_price'] * 0.5,
                                 p['price'] * (1 - action.percentage))

            elif action.action_type == 'increase_price':
                p['price'] = min(p['conf']['product_price'] * 5.0,
                                 p['price'] * (1 + action.percentage))

            elif action.action_type == 'transfer_warehouse':
                p['stock'] = max(0, p['stock'] - action.quantity)

        # 2. Advance simulation
        total_step_reward = 0
        total_step_breakdown = {
            "revenue": 0,
            "restock_cost": 0,
            "storage_cost": 0,
            "stockout_penalty": 0,
            "overstock_penalty": 0,
            "transfer_cost": 0
        }

        current_obs = {p_id: self._get_product_observation(p_id) for p_id in self.products}

        any_stockout = False

        for p_id, p in self.products.items():
            obs = current_obs[p_id]

            demand = obs.demand_rate
            demand_met = min(p['stock'], demand)

            p['unmet_demand'] = int(max(0, demand - p['stock']))
            p['stock'] -= int(demand_met)

            if p['unmet_demand'] > 0:
                any_stockout = True

            # Overstock tracking
            threshold = p['conf']['warehouse_capacity'] * 0.8
            if p['stock'] > threshold:
                self.cumulative_overstock_units += (p['stock'] - threshold)

            # Reward
            p_action = action if p_id == target_p_id else Action(product_id=p_id, action_type='do_nothing')
            reward_dict = compute_reward(obs.dict(), p_action.dict(), demand_met)
            reward_obj = Reward(**reward_dict)

            total_step_reward += reward_obj.value

            for key in total_step_breakdown:
                total_step_breakdown[key] += getattr(reward_obj, key)

            # 🔥 FIX: REAL PROFIT (not reward)
            actual_profit = reward_obj.revenue - (
                reward_obj.restock_cost +
                reward_obj.storage_cost +
                reward_obj.stockout_penalty +
                reward_obj.overstock_penalty +
                reward_obj.transfer_cost
            )

            self.total_profit += actual_profit
            self.cumulative_revenue += reward_obj.revenue
            self.cumulative_holding_cost += reward_obj.storage_cost
            self.cumulative_stockout_penalty += reward_obj.stockout_penalty

        if any_stockout:
            self.cumulative_stockouts += 1

        self.days_left -= 1
        done = self.days_left <= 0

        next_obs = self._get_full_observation()

        # -------- METRICS --------
        from scenarios import SCENARIO_STATS
        stats = SCENARIO_STATS.get(self.scenario_name, SCENARIO_STATS["easy"])

        total_days = stats["days"]
        current_step = max(1, total_days - self.days_left)  # 🔥 FIX

        total_costs = self.cumulative_holding_cost + self.cumulative_stockout_penalty + self.cumulative_overstock_units * 2.0

        cost_efficiency = self.cumulative_revenue / (self.cumulative_revenue + total_costs) \
            if (self.cumulative_revenue + total_costs) > 0 else 0.0

        stockout_control = 1.0 - (self.cumulative_stockouts / current_step)

        total_capacity = sum(p['conf']['warehouse_capacity'] for p in self.products.values()) * current_step

        decision_quality = 1.0 - (self.cumulative_overstock_units / total_capacity) \
            if total_capacity > 0 else 1.0

        profit_metrics = {
            "profit": float(self.total_profit),
            "cost_efficiency": float(cost_efficiency),
            "stockout_control": float(stockout_control),
            "decision_quality": float(decision_quality)
        }

        from scenarios import grade_easy, grade_medium, grade_hard

        if self.scenario_name == 'easy':
            efficiency_score = grade_easy(profit_metrics)
        elif self.scenario_name == 'medium':
            efficiency_score = grade_medium(profit_metrics)
        else:
            efficiency_score = grade_hard(profit_metrics)

        info = {
            "total_profit": float(self.total_profit),
            "step_reward_breakdown": {
                "profit_score": max(0.01, min(0.99, (self.total_profit - stats["baseline"]) / (stats["optimal"] - stats["baseline"] + 1e-6))),
                "cost_efficiency": max(0.01, min(0.99, cost_efficiency)),
                "stockout_control": max(0.01, min(0.99, stockout_control))
            },
            "cumulative_stats": {
                "revenue": float(self.cumulative_revenue),
                "holding_cost": float(self.cumulative_holding_cost),
                "stockout_penalty": float(self.cumulative_stockout_penalty),
                "stockout_days": self.cumulative_stockouts,
                "overstock_units": int(self.cumulative_overstock_units),
                "efficiency_score": float(efficiency_score),
                "profit_metrics": profit_metrics,
                "baseline_profit": float(stats["baseline"]),
                "optimal_profit": float(stats["optimal"])
            }
        }

        return next_obs, total_step_reward, done, info

    def state(self):
        return self._get_full_observation()
