# Auxon Inventory Optimization — RL Environment

This repository contains a complete, production-ready Reinforcement Learning (RL) environment for optimizing inventory management in an e-commerce context.

## 🎯 Project Goal
The objective is to maximize total profit while balancing stock levels, pricing, and warehouse capacity constraints. The agent must navigate stochastic demand influenced by pricing decisions.

---

## 🧠 RL Design

### State (Observation)
Each step returns a `Observation` model:
- `product_id`: Unique identifier
- `current_stock`: Units available
- `demand_rate`: Predicted demand for the step
- `product_price`: Current selling price
- `storage_cost_per_unit`: Cost to hold one unit
- `warehouse_capacity_remaining`: Available space
- `days_left`: Time steps remaining in episode
- `unmet_demand`: Stockout quantity from previous step
- `holding_cost`: Total cost for current stock

### Action Space
- `restock(quantity)`: Increase inventory (at 50% base price cost)
- `reduce_price(percentage)`: Increase demand, lower margin
- `increase_price(percentage)`: Decrease demand, higher margin
- `transfer_warehouse(quantity)`: Move stock (at fixed cost)
- `do_nothing`: No change

### Reward Function
`Reward = Sales_Revenue - Restock_Cost - Storage_Cost - Stockout_Penalty - Overstock_Penalty - Transfer_Cost`

---

## 🧪 Scenarios & Grading
The environment supports three difficulty levels in `scenarios.py`:
1. **Easy (Auxon Basics)**: Low fluctuation, stable demand.
2. **Medium (Seasonal)**: Higher demand variance.
3. **Hard (Mega Sale Day/Festival)**: Extreme demand spikes and tight capacity.

**Grading**: Each scenario calculates a normalized score (0 to 1) based on cumulative profit targets.

---

## 🚀 Setup & Execution

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Dashboard:
   ```bash
   python app.py
   ```
3. Open `http://localhost:8000` in your browser.

### Docker
```bash
docker build -t auxon-inventory-rl .
docker run -p 8000:8000 auxon-inventory-rl
```

### Inference Log Format
To run a strict-format inference (for OpenEnv evaluation):
```bash
python inference.py
```

---

## 🧩 Technical Stack
- **Backend**: FastAPI, Pydantic, NumPy
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphic), JavaScript
- **Visualization**: Chart.js
- **Spec**: OpenEnv v1.0
