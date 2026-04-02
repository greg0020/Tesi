# DRL Trading — Naphtha Crack Spread

Deep Reinforcement Learning (Double DQN) applied to algorithmic trading on the naphtha crack spread (Naphtha - Brent), compared against a classical mean reversion benchmark.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### 0. Generate Sample Data (optional)
```bash
python generate_sample_data.py
```

### 1. Prepare Data
Takes two separate CSVs (naphtha and brent prices), merges them, computes the crack spread, and generates features for **both legs** + cross-asset features so the agent can learn which leg is driving the spread.

```bash
python data/prepare_naphtha_data.py \
    --naphtha Data/naphtha_path.csv \
    --brent Data/brent_path.csv \
    --output_dir data \
    --train_ratio 0.8
```

### 2. Train Agent
```bash
python train.py --train_data data/naphtha_crack_train.csv --n_episodes 500 --reward_type pnl
```

### 3. Evaluate & Compare
```bash
python evaluate_and_compare.py --model_path results/run_XXXX/best_model.pt --test_data data/naphtha_crack_test.csv
```

## Architecture
- **Agent**: Double DQN with experience replay, epsilon-greedy exploration, target network
- **Environment**: Close-only trading env with 3 actions (Hold/Long/Short), trades on the crack spread
- **Features**: Per-leg features (naphtha & brent separately) + spread features + cross-asset features (correlation, beta, volatility ratio, contribution analysis)
- **Benchmark**: Z-score mean reversion + Buy & Hold
- **Rewards**: PnL, Sharpe, or Sortino ratio

## Key Insight
The agent receives features from **both legs** of the crack spread independently. This allows it to learn:
- Which leg (naphtha or brent) is driving the spread movement
- Rolling correlation and beta between the two
- Relative volatility and momentum differences
- Contribution of each leg to spread changes

## Files
| File | Role |
|------|------|
| `generate_sample_data.py` | Generates 500-row sample CSVs for testing |
| `data/prepare_naphtha_data.py` | Merges naphtha & brent CSVs, computes spread & features |
| `train.py` | Training loop |
| `drl_agent.py` | Double DQN agent |
| `trading_environment_close_only.py` | Trading environment |
| `mean_reversion_strategy.py` | Benchmark strategy |
| `evaluate_and_compare.py` | Evaluation & comparison |
| `timeSeriesAnalyser.py` | Time series analysis tools (ADF, ACF, decomposition) |
