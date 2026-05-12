"""
Valutazione dell'agente DRL e confronto con la strategia di mean reversion benchmark.
Produce metriche, tabelle e grafici comparativi.
"""

import numpy as np
import pandas as pd
import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_1samp


from trading_environment_close_only import TradingEnvironmentCloseOnly
from drl_agent import DRLAgent
from mean_reversion_strategy import MeanReversionStrategy


def evaluate_drl_agent(model_path: str, data_path: str, window_size: int = 20,
                       initial_balance: float = 1.0,
                       transaction_cost: float = 0.0,
                       feature_mean=None,
                       feature_std=None,
                       strategy_type: str = 'drl') -> dict:
    """Valuta l'agente DRL sul dataset di test."""
    env = TradingEnvironmentCloseOnly(
        data_path=data_path,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        reward_type='pnl',
        feature_mean=feature_mean,
        feature_std=feature_std,
        strategy_type=strategy_type,
    )

    agent = DRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_n,
    )
    agent.load(model_path)

    state = env.reset()
    total_reward = 0.0
    actions_taken = []
    positions_taken = []

    while True:
        action = agent.select_action(state, training=False)
        actions_taken.append(action)

        next_state, reward, done, info = env.step(action)

        positions_taken.append(info["position"])
        total_reward += reward
        state = next_state

        if done:
            break

    metrics = env.get_metrics()
    metrics['total_reward'] = float(total_reward)
    metrics['portfolio_values'] = [float(v) for v in env.portfolio_values]
    metrics['daily_returns'] = [float(r) for r in env.daily_returns]
    metrics['actions'] = actions_taken
    metrics['positions'] = positions_taken
    metrics['trades'] = env.trades

    return metrics


def evaluate_benchmark(data_path: str,
                       lookback: int = 20,
                       entry_threshold: float = 2.3,
                       initial_balance: float = 1.0,
                       transaction_cost: float = 0.0) -> dict:
    """Valuta la strategia di mean reversion benchmark."""
    strategy = MeanReversionStrategy(
        lookback=lookback,
        entry_threshold=entry_threshold,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost
    )
    return strategy.backtest(data_path)


def compute_drawdown(portfolio_values):
    """Calcola il drawdown dato un vettore di portfolio values."""
    pv = np.array(portfolio_values, dtype=float)
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / (peak + 1e-8)
    return drawdown


def compute_annual_metrics(dates, metrics):
    """Calcola metriche annuali a partire dai daily returns."""
    n = min(len(dates), len(metrics["daily_returns"]))

    df = pd.DataFrame({
        "Date": pd.to_datetime(dates[:n]),
        "return": np.array(metrics["daily_returns"][:n], dtype=float)
    })

    df["year"] = df["Date"].dt.year

    rows = []
    for year, g in df.groupby("year"):
        rets = g["return"].values

        total_return = np.prod(1 + rets) - 1
        volatility = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)

        downside = rets[rets < 0]
        downside_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
        sortino = (rets.mean() / downside_std) * np.sqrt(252)

        rows.append({
            "year": int(year),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "volatility": float(volatility)
        })

    return pd.DataFrame(rows)

def get_positions(metrics):
    """
    Estrae le posizioni se disponibili.
    Se mancano, restituisce None.
    """
    if "positions" in metrics:
        return np.array(metrics["positions"], dtype=float)
    if "signals" in metrics:
        return np.array(metrics["signals"], dtype=float)
    return None

def plot_position_regime_timeline(drl_metrics, mr_metrics, dates, data_path, save_dir, brent_col="Brent_Close", window=20):

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    brent_ret = df[brent_col].pct_change()
    brent_vol = brent_ret.rolling(window).std() * np.sqrt(252)
    threshold = brent_vol.median()
    high_vol = brent_vol > threshold

    drl_pos = get_positions(drl_metrics)
    mr_pos = get_positions(mr_metrics)

    n = min(len(drl_pos), len(mr_pos), len(dates), len(high_vol))

    d = pd.to_datetime(np.array(dates)[:n])
    drl_pos = drl_pos[:n]
    mr_pos = mr_pos[:n]
    high_vol = high_vol.iloc[:n].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    for ax, pos, title in [
        (axes[0], drl_pos, "DRL Agent Position"),
        (axes[1], mr_pos, "Mean Reversion Position")
    ]:
        for i in range(n):
            if high_vol[i]:
                ax.axvspan(d[i], d[i], alpha=0.15)

        ax.step(d, pos, where="post", linewidth=0.9)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Short", "Flat", "Long"])
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Position Regime Timeline with Brent Volatility Regimes")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "position_regime_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()




def get_trade_indices_from_positions(positions):
    """
    Identifica i punti in cui la posizione cambia.
    """
    positions = np.array(positions, dtype=float)
    if len(positions) < 2:
        return np.array([], dtype=int)
    return np.where(np.diff(positions) != 0)[0] + 1


def compute_trade_returns_from_positions(daily_returns, positions):
    """
    Calcola il rendimento cumulato per ogni trade.
    Un trade viene definito come un periodo consecutivo con posizione diversa da zero.
    """
    rets = np.array(daily_returns, dtype=float)
    pos = np.array(positions, dtype=float)

    n = min(len(rets), len(pos))
    rets = rets[:n]
    pos = pos[:n]

    trade_returns = []
    holding_periods = []

    in_trade = False
    trade_rets = []

    for i in range(n):
        if pos[i] != 0:
            in_trade = True
            trade_rets.append(rets[i])
        else:
            if in_trade and len(trade_rets) > 0:
                trade_return = np.prod(1 + np.array(trade_rets)) - 1
                trade_returns.append(trade_return)
                holding_periods.append(len(trade_rets))
                trade_rets = []
                in_trade = False

    if in_trade and len(trade_rets) > 0:
        trade_return = np.prod(1 + np.array(trade_rets)) - 1
        trade_returns.append(trade_return)
        holding_periods.append(len(trade_rets))

    return np.array(trade_returns), np.array(holding_periods)


    """
    Calcola i cumulative returns solo nei giorni in cui la strategia è in posizione.
    Utile per rimuovere i tratti piatti della MR.
    """
    rets = np.array(daily_returns, dtype=float)
    pos = np.array(positions, dtype=float)

    n = min(len(rets), len(pos))
    rets = rets[:n]
    pos = pos[:n]

    active_rets = rets[pos != 0]

    if len(active_rets) == 0:
        return np.array([])

    return np.cumprod(1 + active_rets) - 1


def compute_rolling_exposure(positions, window=60):
    """
    Exposure rolling: percentuale media di giorni in posizione.
    """
    pos = pd.Series(np.abs(np.array(positions, dtype=float)))
    return pos.rolling(window).mean()

    # ============================================================


def compute_trade_analysis(metrics):

    trades = metrics["trades"]

    closed_trades = [
        t for t in trades
        if t["type"].startswith("close")
        or t["type"] == "force_close"
    ]

    if len(closed_trades) == 0:
        return None

    pnls = np.array([t["pnl"] for t in closed_trades], dtype=float)

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    hit_rate = len(wins) / len(pnls)
    loss_rate = len(losses) / len(pnls)

    avg_win = wins.mean()  
    avg_loss = losses.mean() 

    expected_trade_return = hit_rate * avg_win + loss_rate * avg_loss

    analysis = {
    "hit_rate":
        hit_rate,
    "loss_rate":
        loss_rate,
    "avg_win":
        avg_win,
    "avg_loss":
        avg_loss,
    "expected_trade_return":
        expected_trade_return
}

    return analysis


def generate_drl_empirical_plots(drl_metrics,mr_metrics, dates, save_dir):

    os.makedirs(save_dir, exist_ok=True)


    plt.figure(figsize=(12, 5))

    for label, metrics in [
    ("DRL Agent", drl_metrics),
    ("Mean Reversion", mr_metrics)
    ]:

        pv = np.array(
        metrics["portfolio_values"],
        dtype=float
    )

        pv_norm = pv / pv[0]

        d = pd.to_datetime(
            dates[:len(pv_norm)]
    )

        plt.plot(
        d,
        pv_norm,
        linewidth=1.7,
        label=label
    )

    plt.title("Normalized Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
    os.path.join(save_dir, "equity_curve.png"),
    dpi=150,
    bbox_inches="tight"
)

    plt.close()
    
    
    # ============================================================
    # PREPARE DATA
    # ============================================================

    returns = pd.Series(
        np.array(drl_metrics["daily_returns"], dtype=float),
        index=pd.to_datetime(dates[:len(drl_metrics["daily_returns"])])
    )

    portfolio_values = np.array(
        drl_metrics["portfolio_values"],
        dtype=float
    )

    pv_dates = pd.to_datetime(dates[:len(portfolio_values)])

    cum_returns = pd.Series(
        portfolio_values / portfolio_values[0],
        index=pv_dates
    )

    # ============================================================
    # 1. OUT-OF-SAMPLE EQUITY CURVE
    # ============================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        cum_returns.index,
        cum_returns.values,
        linewidth=1.8,
        label="DRL Strategy"
    )

    plt.title("Out-of-Sample Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "drl_equity_curve.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # 2. DRAWDOWN PROFILE
    # ============================================================

    rolling_max = cum_returns.cummax()

    drawdown = (cum_returns - rolling_max) / rolling_max

    plt.figure(figsize=(12, 5))

    plt.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        alpha=0.4
    )

    plt.title("Drawdown Profile")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.gca().yaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "drl_drawdown_profile.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # 3. ROLLING SHARPE RATIO
    # ============================================================

    window = 126

    rolling_mean = returns.rolling(window).mean() * 252

    rolling_std = returns.rolling(window).std() * np.sqrt(252)

    rolling_sharpe = rolling_mean / (rolling_std + 1e-8)

    plt.figure(figsize=(12, 5))

    plt.plot(
        rolling_sharpe.index,
        rolling_sharpe.values,
        linewidth=1.5
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.title("Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "drl_rolling_sharpe.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # 4. ANNUAL RETURNS
    # ============================================================

    annual_returns = returns.groupby(
        returns.index.year
    ).apply(
        lambda x: (1 + x).prod() - 1
    )

    plt.figure(figsize=(12, 5))

    annual_returns.plot(kind="bar")

    plt.title("Annual Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")

    plt.gca().yaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "drl_annual_returns.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # 5. ANNUAL METRICS TABLE
    # ============================================================

    annual_metrics = []

    for year, g in returns.groupby(returns.index.year):

        rets = g.values

        total_return = np.prod(1 + rets) - 1

        volatility = np.std(rets) * np.sqrt(252)

        sharpe = (
            np.mean(rets)
            / (np.std(rets) + 1e-8)
        ) * np.sqrt(252)

        cum = (1 + g).cumprod()

        rolling_max = cum.cummax()

        dd = (cum - rolling_max) / rolling_max

        max_dd = dd.min()

        annual_metrics.append({
            "Year": int(year),
            "Return": total_return,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "Max_Drawdown": max_dd
        })

    annual_metrics_df = pd.DataFrame(annual_metrics)

    annual_metrics_df.to_csv(
        os.path.join(save_dir, "drl_annual_metrics.csv"),
        index=False
    )

    print("\nAnnual Metrics")
    print(annual_metrics_df)

    print("\nPlots saved in:")
    print(save_dir)
    

def compute_annual_metrics_single_strategy(dates, metrics):

    n = min(len(dates), len(metrics["daily_returns"]))

    df = pd.DataFrame({
        "Date": pd.to_datetime(dates[:n]),
        "return": np.array(metrics["daily_returns"][:n], dtype=float)
    })

    df["year"] = df["Date"].dt.year

    rows = []

    for year, g in df.groupby("year"):

        rets = g["return"].values

        total_return = np.prod(1 + rets) - 1

        volatility = np.std(rets) * np.sqrt(252)

        sharpe = (
            np.mean(rets)
            / (np.std(rets) + 1e-8)
        ) * np.sqrt(252)

        downside = rets[rets < 0]

        downside_std = (
            downside.std() + 1e-8
            if len(downside) > 0
            else 1e-8
        )

        sortino = (
            np.mean(rets)
            / downside_std
        ) * np.sqrt(252)

        cum = np.cumprod(1 + rets)

        peak = np.maximum.accumulate(cum)

        drawdown = (cum - peak) / peak

        max_dd = drawdown.min()

        rows.append({
            "Year": int(year),
            "Return": total_return,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max_Drawdown": max_dd
        })

    return pd.DataFrame(rows)
    

def compute_signal_conditioning_analysis(metrics, data_path):

    df = pd.read_csv(data_path)

    feature_cols = [
    "shock",
    "deviation_from_mean",
    "zscore_crack",
    "shock_x_volume",
    "shock_lag1",
    "half_life_proxy"
]

    global_means = df[feature_cols].mean()

    feature_cols = [
        "shock",
        "deviation_from_mean",
        "zscore_crack",
        "shock_x_volume",
        "shock_lag1",
        "half_life_proxy"
    ]

    trades = metrics["trades"]

    closed_trades = [
        t for t in trades
        if t["type"].startswith("close")
        or t["type"] == "force_close"
    ]

    rows = []

    for t in closed_trades:

        step = t["step"]

        if step >= len(df):
            continue

        row = {
            "pnl": t["pnl"],
            "winning_trade": 1 if t["pnl"] > 0 else 0
        }

        for f in feature_cols:
            row[f] = df.iloc[step][f]

        rows.append(row)

    signal_df = pd.DataFrame(rows)

    grouped = signal_df.groupby("winning_trade")[feature_cols].mean()

    grouped.index = ["Losing Trades", "Winning Trades"]

    result = pd.concat([global_means.rename("Global Mean"),grouped.T], axis=1)

    return result

def compute_regime_metrics(data_path, drl_metrics, mr_metrics, dates, brent_col="Brent_Close", window=20):
    """
    Divide il test set in regimi di alta e bassa volatilità Brent.
    """
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    if brent_col not in df.columns:
        return None

    df["brent_return"] = df[brent_col].pct_change()
    df["brent_vol"] = df["brent_return"].rolling(window).std() * np.sqrt(252)

    median_vol = df["brent_vol"].median()

    df["regime"] = np.where(df["brent_vol"] >= median_vol, "High Brent Volatility", "Low Brent Volatility")

    n = min(len(df), len(drl_metrics["daily_returns"]), len(mr_metrics["daily_returns"]))

    df = df.iloc[:n].copy()
    df["drl_return"] = np.array(drl_metrics["daily_returns"][:n], dtype=float)
    df["mr_return"] = np.array(mr_metrics["daily_returns"][:n], dtype=float)

    rows = []

    for regime, g in df.groupby("regime"):

        for strategy_name, col in [("DRL Agent", "drl_return"), ("Mean Reversion", "mr_return")]:

            rets = g[col].dropna().values

            if len(rets) == 0:
                continue

            total_return = np.prod(1 + rets) - 1
            vol = rets.std() * np.sqrt(252)
            sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)

            rows.append({
                "regime": regime,
                "strategy": strategy_name,
                "total_return": total_return,
                "volatility": vol,
                "sharpe_ratio": sharpe
            })

    return pd.DataFrame(rows)

def compute_long_short_signal_analysis(metrics, data_path):

    df = pd.read_csv(data_path)

    feature_cols = [
        "shock",
        "deviation_from_mean",
        "zscore_crack",
        "shock_x_volume",
        "shock_lag1",
        "half_life_proxy"
    ]

    trades = metrics["trades"]

    closed_trades = [
        t for t in trades
        if t["type"].startswith("close")
        or t["type"] == "force_close"
    ]

    rows = []

    for t in closed_trades:

        step = t["step"]

        if step >= len(df):
            continue

        if "long" in t["type"]:
            signal_type = "Long"

        elif "short" in t["type"]:
            signal_type = "Short"

        else:
            continue

        row = {
            "signal_type": signal_type
        }

        for f in feature_cols:
            row[f] = df.iloc[step][f]

        rows.append(row)

    signal_df = pd.DataFrame(rows)

    grouped = signal_df.groupby("signal_type")[feature_cols].mean()

    return grouped

def plot_comparison(drl_metrics: dict,
                    mr_metrics: dict,
                    dates,
                    save_dir: str,
                    data_path: str = None):

    generate_drl_empirical_plots(
        drl_metrics=drl_metrics,
        mr_metrics=mr_metrics,
        dates=dates,
        save_dir=save_dir
    )
    

def save_comparison_table(drl_metrics: dict,
                          mr_metrics: dict,
                          output_dir: str):
    """Salva tabella comparativa delle metriche principali."""
    comparison_rows = []

    for key, label in [
        ('total_return', 'Total Return'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('sortino_ratio', 'Sortino Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('n_trades', 'N. Trades'),
        ('win_rate', 'Win Rate'),
        ('final_balance', 'Final Balance')
    ]:
        row = {'Metric': label}
        for name, m in [('DRL_Agent', drl_metrics), ('Mean_Reversion', mr_metrics)]:
            row[name] = m.get(key, 'N/A')
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(output_dir, 'comparison_table.csv')
    comp_df.to_csv(comp_csv_path, index=False)

    return comp_df


def save_portfolio_values(drl_metrics: dict,
                          mr_metrics: dict,
                          dates,
                          output_dir: str):
    """Salva portfolio values su CSV."""
    max_len = max(
        len(drl_metrics['portfolio_values']),
        len(mr_metrics['portfolio_values'])
    )

    def pad_list(lst, length):
        return lst + [None] * (length - len(lst))

    pv_df = pd.DataFrame({
        'Date': pad_list(list(dates), max_len),
        'step': range(max_len),
        'DRL_Agent': pad_list(drl_metrics['portfolio_values'], max_len),
        'Mean_Reversion': pad_list(mr_metrics['portfolio_values'], max_len)
    })

    pv_csv_path = os.path.join(output_dir, 'portfolio_values.csv')
    pv_df.to_csv(pv_csv_path, index=False)

    return pv_df


def save_summary(drl_metrics: dict,
                 mr_metrics: dict,
                 args,
                 output_dir: str):
    """Salva summary testuale."""
    summary_lines = [
        "=" * 60,
        "EVALUATION SUMMARY",
        "=" * 60,
        f"Test data: {args.test_data}",
        f"Model: {args.model_path}",
        f"Initial balance: {args.initial_balance:,.4f}",
        f"Transaction cost: {args.transaction_cost}",
        "",
    ]

    for name, m in [('DRL Agent', drl_metrics), ('Mean Reversion', mr_metrics)]:
        summary_lines.append(f"--- {name} ---")
        summary_lines.append(f"  Total Return:  {m.get('total_return', 0):.2%}")
        summary_lines.append(f"  Sharpe Ratio:  {m.get('sharpe_ratio', 0):.3f}")
        summary_lines.append(f"  Sortino Ratio: {m.get('sortino_ratio', 0):.3f}")
        summary_lines.append(f"  Max Drawdown:  {m.get('max_drawdown', 0):.2%}")
        summary_lines.append(f"  N. Trades:     {m.get('n_trades', 'N/A')}")
        summary_lines.append(f"  Win Rate:      {m.get('win_rate', 0):.2%}")
        summary_lines.append(f"  Final Balance: {m.get('final_balance', 0):,.4f}")
        summary_lines.append("")

    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)

    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    return summary_text

def compute_action_conditioning_and_surrogate(metrics, data_path, output_dir):

    df = pd.read_csv(data_path)

    feature_cols = [
        "shock",
        "deviation_from_mean",
        "zscore_crack",
        "shock_x_volume",
        "shock_lag1",
        "half_life_proxy"
    ]

    positions = np.array(metrics["positions"], dtype=int)

    n = min(len(df), len(positions))

    X = df[feature_cols].iloc[:n].copy()
    y = positions[:n]

    data = X.copy()
    data["position"] = y

    label_map = {
        -1: "Short",
        0: "Flat",
        1: "Long"
    }

    data["position_label"] = data["position"].map(label_map)

    # ============================================================
    # 1. ACTION / POSITION CONDITIONING TABLE
    # ============================================================

    conditioning = data.groupby("position_label")[feature_cols].mean()

    conditioning.to_csv(
        os.path.join(output_dir, "action_conditioning_analysis.csv")
    )

    # ============================================================
    # 2. MULTINOMIAL LOGISTIC SURROGATE MODEL
    # ============================================================

    model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

    model.fit(X, y)


    importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

    importance_df.to_csv(
        os.path.join(output_dir, "surrogate_logistic_coefficients.csv")
    )

    accuracy = model.score(X, y)

    acc_df = pd.DataFrame([{
        "surrogate_accuracy": accuracy
    }])

    acc_df.to_csv(
        os.path.join(output_dir, "surrogate_logistic_accuracy.csv"),
        index=False
    )

    

    return conditioning, importance_df, accuracy

def compute_statistical_significance(metrics, output_dir, n_bootstrap=5000):

    trades = metrics["trades"]

    closed_trades = [
        t for t in trades
        if t["type"].startswith("close")
        or t["type"] == "force_close"
    ]

    if len(closed_trades) == 0:
        return None

    trade_returns = np.array(
        [t["pnl"] for t in closed_trades],
        dtype=float
    )

    # ============================================================
    # BASIC STATISTICS
    # ============================================================

    mean_trade_return = np.mean(trade_returns)

    std_trade_return = np.std(trade_returns)

    years = 5.25

    trades_per_year = len(trade_returns) / years

    sharpe_trade = (mean_trade_return/ (std_trade_return )) * np.sqrt(trades_per_year)

    # ============================================================
    # T-TEST
    # H0: expected trade return = 0
    # ============================================================

    t_stat, p_value = ttest_1samp(
        trade_returns,
        popmean=0.0
    )

    # ============================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ============================================================

    bootstrap_means = []

    bootstrap_sharpes = []

    n = len(trade_returns)

    for _ in range(n_bootstrap):

        sample = np.random.choice(
            trade_returns,
            size=n,
            replace=True
        )

        sample_mean = np.mean(sample)

        sample_std = np.std(sample)

        sample_trades_per_year = len(sample) / years

        sample_sharpe = (sample_mean/ (sample_std)) * np.sqrt(sample_trades_per_year)

        bootstrap_means.append(sample_mean)

        bootstrap_sharpes.append(sample_sharpe)

    mean_ci = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    sharpe_ci = np.percentile(
        bootstrap_sharpes,
        [2.5, 97.5]
    )

    # ============================================================
    # RESULTS TABLE
    # ============================================================

    results = pd.DataFrame([{

        "mean_trade_return":
            mean_trade_return,

        "mean_trade_return_ci_lower":
            mean_ci[0],

        "mean_trade_return_ci_upper":
            mean_ci[1],

        "trade_sharpe":
            sharpe_trade,

        "trade_sharpe_ci_lower":
            sharpe_ci[0],

        "trade_sharpe_ci_upper":
            sharpe_ci[1],

        "t_statistic":
            t_stat,

        "p_value":
            p_value,

        "n_trades":
            n

    }])

    results.to_csv(
        os.path.join(
            output_dir,
            "statistical_significance_analysis.csv"
        ),
        index=False
    )

    print("\nSTATISTICAL SIGNIFICANCE ANALYSIS")
    print(results)

    return results

def main():
    parser = argparse.ArgumentParser(description='Valutazione e confronto strategie')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Percorso al modello DRL (.pt)')
    parser.add_argument('--test_data', type=str, default='Data/naphtha_crack_test.csv',
                        help='CSV dati di test')
    parser.add_argument('--output_dir', type=str, default='Data/evaluation_results',
                        help='Cartella output')

    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--initial_balance', type=float, default=1.0)
    parser.add_argument('--transaction_cost', type=float, default=0.00001)

    parser.add_argument('--mr_lookback', type=int, default=20,
                        help='Lookback mean reversion')
    parser.add_argument('--mr_entry', type=float, default=1.8,
                        help='Entry threshold mean reversion')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Carica configurazione training
    # ============================================================
    model_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(model_dir, 'config.json')

    with open(config_path, 'r') as f:
        train_config = json.load(f)

    feature_mean_path = os.path.join(model_dir, 'feature_mean.npy')
    feature_std_path = os.path.join(model_dir, 'feature_std.npy')

    feature_mean = np.load(feature_mean_path)
    feature_std = np.load(feature_std_path)

    strategy_type = train_config.get('strategy_type', 'drl')

    # ============================================================
    # Carica date test
    # ============================================================
    test_df = pd.read_csv(args.test_data)
    dates = test_df["Date"].values

    # ============================================================
    # Valuta DRL
    # ============================================================
    drl_metrics = evaluate_drl_agent(
        model_path=args.model_path,
        data_path=args.test_data,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        feature_mean=feature_mean,
        feature_std=feature_std,
        strategy_type=strategy_type,
    )

    # ============================================================
    # Valuta Mean Reversion
    # ============================================================
    mr_metrics = evaluate_benchmark(
        data_path=args.test_data,
        lookback=args.mr_lookback,
        entry_threshold=args.mr_entry,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost
    )

    # ============================================================
    # Salva risultati JSON
    # ============================================================
    results = {
        'drl_agent': {
            k: v for k, v in drl_metrics.items()
            if k not in ['portfolio_values', 'daily_returns', 'actions', 'positions', 'trades']
        },
        'mean_reversion': {
            k: v for k, v in mr_metrics.items()
            if k not in ['portfolio_values', 'daily_returns', 'signals', 'trades']
        }
    }

    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ============================================================
    # Salva tabelle
    # ============================================================
    comp_df = save_comparison_table(drl_metrics, mr_metrics, args.output_dir)
    save_portfolio_values(drl_metrics, mr_metrics, dates, args.output_dir)

    # ============================================================
# Annual metrics separate
# ============================================================

    drl_annual = compute_annual_metrics_single_strategy(
        dates,
        drl_metrics
)

    mr_annual = compute_annual_metrics_single_strategy(
    dates,
    mr_metrics
)

    drl_annual.to_csv(
    os.path.join(args.output_dir, "drl_annual_metrics.csv"),
    index=False
)

    mr_annual.to_csv(
    os.path.join(args.output_dir, "mr_annual_metrics.csv"),
    index=False
)

    # ============================================================
    # Genera grafici
    # ============================================================
    plot_comparison(drl_metrics, mr_metrics, dates, args.output_dir, data_path=args.test_data)

    # ============================================================
    # Summary
    # ============================================================
    summary_text = save_summary(drl_metrics, mr_metrics, args, args.output_dir)

    trade_analysis = compute_trade_analysis(drl_metrics)

    conditioning, surrogate_coef, surrogate_acc = compute_action_conditioning_and_surrogate(
    metrics=drl_metrics,
    data_path=args.test_data,
    output_dir=args.output_dir
)

    trade_df = pd.DataFrame([trade_analysis])

    significance_results = compute_statistical_significance(
    metrics=drl_metrics,
    output_dir=args.output_dir
)

    trade_df.to_csv(os.path.join(args.output_dir, "trade_analysis.csv"),index=False)

    signal_conditioning = compute_signal_conditioning_analysis(drl_metrics,args.test_data)

    signal_conditioning.to_csv(os.path.join(args.output_dir, "signal_conditioning_analysis.csv"))

    long_short_analysis = compute_long_short_signal_analysis(drl_metrics,args.test_data)

    long_short_analysis.to_csv(os.path.join(args.output_dir, "long_short_signal_analysis.csv"))




    print("\nTRADE ANALYSIS")
    print(trade_df)

    print("\n" + summary_text)

    print("\n" + "=" * 60)
    print("TABELLA RIASSUNTIVA")
    print("=" * 60)

    header = f"{'Metrica':<20} {'DRL Agent':>15} {'Mean Reversion':>15}"
    print(header)
    print("-" * len(header))

    for key, label in [
        ('total_return', 'Total Return'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('sortino_ratio', 'Sortino Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('n_trades', 'N. Trades'),
        ('win_rate', 'Win Rate')
    ]:
        vals = []
        for m in [drl_metrics, mr_metrics]:
            v = m.get(key, 'N/A')
            if isinstance(v, float):
                if key in ['total_return', 'max_drawdown', 'win_rate']:
                    vals.append(f'{v:.2%}')
                else:
                    vals.append(f'{v:.3f}')
            else:
                vals.append(str(v))

        print(f"{label:<20} {vals[0]:>15} {vals[1]:>15}")

    print("=" * 60)
    print("Comparison complete.", flush=True)
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()