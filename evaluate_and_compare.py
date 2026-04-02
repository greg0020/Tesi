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

from trading_environment_close_only import TradingEnvironmentCloseOnly
from drl_agent import DRLAgent
from mean_reversion_strategy import MeanReversionStrategy


def evaluate_drl_agent(model_path: str, data_path: str, window_size: int = 20,
                       initial_balance: float = 100000.0,
                       transaction_cost: float = 0.001) -> dict:
    """Valuta l'agente DRL sul dataset di test."""
    env = TradingEnvironmentCloseOnly(
        data_path=data_path,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        reward_type='pnl'
    )

    agent = DRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_n,
    )
    agent.load(model_path)

    state = env.reset()
    total_reward = 0.0
    actions_taken = []

    while True:
        action = agent.select_action(state, training=False)
        actions_taken.append(action)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    metrics = env.get_metrics()
    metrics['total_reward'] = float(total_reward)
    metrics['portfolio_values'] = [float(v) for v in env.portfolio_values]
    metrics['daily_returns'] = [float(r) for r in env.daily_returns]
    metrics['actions'] = actions_taken
    metrics['trades'] = env.trades

    return metrics


def evaluate_benchmark(data_path: str, lookback: int = 20,
                       entry_threshold: float = 2.0,
                       initial_balance: float = 100000.0,
                       transaction_cost: float = 0.001) -> dict:
    """Valuta la strategia di mean reversion benchmark (Scaillet-style)."""
    strategy = MeanReversionStrategy(
        lookback=lookback,
        entry_threshold=entry_threshold,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost
    )
    return strategy.backtest(data_path)


def buy_and_hold(data_path: str, initial_balance: float = 100000.0) -> dict:
    """Calcola il benchmark Buy & Hold."""
    df = pd.read_csv(data_path)
    prices = df['Close'].values.astype(np.float64)
    returns = np.diff(prices) / prices[:-1]

    portfolio = initial_balance
    portfolio_values = [portfolio]
    for r in returns:
        portfolio *= (1 + r)
        portfolio_values.append(portfolio)

    portfolio_values = np.array(portfolio_values)
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance
    mean_ret = returns.mean()
    std_ret = returns.std() + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(252)

    downside = returns[returns < 0]
    down_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
    sortino = (mean_ret / down_std) * np.sqrt(252)

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + 1e-8)
    max_dd = drawdown.max()

    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'n_trades': 1,
        'win_rate': 1.0 if total_return > 0 else 0.0,
        'final_balance': float(portfolio_values[-1]),
        'portfolio_values': portfolio_values.tolist(),
        'daily_returns': returns.tolist(),
    }


def plot_comparison(drl_metrics: dict, mr_metrics: dict, bh_metrics: dict,
                    save_path: str = None):
    """Genera grafici comparativi tra le strategie."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(drl_metrics['portfolio_values'], label='DRL Agent', linewidth=1.5)
    ax.plot(mr_metrics['portfolio_values'], label='Mean Reversion', linewidth=1.5, alpha=0.8)
    ax.plot(bh_metrics['portfolio_values'], label='Buy & Hold', linewidth=1.5, alpha=0.8)
    ax.set_title('Equity Curve', fontsize=14)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    for label, metrics in [('DRL Agent', drl_metrics), ('Mean Reversion', mr_metrics), ('Buy & Hold', bh_metrics)]:
        pv = np.array(metrics['portfolio_values'])
        peak = np.maximum.accumulate(pv)
        dd = (peak - pv) / (peak + 1e-8)
        ax.fill_between(range(len(dd)), dd, alpha=0.3, label=label)
    ax.set_title('Drawdown', fontsize=14)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Rendimenti cumulativi
    ax = axes[1, 0]
    for label, metrics in [('DRL Agent', drl_metrics), ('Mean Reversion', mr_metrics), ('Buy & Hold', bh_metrics)]:
        rets = np.array(metrics['daily_returns'])
        cum_rets = np.cumprod(1 + rets) - 1
        ax.plot(cum_rets, label=label, linewidth=1.5)
    ax.set_title('Cumulative Returns', fontsize=14)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Tabella metriche
    ax = axes[1, 1]
    ax.axis('off')
    metrics_keys = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'n_trades', 'win_rate']
    labels = ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'N. Trades', 'Win Rate']

    table_data = []
    for key, lbl in zip(metrics_keys, labels):
        row = [lbl]
        for m in [drl_metrics, mr_metrics, bh_metrics]:
            val = m.get(key, 'N/A')
            if isinstance(val, float):
                if 'return' in key.lower() or 'drawdown' in key.lower() or 'win_rate' in key.lower():
                    row.append(f'{val:.2%}')
                else:
                    row.append(f'{val:.3f}')
            else:
                row.append(str(val))
        table_data.append(row)

    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'DRL Agent', 'Mean Reversion', 'Buy & Hold'],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Performance Comparison', fontsize=14, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Valutazione e confronto strategie')
    parser.add_argument('--model_path', type=str, required=True, help='Percorso al modello DRL (.pt)')
    parser.add_argument('--test_data', type=str, default= 'Data/naphtha_crack_test.csv', help='CSV dati di test')
    parser.add_argument('--output_dir', type=str, default='Data/evaluation_results', help='Cartella output')
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_cost', type=float, default=0.001)
    parser.add_argument('--mr_lookback', type=int, default=20, help='Lookback mean reversion')
    parser.add_argument('--mr_entry', type=float, default=1.0, help='Entry threshold mean reversion (z-score). '
                        'Lower = more trades. Default 1.0 for synthetic data, use 2.0 for real data.')
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}", flush=True)
    print(f"Test data: {args.test_data}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("VALUTAZIONE E CONFRONTO STRATEGIE")
    print("=" * 60)

    # Valuta DRL
    print("\n[1/3] Valutazione agente DRL...")
    drl_metrics = evaluate_drl_agent(
        model_path=args.model_path,
        data_path=args.test_data,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost
    )
    print(f"DRL evaluation complete", flush=True)
    print(f"  Return: {drl_metrics['total_return']:.2%} | Sharpe: {drl_metrics['sharpe_ratio']:.3f} | "
          f"Trades: {drl_metrics['n_trades']} | Win Rate: {drl_metrics['win_rate']:.2%}")

    # Valuta Mean Reversion
    print("\n[2/3] Valutazione strategia Mean Reversion...")
    mr_metrics = evaluate_benchmark(
        data_path=args.test_data,
        lookback=args.mr_lookback,
        entry_threshold=args.mr_entry,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost
    )
    print(f"Benchmark evaluation complete", flush=True)
    print(f"  Return: {mr_metrics['total_return']:.2%} | Sharpe: {mr_metrics['sharpe_ratio']:.3f} | "
          f"Trades: {mr_metrics['n_trades']} | Win Rate: {mr_metrics['win_rate']:.2%}")

    # Buy & Hold
    print("\n[3/3] Calcolo Buy & Hold benchmark...")
    bh_metrics = buy_and_hold(args.test_data, args.initial_balance)
    print(f"  Return: {bh_metrics['total_return']:.2%} | Sharpe: {bh_metrics['sharpe_ratio']:.3f}")

    # Salva risultati
    results = {
        'drl_agent': {k: v for k, v in drl_metrics.items() if k not in ['portfolio_values', 'daily_returns', 'actions', 'trades']},
        'mean_reversion': {k: v for k, v in mr_metrics.items() if k not in ['portfolio_values', 'daily_returns', 'signals', 'trades']},
        'buy_and_hold': {k: v for k, v in bh_metrics.items() if k not in ['portfolio_values', 'daily_returns']},
    }
    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRisultati salvati in: {results_path}")

    # Genera grafici
    plot_path = os.path.join(args.output_dir, 'comparison_plot.png')
    plot_comparison(drl_metrics, mr_metrics, bh_metrics, save_path=plot_path)

    # =====================================================================
    # NEW: Save comparison as readable CSV
    # =====================================================================
    comparison_rows = []
    for key, label in [('total_return', 'Total Return'), ('sharpe_ratio', 'Sharpe Ratio'),
                       ('sortino_ratio', 'Sortino Ratio'), ('max_drawdown', 'Max Drawdown'),
                       ('n_trades', 'N. Trades'), ('win_rate', 'Win Rate'),
                       ('final_balance', 'Final Balance')]:
        row = {'Metric': label}
        for name, m in [('DRL_Agent', drl_metrics), ('Mean_Reversion', mr_metrics), ('Buy_Hold', bh_metrics)]:
            row[name] = m.get(key, 'N/A')
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(args.output_dir, 'comparison_table.csv')
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\n📄 Comparison CSV saved: {comp_csv_path}", flush=True)

    # =====================================================================
    # NEW: Save portfolio values as CSV (for external analysis)
    # =====================================================================
    max_len = max(len(drl_metrics['portfolio_values']),
                  len(mr_metrics['portfolio_values']),
                  len(bh_metrics['portfolio_values']))
    
    def pad_list(lst, length):
        return lst + [None] * (length - len(lst))

    pv_df = pd.DataFrame({
        'step': range(max_len),
        'DRL_Agent': pad_list(drl_metrics['portfolio_values'], max_len),
        'Mean_Reversion': pad_list(mr_metrics['portfolio_values'], max_len),
        'Buy_Hold': pad_list(bh_metrics['portfolio_values'], max_len),
    })
    pv_csv_path = os.path.join(args.output_dir, 'portfolio_values.csv')
    pv_df.to_csv(pv_csv_path, index=False)
    print(f"📄 Portfolio values CSV saved: {pv_csv_path}", flush=True)

    # =====================================================================
    # NEW: Save human-readable text summary
    # =====================================================================
    summary_lines = [
        "=" * 60,
        "EVALUATION SUMMARY",
        "=" * 60,
        f"Test data: {args.test_data}",
        f"Model: {args.model_path}",
        f"Initial balance: ${args.initial_balance:,.2f}",
        "",
    ]
    for name, m in [('DRL Agent', drl_metrics), ('Mean Reversion', mr_metrics), ('Buy & Hold', bh_metrics)]:
        summary_lines.append(f"--- {name} ---")
        summary_lines.append(f"  Total Return:  {m.get('total_return', 0):.2%}")
        summary_lines.append(f"  Sharpe Ratio:  {m.get('sharpe_ratio', 0):.3f}")
        summary_lines.append(f"  Sortino Ratio: {m.get('sortino_ratio', 0):.3f}")
        summary_lines.append(f"  Max Drawdown:  {m.get('max_drawdown', 0):.2%}")
        summary_lines.append(f"  N. Trades:     {m.get('n_trades', 'N/A')}")
        summary_lines.append(f"  Win Rate:      {m.get('win_rate', 0)::.2%}")
        summary_lines.append(f"  Final Balance: ${m.get('final_balance', 0):,.2f}")
        summary_lines.append("")
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"📄 Text summary saved: {summary_path}", flush=True)

    # Print it too
    print("\n" + summary_text)

    # Stampa tabella riassuntiva
    print("\n" + "=" * 60)
    print("TABELLA RIASSUNTIVA")
    print("=" * 60)
    header = f"{'Metrica':<20} {'DRL Agent':>15} {'Mean Reversion':>15} {'Buy & Hold':>15}"
    print(header)
    print("-" * len(header))
    for key, label in [('total_return', 'Total Return'), ('sharpe_ratio', 'Sharpe Ratio'),
                       ('sortino_ratio', 'Sortino Ratio'), ('max_drawdown', 'Max Drawdown'),
                       ('n_trades', 'N. Trades'), ('win_rate', 'Win Rate')]:
        vals = []
        for m in [drl_metrics, mr_metrics, bh_metrics]:
            v = m.get(key, 'N/A')
            if isinstance(v, float):
                if key in ['total_return', 'max_drawdown', 'win_rate']:
                    vals.append(f'{v:.2%}')
                else:
                    vals.append(f'{v:.3f}')
            else:
                vals.append(str(v))
        print(f"{label:<20} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print("=" * 60)
    print(f"Comparison complete.", flush=True)


if __name__ == '__main__':
    main()
