import numpy as np
import pandas as pd


class MeanReversionStrategy:
    """
    Shock-based mean-reversion strategy (Scaillet et al.).

    """

    def __init__(self,
                 lookback: int = 20,
                 entry_threshold: float = 2,
                 holding_period: int = 10,
                 stop_loss: float = 0.0,
                 take_profit: float = 0.0,
                 initial_balance: float = 1.0,
                 transaction_cost: float = 0.0):
        """
        Args:
            lookback: rolling window for volatility of daily changes
            entry_threshold: shock magnitude to trigger entry
            holding_period: maximum days to hold a position before forced exit
            stop_loss: two-leg unrealised PnL loss threshold to trigger early exit (>0)
            take_profit: two-leg unrealised PnL profit threshold to trigger early exit (>0)
            initial_balance: starting capital
            transaction_cost: proportional cost per trade (applied once at open, once at close)
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.holding_period = holding_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

    def compute_shocks(self, prices: np.ndarray) -> tuple:
        n = len(prices)
        delta = np.full(n, np.nan)
        vol = np.full(n, np.nan)
        shock = np.full(n, np.nan)

        # delta_t = P_t - P_{t-1}
        delta[1:] = prices[1:] - prices[:-1]

        # rolling std of delta over [t-lookback, t-1]
        for i in range(self.lookback + 1, n):
            window = delta[i - self.lookback:i]  # lookback past deltas
            std = np.nanstd(window, ddof=1)
            vol[i] = std if std > 1e-12 else 1e-12

        # shock_t = delta_t / vol_t
        valid = ~np.isnan(vol) & ~np.isnan(delta)
        shock[valid] = delta[valid] / vol[valid]

        return delta, vol, shock



    def generate_signals(self,
                         prices: np.ndarray,
                         naphtha: np.ndarray,
                         brent: np.ndarray,
                         half_life: np.ndarray) -> dict:

        _, vol, shock = self.compute_shocks(prices)
        n = len(prices)

        signals = np.zeros(n, dtype=int)
        positions = np.zeros(n, dtype=int)
        entry_prices = np.zeros(n, dtype=np.float64)
        time_in_position = np.zeros(n, dtype=int)
        ma = pd.Series(prices).rolling(50).mean().values

        # State variables
        pos = 0          # current position: -1, 0, +1
        entry_price = 0.0
        entry_naphtha = 0.0
        entry_brent = 0.0
        bars_held = 0

        for i in range(n):
            s = shock[i]

            # 1. If in a position, check exit conditions FIRST                #

            if pos != 0:
                bars_held += 1

                # Two-leg unrealised PnL
                if pos == 1:
                    unrealised = ((naphtha[i] - entry_naphtha) / entry_naphtha
                                  + (entry_brent - brent[i]) / entry_brent)
                else:
                    unrealised = ((entry_naphtha - naphtha[i]) / entry_naphtha
                                  + (brent[i] - entry_brent) / entry_brent)

                exit_now = False
                if bars_held >= self.holding_period:
                    exit_now = True
                if unrealised <= -self.stop_loss:
                    exit_now = True
                if unrealised >= self.take_profit:
                    exit_now = True

                if exit_now:
                    signals[i] = 2 if pos == 1 else 1
                    pos = 0
                    entry_price = 0.0
                    entry_naphtha = 0.0
                    entry_brent = 0.0
                    bars_held = 0
                    positions[i] = 0
                    entry_prices[i] = 0.0
                    time_in_position[i] = 0
                    continue
                else:
                    positions[i] = pos
                    entry_prices[i] = entry_price
                    time_in_position[i] = bars_held
                    signals[i] = 0  # hold
                    continue

            # 2. Flat – check entry conditions                                #

            if np.isnan(s):
                signals[i] = 0
                positions[i] = 0
                continue

            # Volatility filter: trade only when vol is above the 30th percentile
            #if np.isnan(vol[i]) or vol[i] < np.nanpercentile(vol, 30):
                #signals[i] = 0
                #positions[i] = 0
                #continue

            # Half-life filter: trade only when mean reversion is expected to be fast
            if np.isnan(half_life[i]) or half_life[i] <= np.nanpercentile(half_life, 30):
                signals[i] = 0
                positions[i] = 0
                continue

            if s > self.entry_threshold:
                # Large positive shock → contrarian SHORT
                signals[i] = 2
                pos = -1
                entry_price = prices[i]
                entry_naphtha = naphtha[i]
                entry_brent = brent[i]
                bars_held = 0
            elif s < -self.entry_threshold:
                # Large negative shock → contrarian LONG
                signals[i] = 1
                pos = 1
                entry_price = prices[i]
                entry_naphtha = naphtha[i]
                entry_brent = brent[i]
                bars_held = 0
            else:
                signals[i] = 0

            positions[i] = pos
            entry_prices[i] = entry_price
            time_in_position[i] = bars_held

        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'time_in_position': time_in_position,
        }


    def backtest(self, data_path: str) -> dict:
        
        df = pd.read_csv(data_path)
        prices = df['Close'].values.astype(np.float64)
        naphtha = df['Naphtha_Close'].values.astype(np.float64)
        brent = df['Brent_Close'].values.astype(np.float64)
        half_life = df['half_life_proxy'].values.astype(np.float64)
        n = len(prices)

        # Diagnostic: print shock statistics
        _, _, shock = self.compute_shocks(prices)
        valid_shocks = shock[~np.isnan(shock)]
        if len(valid_shocks) > 0:
            print(f"Shock stats: min={valid_shocks.min():.2f}, max={valid_shocks.max():.2f}, "
                  f"mean={valid_shocks.mean():.2f}, std={valid_shocks.std():.2f}", flush=True)
            print(f"Shocks > +{self.entry_threshold:.1f}: {(valid_shocks > self.entry_threshold).sum()}, "
                  f"Shocks < -{self.entry_threshold:.1f}: {(valid_shocks < -self.entry_threshold).sum()}", flush=True)
        else:
            print(f"No shocks", flush=True)

        sig_data = self.generate_signals(prices, naphtha, brent, half_life)
        signals = sig_data['signals']
        positions_arr = sig_data['positions']

        # ---- Replay with transaction costs and PnL tracking --------------- #
        balance = self.initial_balance
        pos = 0
        entry_naphtha = 0.0
        entry_brent = 0.0
        bars_held = 0

        daily_pnl = np.zeros(n, dtype=np.float64)
        portfolio_values = np.zeros(n, dtype=np.float64)
        portfolio_values[0] = self.initial_balance
        realised_positions = np.zeros(n, dtype=int)
        trades = []

        for i in range(n):
            action = signals[i]
            nph = naphtha[i]
            brt = brent[i]
            prev_nph = naphtha[i - 1] if i > 0 else nph
            prev_brt = brent[i - 1] if i > 0 else brt

            # Daily mark-to-market PnL on open position (before any action)
            # Long crack:  long Naphtha + short Brent
            # Short crack: short Naphtha + long Brent
            mtm = 0.0
            if pos == 1:
                mtm = 0.5* ((nph - prev_nph) / prev_nph + (prev_brt - brt) / prev_brt)
            elif pos == -1:
                mtm = 0.5* ((prev_nph - nph) / prev_nph + (brt - prev_brt) / prev_brt)

            trade_pnl = 0.0
            tc = 0.0

            # ---- Process exit -------------------------------------------- #
            if pos != 0 and action != 0:
                # Closing the position – total PnL from entry to exit on both legs
                if pos == 1:
                    raw_pnl = 0.5 * ((nph - entry_naphtha) / entry_naphtha + (entry_brent - brt) / entry_brent)
                else:
                    raw_pnl = 0.5 * ((entry_naphtha - nph) / entry_naphtha + (brt - entry_brent) / entry_brent)
                tc_close = self.transaction_cost
                trade_pnl = raw_pnl - tc_close
                trades.append({
                    'type': 'close_long' if pos == 1 else 'close_short',
                    'entry_step': i - bars_held,
                    'exit_step': i,
                    'entry_naphtha': float(entry_naphtha),
                    'entry_brent': float(entry_brent),
                    'exit_naphtha': float(nph),
                    'exit_brent': float(brt),
                    'bars_held': int(bars_held),
                    'raw_pnl': float(raw_pnl),
                    'tc': float(tc_close),
                    'net_pnl': float(trade_pnl),
                })
                balance += trade_pnl
                pos = 0
                entry_naphtha = 0.0
                entry_brent = 0.0
                bars_held = 0
                # mtm for the day is captured via the trade PnL
                # (already in balance), so reset mtm to avoid double counting
                mtm = 0.0
                daily_pnl[i] = trade_pnl

            # ---- Process entry (may happen on same bar after exit) ------- #
            if pos == 0 and action != 0:
                # action==1 → LONG crack, action==2 → SHORT crack
                # For the exit bar, pos was just set to 0 above; the signal
                # was the *exit* signal, not a new entry.  We skip re-entry
                # on exit bars to keep it clean.
                if daily_pnl[i] != 0.0:
                    # This bar was an exit bar; do not re-enter immediately
                    pass
                else:
                    tc_open = self.transaction_cost
                    balance -= tc_open
                    entry_naphtha = nph
                    entry_brent = brt
                    bars_held = 0
                    pos = 1 if action == 1 else -1
                    trades.append({
                        'type': 'open_long' if pos == 1 else 'open_short',
                        'entry_step': i,
                        'exit_step': None,
                        'entry_naphtha': float(nph),
                        'entry_brent': float(brt),
                        'exit_naphtha': None,
                        'exit_brent': None,
                        'bars_held': 0,
                        'raw_pnl': 0.0,
                        'tc': float(tc_open),
                        'net_pnl': -float(tc_open),
                    })
                    daily_pnl[i] = -tc_open

            elif pos != 0 and action == 0:
                # Holding – daily PnL is the mark-to-market change
                bars_held += 1
                daily_pnl[i] = mtm

            realised_positions[i] = pos

            # Portfolio value = balance + unrealised PnL on both legs
            unrealised = 0.0
            if pos == 1:
                unrealised = 0.5 * ((nph - entry_naphtha) / entry_naphtha + (entry_brent - brt) / entry_brent)
            elif pos == -1:
                unrealised = 0.5 * ((entry_naphtha - nph) / entry_naphtha + (brt - entry_brent) / entry_brent)
            pv = balance + unrealised
            portfolio_values[i] = pv

        if pos != 0:
            final_nph = naphtha[-1]
            final_brt = brent[-1]
            if pos == 1:
                raw_pnl = 0.5 * ((final_nph - entry_naphtha) / entry_naphtha + (entry_brent - final_brt) / entry_brent)
            else:
                raw_pnl = 0.5 * ((entry_naphtha - final_nph) / entry_naphtha + (final_brt - entry_brent) / entry_brent)
            tc_close = self.transaction_cost
            trade_pnl = raw_pnl - tc_close
            balance += trade_pnl
            trades.append({
                'type': 'force_close',
                'entry_step': int(n - 1 - bars_held),
                'exit_step': int(n - 1),
                'entry_naphtha': float(entry_naphtha),
                'entry_brent': float(entry_brent),
                'exit_naphtha': float(final_nph),
                'exit_brent': float(final_brt),
                'bars_held': int(bars_held),
                'raw_pnl': float(raw_pnl),
                'tc': float(tc_close),
                'net_pnl': float(trade_pnl),
            })
            daily_pnl[-1] = trade_pnl
            portfolio_values[-1] = balance

        # ---- Compute performance metrics --------------------------------- #
        cumulative_pnl = np.cumsum(daily_pnl)

        daily_returns = np.zeros(n)
        daily_returns[1:] = (portfolio_values[1:] - portfolio_values[:-1]) / (np.abs(portfolio_values[:-1]) + 1e-12)

        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        mean_ret = daily_returns[1:].mean()
        std_ret = daily_returns[1:].std() + 1e-12
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() + 1e-12 if len(downside) > 0 else 1e-12
        sortino_ratio = (mean_ret / downside_std) * np.sqrt(252)

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-12)
        max_drawdown = float(drawdown.max())

        closed_trades = [t for t in trades if t['type'].startswith('close') or t['type'] == 'force_close']
        n_trades = len(closed_trades)
        winning = [t for t in closed_trades if t['net_pnl'] > 0]
        win_rate = len(winning) / n_trades if n_trades > 0 else 0.0

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': float(win_rate),
            'final_balance': float(portfolio_values[-1]),
            'total_pnl': float(cumulative_pnl[-1]),
            'mean_daily_return': float(mean_ret),
            'std_daily_return': float(std_ret),
            'portfolio_values': portfolio_values.tolist(),
            'daily_pnl': daily_pnl.tolist(),
            'cumulative_pnl': cumulative_pnl.tolist(),
            'daily_returns': daily_returns.tolist(),
            'positions': realised_positions.tolist(),
            'trades': trades,
            'signals': signals.tolist(),
        }

if __name__ == "__main__":
    import pandas as pd

    data_path = "Data/naphtha_crack_test.csv"

    holding_periods = [8, 9, 10]
    entry_thresholds = [2.6, 2.7, 2.8]
    

    results_list = []

    for hp in holding_periods:
        for et in entry_thresholds:

                    strategy = MeanReversionStrategy(
                        entry_threshold=et,
                        holding_period=hp,
                        initial_balance=1.0,
                        transaction_cost=0.0,
                    )

                    results = strategy.backtest(data_path)

                    results_list.append({
                        "holding_period": hp,
                        "entry_threshold": et,
                        "total_return": results["total_return"],
                        "sharpe_ratio": results["sharpe_ratio"],
                        "max_drawdown": results["max_drawdown"],
                        "n_trades": results["n_trades"],
                        "win_rate": results["win_rate"],
                    })

                    print(
                        f"hp={hp}, et={et:.1f} | "
                        f"Ret={results['total_return']:.2%} | "
                        f"Sharpe={results['sharpe_ratio']:.3f} | "
                        f"Trades={results['n_trades']}"
                    )

    df_results = pd.DataFrame(results_list)

    # salva tutto
    df_results.to_csv("Data/mr_grid_search_full.csv", index=False)
