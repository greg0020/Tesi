"""
Scaillet et al. shock-based mean-reversion strategy for the naphtha crack spread.

=== Conceptual difference ===

Classical mean reversion (Bollinger / z-score on price):
    z_t = (P_t - MA_t) / sigma_t
    Entry when |z_t| > threshold, exit when z_t reverts toward 0.
    This measures how far the *level* of the price is from its recent mean.
    It implicitly assumes the price level is stationary around a slow-moving
    average – an assumption that often fails for commodity spreads with
    trending or shifting fundamentals.

Scaillet-style mean reversion (standardized daily crack shocks):
    delta_t   = P_t - P_{t-1}          (daily change in the crack spread)
    vol_t     = rolling_std(delta, w)   (local volatility of daily changes)
    shock_t   = delta_t / vol_t         (standardized shock)
    Entry is *contrarian* on the shock: a large positive shock (shock > +k)
    suggests an abnormal upward move that is likely to revert, so we go SHORT;
    a large negative shock (shock < -k) triggers a LONG.
    Exit is NOT based on the shock returning to zero.  Instead, positions are
    held for a fixed horizon h, or closed early by a stop-loss / take-profit
    on the raw PnL.  This avoids the whipsaw problem of z-score-exit rules
    and directly targets the transient, mean-reverting component of daily
    spread innovations.

This module serves as a benchmark for comparing DRL agent performance.
"""

import numpy as np
import pandas as pd


class MeanReversionStrategy:
    """
    Shock-based mean-reversion strategy (Scaillet et al.).

    Signal logic:
        shock_t = (Close_t - Close_{t-1}) / rolling_std(delta, lookback)
        - shock_t >  entry_threshold  →  open SHORT (contrarian)
        - shock_t < -entry_threshold  →  open LONG  (contrarian)

    Exit logic (whichever comes first):
        - fixed holding period h expired
        - raw unrealised PnL hits +take_profit  → close with profit
        - raw unrealised PnL hits -stop_loss    → close with loss
    """

    def __init__(self,
                 lookback: int = 20,
                 entry_threshold: float = 1.0,
                 holding_period: int = 5,
                 stop_loss: float = 50.0,
                 take_profit: float = 50.0,
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001):
        """
        Args:
            lookback: rolling window for volatility of daily changes
            entry_threshold: shock magnitude to trigger entry
            holding_period: max days to hold a position before forced exit
            stop_loss: raw PnL loss threshold to trigger early exit (>0)
            take_profit: raw PnL profit threshold to trigger early exit (>0)
            initial_balance: starting capital
            transaction_cost: proportional cost per trade (applied to price)
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.holding_period = holding_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

    # --------------------------------------------------------------------- #
    #  Shock computation                                                      #
    # --------------------------------------------------------------------- #

    def compute_shocks(self, prices: np.ndarray) -> tuple:
        """
        Compute the standardised daily crack-spread shocks.

        Returns:
            delta:   array of daily price changes  (NaN at index 0)
            vol:     rolling std of delta           (NaN for first lookback+1 bars)
            shock:   delta / vol                    (NaN where vol is undefined)
        """
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

    # --------------------------------------------------------------------- #
    #  Signal generation + position tracking                                  #
    # --------------------------------------------------------------------- #

    def generate_signals(self, prices: np.ndarray) -> dict:
        """
        Generate signals and track positions bar-by-bar.

        Returns a dict with arrays of length n:
            signals:          0=hold, 1=buy/long, 2=sell/short
            positions:        position after action at bar i  (-1, 0, +1)
            entry_prices:     entry price of the current position (0 if flat)
            time_in_position: how many bars the current position has been held
            unrealised_pnl:   mark-to-market PnL of the open position
        """
        _, _, shock = self.compute_shocks(prices)
        n = len(prices)

        signals = np.zeros(n, dtype=int)
        positions = np.zeros(n, dtype=int)
        entry_prices = np.zeros(n, dtype=np.float64)
        time_in_position = np.zeros(n, dtype=int)
        unrealised_pnl = np.zeros(n, dtype=np.float64)

        # State variables
        pos = 0          # current position: -1, 0, +1
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            s = shock[i]

            # -------------------------------------------------------------- #
            # 1. If in a position, check exit conditions FIRST                #
            # -------------------------------------------------------------- #
            if pos != 0:
                bars_held += 1
                if pos == 1:
                    pnl = prices[i] - entry_price
                else:  # pos == -1
                    pnl = entry_price - prices[i]

                exit_now = False
                # Fixed holding period
                if bars_held >= self.holding_period:
                    exit_now = True
                # Stop-loss
                if pnl <= -self.stop_loss:
                    exit_now = True
                # Take-profit
                if pnl >= self.take_profit:
                    exit_now = True

                if exit_now:
                    # Signal to close: sell if long, buy if short
                    signals[i] = 2 if pos == 1 else 1
                    pos = 0
                    entry_price = 0.0
                    bars_held = 0
                    # Record state after exit
                    positions[i] = 0
                    entry_prices[i] = 0.0
                    time_in_position[i] = 0
                    unrealised_pnl[i] = 0.0
                    continue
                else:
                    # Still holding – record state
                    unrealised_pnl[i] = pnl
                    positions[i] = pos
                    entry_prices[i] = entry_price
                    time_in_position[i] = bars_held
                    signals[i] = 0  # hold
                    continue

            # -------------------------------------------------------------- #
            # 2. Flat – check entry conditions                                #
            # -------------------------------------------------------------- #
            if np.isnan(s):
                signals[i] = 0
                positions[i] = 0
                continue

            if s > self.entry_threshold:
                # Large positive shock → contrarian SHORT
                signals[i] = 2
                pos = -1
                entry_price = prices[i]
                bars_held = 0
            elif s < -self.entry_threshold:
                # Large negative shock → contrarian LONG
                signals[i] = 1
                pos = 1
                entry_price = prices[i]
                bars_held = 0
            else:
                signals[i] = 0

            positions[i] = pos
            entry_prices[i] = entry_price
            time_in_position[i] = bars_held
            unrealised_pnl[i] = 0.0

        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'time_in_position': time_in_position,
            'unrealised_pnl': unrealised_pnl,
        }

    # --------------------------------------------------------------------- #
    #  Backtest                                                               #
    # --------------------------------------------------------------------- #

    def backtest(self, data_path: str) -> dict:
        """
        Run the backtest on a CSV file with a 'Close' column.
        """
        df = pd.read_csv(data_path)
        prices = df['Close'].values.astype(np.float64)
        n = len(prices)

        # Diagnostic: print shock statistics
        _, _, shock = self.compute_shocks(prices)
        valid_shocks = shock[~np.isnan(shock)]
        if len(valid_shocks) > 0:
            print(f"  📊 Shock stats: min={valid_shocks.min():.2f}, max={valid_shocks.max():.2f}, "
                  f"mean={valid_shocks.mean():.2f}, std={valid_shocks.std():.2f}", flush=True)
            print(f"  📊 Shocks > +{self.entry_threshold:.1f}: {(valid_shocks > self.entry_threshold).sum()}, "
                  f"Shocks < -{self.entry_threshold:.1f}: {(valid_shocks < -self.entry_threshold).sum()}", flush=True)
        else:
            print(f"  ⚠️ No valid shocks computed (not enough data?)", flush=True)

        sig_data = self.generate_signals(prices)
        signals = sig_data['signals']
        positions_arr = sig_data['positions']

        # ---- Replay with transaction costs and PnL tracking --------------- #
        balance = self.initial_balance
        pos = 0
        entry_price = 0.0
        bars_held = 0

        daily_pnl = np.zeros(n, dtype=np.float64)
        portfolio_values = np.zeros(n, dtype=np.float64)
        portfolio_values[0] = self.initial_balance
        realised_positions = np.zeros(n, dtype=int)
        trades = []

        for i in range(n):
            action = signals[i]
            price = prices[i]
            prev_price = prices[i - 1] if i > 0 else price

            # Daily mark-to-market PnL on open position (before any action)
            mtm = 0.0
            if pos == 1:
                mtm = price - prev_price
            elif pos == -1:
                mtm = prev_price - price

            trade_pnl = 0.0
            tc = 0.0

            # ---- Process exit -------------------------------------------- #
            if pos != 0 and action != 0:
                # Closing the position
                if pos == 1:
                    raw_pnl = price - entry_price
                else:
                    raw_pnl = entry_price - price
                tc_close = abs(price) * self.transaction_cost
                trade_pnl = raw_pnl - tc_close
                trades.append({
                    'type': 'close_long' if pos == 1 else 'close_short',
                    'entry_step': i - bars_held,
                    'exit_step': i,
                    'entry_price': float(entry_price),
                    'exit_price': float(price),
                    'bars_held': int(bars_held),
                    'raw_pnl': float(raw_pnl),
                    'tc': float(tc_close),
                    'net_pnl': float(trade_pnl),
                })
                balance += trade_pnl
                pos = 0
                entry_price = 0.0
                bars_held = 0
                # mtm for the day is captured via the trade PnL
                # (already in balance), so reset mtm to avoid double counting
                mtm = 0.0
                daily_pnl[i] = trade_pnl

            # ---- Process entry (may happen on same bar after exit) ------- #
            if pos == 0 and action != 0:
                # Check if this is a pure entry signal
                # action==1 → LONG, action==2 → SHORT
                # But entries after an exit on the same bar only happen
                # if generate_signals re-enters (it does not in current logic).
                # So this handles initial entries only.
                # For the exit bar, pos was just set to 0 above; the signal
                # was the *exit* signal, not a new entry.  We skip re-entry
                # on exit bars to keep it clean.
                if daily_pnl[i] != 0.0:
                    # This bar was an exit bar; do not re-enter immediately
                    pass
                else:
                    tc_open = abs(price) * self.transaction_cost
                    balance -= tc_open
                    entry_price = price
                    bars_held = 0
                    pos = 1 if action == 1 else -1
                    trades.append({
                        'type': 'open_long' if pos == 1 else 'open_short',
                        'entry_step': i,
                        'exit_step': None,
                        'entry_price': float(price),
                        'exit_price': None,
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

            # Portfolio value = balance + unrealised PnL
            unrealised = 0.0
            if pos == 1:
                unrealised = price - entry_price
            elif pos == -1:
                unrealised = entry_price - price
            pv = balance + unrealised
            portfolio_values[i] = pv

        # ---- Force-close any open position at the end -------------------- #
        if pos != 0:
            final_price = prices[-1]
            if pos == 1:
                raw_pnl = final_price - entry_price
            else:
                raw_pnl = entry_price - final_price
            tc_close = abs(final_price) * self.transaction_cost
            trade_pnl = raw_pnl - tc_close
            balance += trade_pnl
            trades.append({
                'type': 'force_close',
                'entry_step': int(n - 1 - bars_held),
                'exit_step': int(n - 1),
                'entry_price': float(entry_price),
                'exit_price': float(final_price),
                'bars_held': int(bars_held),
                'raw_pnl': float(raw_pnl),
                'tc': float(tc_close),
                'net_pnl': float(trade_pnl),
            })
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
