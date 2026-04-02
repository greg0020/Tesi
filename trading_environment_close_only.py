"""
Ambiente di trading per DRL adattato a dati con solo prezzo di chiusura.
Basato sull'ambiente originale ma semplificato per il naphtha crack spread.
Azioni: 0=Hold, 1=Buy/Long, 2=Sell/Short
"""

import numpy as np
import pandas as pd
from collections import deque


class TradingEnvironmentCloseOnly:
    """
    Ambiente di trading che usa solo prezzi di chiusura e feature derivate.
    Pensato per strategie di mean reversion su spread (naphtha crack).
    """

    def __init__(self, data_path: str, window_size: int = 20,
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001,
                 reward_type: str = 'pnl',
                 feature_groups: list = None):
        """
        Inizializza l'ambiente di trading.

        Args:
            data_path: percorso al CSV con le feature
            window_size: dimensione della finestra di osservazione
            initial_balance: capitale iniziale
            transaction_cost: costo di transazione (percentuale)
            reward_type: tipo di reward ('pnl', 'sharpe', 'sortino')
            feature_groups: lista di prefissi feature da usare, es. ['naphtha', 'brent', 'crack'].
                           None = usa tutte. Opzioni: 'naphtha', 'brent', 'crack', 
                           'correlation', 'beta', 'vol_diff', 'vol_ratio', 'momentum_diff',
                           'ratio_dist', 'return_diff', 'naphtha_brent_ratio',
                           'naphtha_contribution', 'brent_contribution',
                           'naphtha_direction', 'brent_direction'
        """
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type

        # Carica e prepara dati
        self.df = pd.read_csv(data_path)
        self.prices = self.df['Close'].values.astype(np.float64)

        # Feature: tutte le colonne tranne Date, Close e prezzi grezzi delle gambe
        exclude_cols = ['Date', 'Close', 'Naphtha_Close', 'Brent_Close', 'Crack_Spread']
        all_feature_cols = [c for c in self.df.columns if c not in exclude_cols]

        # Filtra feature per gruppo se specificato
        if feature_groups is not None:
            feature_cols = [c for c in all_feature_cols 
                          if any(c.startswith(prefix) for prefix in feature_groups)]
            if len(feature_cols) == 0:
                raise ValueError(f"Nessuna feature trovata per i gruppi: {feature_groups}. "
                               f"Colonne disponibili: {all_feature_cols[:10]}...")
        else:
            feature_cols = all_feature_cols

        self.feature_names = feature_cols
        self.features = self.df[feature_cols].values.astype(np.float64)

        # Normalizza le feature (z-score sul dataset)
        self.feature_mean = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0) + 1e-8
        self.features_norm = (self.features - self.feature_mean) / self.feature_std

        self.n_features = self.features_norm.shape[1]
        self.n_steps = len(self.prices)

        # Dimensione stato: feature di mercato + posizione + unrealized PnL + time in position
        # position_encoding:  -1 (short), 0 (flat), 1 (long)
        # unrealized_pnl:     PnL non realizzato normalizzato per entry price
        # time_in_position:   step in posizione, normalizzato per max_holding
        self.max_holding = 20  # normalizzazione time_in_position (≈1 mese di trading)
        self.state_dim = self.n_features + 3  # feature + position + unrealized_pnl + time_in_position

        # Spazio azioni: 0=Hold, 1=Buy/Long, 2=Sell/Short
        self.action_space_n = 3

        # Variabili di stato
        self.reset()

    def reset(self):
        """Resetta l'ambiente all'inizio di un episodio."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.time_in_position = 0  # contatore step in posizione

        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.daily_returns = []
        self.rewards_history = deque(maxlen=20)

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Costruisce il vettore di stato corrente (18 dimensioni).

        Struttura:
          [0:15]  15 feature di mercato (normalizzate z-score)
          [15]    position_encoding: -1=short, 0=flat, 1=long
          [16]    unrealized_pnl: PnL non realizzato normalizzato per entry price
          [17]    time_in_position: step in posizione / max_holding (saturato a 1.0)

        Motivazioni per le 3 variabili di stato dell'agente:
        - position_encoding: l'agente deve sapere la sua posizione corrente
          per decidere se mantenerla, chiuderla o invertirla.
        - unrealized_pnl: cattura il rischio corrente e incentiva il profit-taking
          o lo stop-loss in modo informato.
        - time_in_position: nel framework Scaillet, il holding period è cruciale.
          Un trade mean-reversion ha un half-life stimato; se il tempo in posizione
          supera il half-life atteso senza convergenza, è segnale di uscita.
          Normalizzato a [0,1] dove 1.0 = max_holding raggiunto.
        """
        # Feature normalizzate al passo corrente
        features = self.features_norm[self.current_step]

        # Posizione corrente codificata
        pos_encoding = float(self.position)

        # PnL non realizzato normalizzato
        if self.position != 0 and abs(self.entry_price) > 1e-8:
            current_price = self.prices[self.current_step]
            unrealized = self.position * (current_price - self.entry_price) / abs(self.entry_price)
        else:
            unrealized = 0.0

        # Tempo in posizione normalizzato (saturato a 1.0)
        time_norm = min(self.time_in_position / self.max_holding, 1.0)

        state = np.concatenate([features, [pos_encoding, unrealized, time_norm]])
        return state.astype(np.float32)

    def step(self, action: int):
        """
        Esegue un passo nell'ambiente.

        Args:
            action: 0=Hold, 1=Buy/Long, 2=Sell/Short

        Returns:
            next_state, reward, done, info
        """
        assert 0 <= action <= 2, f"Azione non valida: {action}"

        current_price = self.prices[self.current_step]
        prev_portfolio = self._get_portfolio_value()
        trade_pnl = 0.0
        transaction_cost_paid = 0.0

        # Gestione posizioni
        if action == 1:  # Buy/Long
            if self.position == -1:
                # Chiudi short
                trade_pnl = self.entry_price - current_price
                transaction_cost_paid += abs(trade_pnl) * self.transaction_cost if trade_pnl != 0 else current_price * self.transaction_cost
                self.balance += trade_pnl - transaction_cost_paid
                self.total_pnl += trade_pnl - transaction_cost_paid
                self.trades.append({
                    'type': 'close_short',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl': trade_pnl - transaction_cost_paid,
                    'holding_period': self.time_in_position
                })
                self.position = 0
                self.entry_price = 0.0
                self.time_in_position = 0
            if self.position == 0:
                # Apri long
                self.position = 1
                self.entry_price = current_price
                self.time_in_position = 0
                transaction_cost_paid += current_price * self.transaction_cost
                self.balance -= transaction_cost_paid
                self.trades.append({
                    'type': 'open_long',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl': 0.0
                })

        elif action == 2:  # Sell/Short
            if self.position == 1:
                # Chiudi long
                trade_pnl = current_price - self.entry_price
                transaction_cost_paid += abs(trade_pnl) * self.transaction_cost if trade_pnl != 0 else current_price * self.transaction_cost
                self.balance += trade_pnl - transaction_cost_paid
                self.total_pnl += trade_pnl - transaction_cost_paid
                self.trades.append({
                    'type': 'close_long',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl': trade_pnl - transaction_cost_paid,
                    'holding_period': self.time_in_position
                })
                self.position = 0
                self.entry_price = 0.0
                self.time_in_position = 0
            if self.position == 0:
                # Apri short
                self.position = -1
                self.entry_price = current_price
                self.time_in_position = 0
                transaction_cost_paid += current_price * self.transaction_cost
                self.balance -= transaction_cost_paid
                self.trades.append({
                    'type': 'open_short',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl': 0.0
                })

        # action == 0: Hold, non fare nulla

        # Incrementa tempo in posizione se siamo in una posizione
        if self.position != 0:
            self.time_in_position += 1

        # Calcola portfolio value dopo l'azione
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Se finito, chiudi posizioni aperte
        if done and self.position != 0:
            close_price = self.prices[self.current_step]
            if self.position == 1:
                trade_pnl = close_price - self.entry_price
            else:
                trade_pnl = self.entry_price - close_price
            tc = abs(trade_pnl) * self.transaction_cost if trade_pnl != 0 else close_price * self.transaction_cost
            self.balance += trade_pnl - tc
            self.total_pnl += trade_pnl - tc
            self.trades.append({
                'type': 'force_close',
                'step': self.current_step,
                'price': close_price,
                'pnl': trade_pnl - tc,
                'holding_period': self.time_in_position
            })
            self.position = 0
            self.entry_price = 0.0
            self.time_in_position = 0

        current_portfolio = self._get_portfolio_value()
        self.portfolio_values.append(current_portfolio)

        # Rendimento giornaliero
        daily_ret = (current_portfolio - prev_portfolio) / prev_portfolio if prev_portfolio != 0 else 0.0
        self.daily_returns.append(daily_ret)

        # Calcola reward
        reward = self._compute_reward(daily_ret)
        self.rewards_history.append(reward)

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = {
            'portfolio_value': current_portfolio,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'daily_return': daily_ret
        }

        return next_state, reward, done, info

    def _get_portfolio_value(self) -> float:
        """Calcola il valore corrente del portafoglio."""
        unrealized = 0.0
        if self.position != 0:
            current_price = self.prices[self.current_step]
            if self.position == 1:
                unrealized = current_price - self.entry_price
            else:
                unrealized = self.entry_price - current_price
        return self.balance + unrealized

    def _compute_reward(self, daily_return: float) -> float:
        """Calcola il reward in base al tipo selezionato."""
        if self.reward_type == 'pnl':
            return daily_return * 100  # scala per stabilità

        elif self.reward_type == 'sharpe':
            if len(self.daily_returns) < 2:
                return 0.0
            returns = np.array(self.daily_returns[-20:])
            mean_r = returns.mean()
            std_r = returns.std() + 1e-8
            return (mean_r / std_r) * np.sqrt(252)

        elif self.reward_type == 'sortino':
            if len(self.daily_returns) < 2:
                return 0.0
            returns = np.array(self.daily_returns[-20:])
            mean_r = returns.mean()
            downside = returns[returns < 0]
            downside_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
            return (mean_r / downside_std) * np.sqrt(252)

        return daily_return * 100

    def get_metrics(self) -> dict:
        """Calcola le metriche di performance dell'episodio."""
        portfolio_values = np.array(self.portfolio_values)
        returns = np.array(self.daily_returns) if len(self.daily_returns) > 0 else np.array([0.0])

        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8

        # Sharpe annualizzato
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 1e-8 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252)

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-8)
        max_drawdown = drawdown.max()

        # Trades
        n_trades = len([t for t in self.trades if t['type'].startswith('close') or t['type'] == 'force_close'])
        winning_trades = [t for t in self.trades if (t['type'].startswith('close') or t['type'] == 'force_close') and t['pnl'] > 0]
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0.0

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'n_trades': int(n_trades),
            'win_rate': float(win_rate),
            'final_balance': float(portfolio_values[-1]),
            'total_pnl': float(self.total_pnl),
            'mean_daily_return': float(mean_return),
            'std_daily_return': float(std_return),
        }

    def print_state_info(self):
        """Stampa informazioni dettagliate sul vettore di stato."""
        print(f"\n{'='*60}")
        print(f"STRUTTURA VETTORE DI STATO (input della rete neurale)")
        print(f"{'='*60}")
        print(f"Dimensione totale: {self.state_dim}")
        print(f"  - Feature di mercato: {self.n_features}")
        print(f"  - Posizione corrente: 1 (valori: -1=short, 0=flat, 1=long)")
        print(f"  - PnL non realizzato: 1 (normalizzato per entry price)")
        print(f"  - Tempo in posizione: 1 (normalizzato a [0,1], max={self.max_holding})")
        print(f"\nFeature di mercato ({self.n_features}):")
        
        # Raggruppa per prefisso
        groups = {}
        for name in self.feature_names:
            prefix = name.split('_')[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(name)
        
        idx = 0
        for group, cols in groups.items():
            print(f"\n  [{group}] ({len(cols)} feature):")
            for col in cols:
                print(f"    [{idx:3d}] {col}")
                idx += 1
        
        print(f"\n  [{idx:3d}] position_encoding (-1/0/1)")
        print(f"  [{idx+1:3d}] unrealized_pnl_normalized")
        print(f"  [{idx+2:3d}] time_in_position_normalized (0.0 to 1.0)")
        print(f"{'='*60}\n")
