"""
Script di addestramento dell'agente DRL sul naphtha crack spread.
Addestra un agente Double DQN e salva il modello migliore.
"""

import numpy as np
import argparse
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from trading_environment_close_only import TradingEnvironmentCloseOnly
from drl_agent import DRLAgent


def train(args):
    """Ciclo principale di addestramento."""
    print("=" * 60)
    print("ADDESTRAMENTO AGENTE DRL - NAPHTHA CRACK SPREAD")
    print("=" * 60)

    # Crea cartella per i risultati
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Salva configurazione
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Inizializza ambiente di training
    env = TradingEnvironmentCloseOnly(
        data_path=args.train_data,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        reward_type=args.reward_type,
        feature_groups=args.feature_groups
    )

    # Stampa struttura dello stato
    env.print_state_info()

    # Inizializza agente
    agent = DRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_n,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        hidden_sizes=[args.hidden1, args.hidden2, args.hidden3]
    )

    print(f"\nDimensione stato: {env.state_dim}")
    print(f"Numero azioni: {env.action_space_n}")
    print(f"Device: {agent.device}")
    print(f"Dati training: {args.train_data}")
    print(f"Episodi: {args.n_episodes}")
    print(f"Reward type: {args.reward_type}\n")

    # Log inizio training
    print(f"Starting training: {args.n_episodes} episodes, reward={args.reward_type}", flush=True)
    print(f"Train data: {args.train_data}", flush=True)

    # Variabili per il monitoraggio
    best_reward = -np.inf
    episode_rewards = []
    episode_metrics = []
    training_log = []  # <-- NEW: per-episode log

    for episode in range(args.n_episodes):
        state = env.reset()
        total_reward = 0.0
        step = 0
        episode_loss = 0.0
        loss_count = 0

        while True:
            # Seleziona azione
            action = agent.select_action(state, training=True)

            # Esegui azione nell'ambiente
            next_state, reward, done, info = env.step(action)

            # Salva transizione
            agent.store_transition(state, action, reward, next_state, float(done))

            # Addestra l'agente
            if step % 4 == 0 :
                loss = agent.learn()
            else: 
                loss = None   

            # Track loss
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            total_reward += reward
            state = next_state
            step += 1

            if done:
                break

        # Metriche dell'episodio
        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_metrics.append(metrics)
        epsilon = agent.get_epsilon()
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0

        # Record to training log
        training_log.append({
            'episode': episode + 1,
            'total_reward': round(total_reward, 6),
            'total_return': round(metrics.get('total_return', 0.0), 6),
            'final_portfolio_value': round(metrics.get('final_balance', args.initial_balance), 2),
            'n_trades': metrics.get('n_trades', 0),
            'sharpe_ratio': round(metrics.get('sharpe_ratio', 0.0), 4),
            'win_rate': round(metrics.get('win_rate', 0.0), 4),
            'epsilon': round(epsilon, 6),
            'avg_loss': round(avg_loss, 6),
        })

        # Stampa progresso ogni N episodi
        if (episode + 1) % args.print_every == 0:
            avg_reward = np.mean(episode_rewards[-args.print_every:])
            print(f"Ep. {episode + 1}/{args.n_episodes} | "
                  f"Reward: {total_reward:.4f} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Return: {metrics['total_return']:.4%} | "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
                  f"Trades: {metrics['n_trades']} | "
                  f"Win Rate: {metrics['win_rate']:.2%} | "
                  f"Epsilon: {epsilon:.4f} | "
                  f"Loss: {avg_loss:.6f}", flush=True)

        # Log ogni 50 episodi
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{args.n_episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f}", flush=True)

        # Salva modello migliore
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(save_dir, 'best_model.pt'))

        # Salva checkpoint periodico
        if (episode + 1) % args.save_every == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_ep{episode + 1}.pt'))

    # Salva modello finale e storico
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    np.save(os.path.join(save_dir, 'episode_rewards.npy'), np.array(episode_rewards))
    with open(os.path.join(save_dir, 'episode_metrics.json'), 'w') as f:
        json.dump(episode_metrics, f, indent=2, default=str)

    # =====================================================================
    # NEW: Save training_log.csv
    # =====================================================================
    log_df = pd.DataFrame(training_log)
    log_csv_path = os.path.join(save_dir, 'training_log.csv')
    log_df.to_csv(log_csv_path, index=False)
    print(f"\n📄 Training log saved: {log_csv_path}", flush=True)

    # =====================================================================
    # NEW: Save summary metrics.json
    # =====================================================================
    summary = {
        'best_reward': round(float(best_reward), 6),
        'final_portfolio_value': round(float(training_log[-1]['final_portfolio_value']), 2),
        'average_reward': round(float(np.mean(episode_rewards)), 6),
        'average_reward_last_50': round(float(np.mean(episode_rewards[-50:])), 6),
        'best_sharpe': round(float(max(m.get('sharpe_ratio', 0) for m in episode_metrics)), 4),
        'total_episodes': args.n_episodes,
        'final_epsilon': round(float(agent.get_epsilon()), 6),
    }
    summary_path = os.path.join(save_dir, 'metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Summary metrics saved: {summary_path}", flush=True)

    # =====================================================================
    # NEW: Generate and save plots
    # =====================================================================
    episodes_range = range(1, args.n_episodes + 1)

    # Plot 1: Reward per episode
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes_range, episode_rewards, alpha=0.4, linewidth=0.8, label='Episode Reward')
    window = min(50, len(episode_rewards))
    if window > 1:
        rolling_avg = pd.Series(episode_rewards).rolling(window).mean()
        ax.plot(episodes_range, rolling_avg, linewidth=2, color='red', label=f'Rolling Avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    reward_plot_path = os.path.join(save_dir, 'reward_per_episode.png')
    fig.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Plot saved: {reward_plot_path}", flush=True)

    # Plot 2: Portfolio value per episode
    fig, ax = plt.subplots(figsize=(12, 5))
    portfolio_vals = [row['final_portfolio_value'] for row in training_log]
    ax.plot(episodes_range, portfolio_vals, linewidth=1.2, color='green')
    ax.axhline(y=args.initial_balance, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Final Portfolio Value ($)')
    ax.set_title('Portfolio Value per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    portfolio_plot_path = os.path.join(save_dir, 'portfolio_per_episode.png')
    fig.savefig(portfolio_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Plot saved: {portfolio_plot_path}", flush=True)

    # Plot 3: Epsilon decay
    fig, ax = plt.subplots(figsize=(12, 5))
    epsilons = [row['epsilon'] for row in training_log]
    ax.plot(episodes_range, epsilons, linewidth=1.5, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Epsilon Decay')
    ax.grid(True, alpha=0.3)
    epsilon_plot_path = os.path.join(save_dir, 'epsilon_decay.png')
    fig.savefig(epsilon_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Plot saved: {epsilon_plot_path}", flush=True)

    print(f"\n✅ Addestramento completato. Risultati salvati in: {save_dir}")
    print(f"   Miglior reward: {best_reward:.4f}")
    print(f"   File generati:")
    print(f"     - training_log.csv")
    print(f"     - metrics.json")
    print(f"     - reward_per_episode.png")
    print(f"     - portfolio_per_episode.png")
    print(f"     - epsilon_decay.png")

    return save_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Addestramento agente DRL')
    # Dati
    parser.add_argument('--train_data', type=str, default='Data/naphtha_crack_train.csv', help='CSV dati di training')
    parser.add_argument('--save_dir', type=str, default='Data/results_training', help='Cartella per salvare i risultati')
    # Ambiente
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--initial_balance', type=float, default=100000.0)
    parser.add_argument('--transaction_cost', type=float, default=0.001)
    parser.add_argument('--reward_type', type=str, default='pnl', choices=['pnl', 'sharpe', 'sortino'])
    # Agente
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('--hidden1', type=int, default=128)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--hidden3', type=int, default=32)
    # Training
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=100)
    # Feature selection
    parser.add_argument('--feature_groups', type=str, nargs='+', default=None,
                        help='Gruppi di feature da usare (es. naphtha brent crack correlation). '
                             'Default: tutte. Opzioni: naphtha, brent, crack, correlation, beta, '
                             'vol_diff, vol_ratio, momentum_diff, ratio_dist, return_diff, '
                             'naphtha_brent_ratio, naphtha_contribution, brent_contribution, '
                             'naphtha_direction, brent_direction')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
