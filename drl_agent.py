"""
Agente Deep Q-Network (DQN) per trading su naphtha crack spread.
Implementa Double DQN con Experience Replay e target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os

# Struttura per le transizioni nel replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """Experience Replay Buffer per stabilizzare il training."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Rete neurale per approssimare la funzione Q."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list):
        super(QNetwork, self).__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DRLAgent:
    """
    Agente Double DQN per il trading algoritmico.
    Usa due reti (policy e target) per ridurre l'overestimation bias.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update: int = 1000,
                 hidden_sizes: list = None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        # Reti policy e target
        self.policy_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def get_epsilon(self) -> float:
        "Calcola epsilon corrente con decadimento esponenziale."
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              np.exp(-self.steps_done / self.epsilon_decay)
        return eps

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        "Seleziona un'azione con politica epsilon-greedy (training) o greedy (eval)."
        if training:
            epsilon = self.get_epsilon()
            if random.random() < epsilon:
                return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Salva una transizione nel replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """Esegue un passo di apprendimento con Double DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-values correnti
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Double DQN: la policy_net sceglie l'azione, la target_net valuta
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping per stabilità
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1

        # Aggiorna target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Salva il modello su disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)

    def load(self, path: str):
        """Carica il modello da disco."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
