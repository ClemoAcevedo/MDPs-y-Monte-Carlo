import random
from collections import defaultdict
from typing import Any, List, Tuple, Dict

class MonteCarlo:
    def __init__(self, actions: List[Any], epsilon: float, gamma: float, seed: int | None = None):
        self.actions = actions
        self.eps = float(epsilon)
        self.gamma = float(gamma)
        self.Q: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.N: Dict[Tuple[Any, Any], int] = defaultdict(int)
        if seed is not None:
            random.seed(seed)

    def select_action(self, state: Any) -> Any:
        if random.random() < self.eps:
            return random.choice(self.actions)
        else:
            q_values = [self.Q.get((state, a), 0.0) for a in self.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def train_one_episode(self, env: Any):
        episode: List[Tuple[Any, Any, float]] = []
        state = env.reset()
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            key = (state, action)
            self.N[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.N[key]
            
    def get_greedy_action(self, state: Any) -> Any:
        q_values = [self.Q.get((state, a), 0.0) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)
