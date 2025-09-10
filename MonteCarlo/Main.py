import time
from datetime import datetime
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
from MonteCarlo import MonteCarlo

def evaluate_policy(agent: MonteCarlo, env: Any, num_episodes: int) -> float:
    total_return = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done = env.step(action)
            episode_return += reward
        total_return += episode_return
    return total_return / num_episodes

def run_experiment(
    title: str,
    env: Any,
    num_runs: int,
    train_episodes: int,
    eval_every: int,
    eval_episodes: int,
    epsilon: float,
    gamma: float,
    experiment_tag: str = "",
    grid_width: int | None = None,
    grid_height: int | None = None
) -> Tuple[List[int], List[List[float]], List[MonteCarlo]]:
    print(f"--- Iniciando Experimento: {title} ---")
    t0_exp = time.time()
    all_runs_returns: List[List[float]] = []
    trained_agents: List[MonteCarlo] = []
    for r in range(num_runs):
        t0_run = time.time()
        print(f"  Run {r + 1}/{num_runs}...")
        agent = MonteCarlo(actions=env.action_space, epsilon=epsilon, gamma=gamma, seed=1000 + r)
        checkpoints: List[int] = []
        run_returns: List[float] = []
        for ep in range(1, train_episodes + 1):
            agent.train_one_episode(env)
            if ep == 1 or ep % eval_every == 0:
                avg_return = evaluate_policy(agent, env, eval_episodes)
                checkpoints.append(ep)
                run_returns.append(avg_return)
                print(f"    [Run {r+1:02d}] Ep: {ep:<10,d} | Retorno Promedio: {avg_return:.4f}")
        all_runs_returns.append(run_returns)
        trained_agents.append(agent)
        if title.startswith("Cliff") and grid_width and grid_height:
            plot_cliff_trajectory_console(agent, grid_width, grid_height, title=f"Trayectoria Final - Run {r+1}")
            plot_cliff_trajectory_snapshot(agent, grid_width, grid_height,
                                           title=f"Trayectoria Final - Run {r+1}",
                                           filename=f"{experiment_tag}_final_trajectory_run{r+1:02d}.png")
        t_run = time.time() - t0_run
        print(f"  Run {r + 1} finalizado en {t_run:.1f} segundos.")
    t_exp = time.time() - t0_exp
    print(f"--- Experimento '{title}' completado en {t_exp / 60:.1f} minutos ---\n")
    return checkpoints, all_runs_returns, trained_agents

def plot_results(checkpoints: List[int], all_runs_returns: List[List[float]], title: str, ylabel: str, filename: str):
    avg_returns = np.mean(np.array(all_runs_returns), axis=0)
    plt.figure(figsize=(12, 7))
    for i, ret in enumerate(all_runs_returns):
        plt.plot(checkpoints, ret, alpha=0.4, linewidth=1.0, label=f"Run {i+1}")
    plt.plot(checkpoints, avg_returns, color='blue', linewidth=2.5, label="Promedio", zorder=5)
    plt.xlabel("Episodios de Entrenamiento")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if "Cliff Walking" in title:
        plt.ylim(bottom=-100, top=0)
        plt.title(title + " (Vista Ampliada)")
        zoomed_filename = filename.replace(".png", "_zoomed.png")
        plt.savefig(zoomed_filename, dpi=150)
    plt.close()

def plot_cliff_trajectory_console(agent: MonteCarlo, width: int, height: int, title: str):
    action_arrows = { (1, 0): "↑", (-1, 0): "↓", (0, 1): "→", (0, -1): "←" }
    grid = [["·" for _ in range(width)] for _ in range(height)]
    env = CliffEnv(width=width)
    state = env.reset()
    done = False
    max_steps = height * width
    steps = 0
    while not done and steps < max_steps:
        action = agent.get_greedy_action(state)
        r, c = state
        grid[r][c] = action_arrows.get(action, "?")
        state, _, done = env.step(action)
        steps += 1
    goal_pos = (0, width - 1)
    for c in range(1, width - 1):
        grid[0][c] = "C"
    grid[goal_pos[0]][goal_pos[1]] = "G"
    print(f"\n--- {title} ---")
    for r in reversed(range(height)):
        print(" ".join(f"{cell:^3}" for cell in grid[r]))
    print("-" * (4 * width))

def plot_cliff_trajectory_snapshot(agent: MonteCarlo, width: int, height: int, title: str, filename: str):
    if width is None or height is None:
        print("Advertencia: No se pueden generar snapshots de trayectoria sin dimensiones de grilla.")
        return
    action_arrows = { (1, 0): "↑", (-1, 0): "↓", (0, 1): "→", (0, -1): "←" }
    grid_data = np.full((height, width), 0.5)
    env = CliffEnv(width=width)
    state = env.reset()
    done = False
    trajectory = []
    max_steps = 2 * (width * height)
    steps = 0
    while not done and steps < max_steps:
        action = agent.get_greedy_action(state)
        trajectory.append((state, action))
        state, _, done = env.step(action)
        steps += 1
    for r in range(height):
        for c in range(width):
            if r == 0 and c > 0 and c < width - 1:
                grid_data[r, c] = 0
    for (r, c), action in trajectory:
        grid_data[r, c] = 1
    start_pos = env.reset()
    goal_pos = (0, width - 1)
    plt.figure(figsize=(width * 0.7, height * 0.7))
    ax = plt.gca()
    cmap = plt.get_cmap('RdYlGn', 3)
    ax.imshow(grid_data, cmap=cmap, origin='lower', vmin=0, vmax=1)
    for (r, c), action in trajectory:
        if (r, c) != start_pos and (r, c) != goal_pos:
            ax.text(c, r, action_arrows.get(action, "?"), ha="center", va="center", color="black", fontsize=10)
    ax.text(start_pos[1], start_pos[0], "S", ha="center", va="center", color="white", fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.3", fc="blue", ec="k", lw=1))
    ax.text(goal_pos[1], goal_pos[0], "G", ha="center", va="center", color="white", fontweight='bold',
            bbox=dict(boxstyle="circle,pad=0.3", fc="black", ec="k", lw=1))
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def generate_and_print_blackjack_policy(agent: MonteCarlo, run_num: int, action_space: List[Any]):
    action_map = {action_space[0]: 'H', action_space[1]: 'P'}

    def get_policy_table(usable_ace: bool):
        player_sums = range(21, 11, -1)
        dealer_cards = range(1, 11)
        
        policy_grid = []
        for p_sum in player_sums:
            row = []
            for d_card in dealer_cards:
                state = (p_sum, usable_ace, d_card)
                action = agent.get_greedy_action(state)
                row.append(action_map.get(action, '?'))
            policy_grid.append([str(p_sum)] + row)
        
        headers = ["A"] + [str(d) for d in range(2, 11)]
        return policy_grid, headers

    print(f"\n policy for Blackjack - Run {run_num} ----------")
    
    print("\nPolítica para Manos Duras (Sin As Usable)")
    hard_policy, headers = get_policy_table(usable_ace=False)
    header_line = ["Suma"] + headers
    print(f"{' | '.join(f'{h:^4}' for h in header_line)}")
    print("-" * len(' | '.join(f'{h:^4}' for h in header_line)))
    for row in hard_policy:
        print(f"{' | '.join(f'{item:^4}' for item in row)}")

    print("\nPolítica para Manos Blandas (Con As Usable)")
    soft_policy, headers = get_policy_table(usable_ace=True)
    header_line = ["Suma"] + headers
    print(f"{' | '.join(f'{h:^4}' for h in header_line)}")
    print("-" * len(' | '.join(f'{h:^4}' for h in header_line)))
    for row in soft_policy:
        print(f"{' | '.join(f'{item:^4}' for item in row)}")
    print("-" * len(' | '.join(f'{h:^4}' for h in header_line)))

if __name__ == "__main__":
    now = datetime.now().strftime('%Y%m%d_%H%M')
    bj_checkpoints, bj_all_returns, bj_agents = run_experiment(
        title="Blackjack",
        env=BlackjackEnv(),
        num_runs=5,
        train_episodes=10_000_000,
        eval_every=500_000,
        eval_episodes=100_000,
        epsilon=0.01,
        gamma=1.0,
        experiment_tag=f"blackjack_{now}"
    )
    plot_results(bj_checkpoints, bj_all_returns,
                 "Rendimiento de Monte Carlo en Blackjack",
                 "Retorno Promedio en Evaluación",
                 f"blackjack_perf_{now}.png")

    print("\n" + "="*60)
    print("      GENERANDO POLÍTICAS FINALES DE BLACKJACK")
    print("="*60)
    print("Leyenda: P = Plantarse (Stick), H = Pedir (Hit)\n")
    
    bj_env = BlackjackEnv()
    for i, agent in enumerate(bj_agents):
        generate_and_print_blackjack_policy(agent, run_num=i + 1, action_space=bj_env.action_space)
        
    print("\n" + "="*60 + "\n")
    
    c6_env = CliffEnv(width=6)
    c6_checkpoints, c6_all_returns, c6_agents = run_experiment(
        title="Cliff Walking (width=6)",
        env=c6_env,
        num_runs=5,
        train_episodes=200_000,
        eval_every=1_000,
        eval_episodes=1,
        epsilon=0.1,
        gamma=1.0,
        experiment_tag=f"cliff6_{now}",
        grid_width=6,
        grid_height=4
    )
    plot_results(c6_checkpoints, c6_all_returns,
                 "Rendimiento de MC en Cliff Walking (width=6)",
                 "Retorno Obtenido en Evaluación (1 episodio)",
                 f"cliff6_perf_{now}.png")
                 
    c12_env = CliffEnv(width=12)
    c12_checkpoints, c12_all_returns, c12_agents = run_experiment(
        title="Cliff Walking (width=12)",
        env=c12_env,
        num_runs=5,
        train_episodes=200_000,
        eval_every=1_000,
        eval_episodes=1,
        epsilon=0.1,
        gamma=1.0,
        experiment_tag=f"cliff12_{now}",
        grid_width=12,
        grid_height=4
    )
    plot_results(c12_checkpoints, c12_all_returns,
                 "Rendimiento de MC en Cliff Walking (width=12)",
                 "Retorno Obtenido en Evaluación (1 episodio)",
                 f"cliff12_{now}.png")