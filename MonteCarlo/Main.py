import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
from MonteCarlo import MonteCarlo

def plot_results(checkpoints, all_runs_returns, avg_returns, title, ylabel, filename):
    """
    Función mejorada para graficar los resultados.
    Dibuja cada corrida individualmente con transparencia y el promedio de forma destacada.
    """
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Paleta de colores más profesional
    individual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    average_color = '#e377c2'
    
    # Dibujar corridas individuales
    for i, run_returns in enumerate(all_runs_returns):
        ax.plot(checkpoints, run_returns, 
                color=individual_colors[i % len(individual_colors)], 
                linestyle='-', alpha=0.6, linewidth=1.5,
                label=f'Corrida {i+1}')

    # Dibujar promedio destacado
    ax.plot(checkpoints, avg_returns, 
            color=average_color, marker='o', markersize=4,
            linestyle='-', linewidth=3.0, alpha=0.9,
            label=f'Promedio ({len(all_runs_returns)} corridas)',
            markerfacecolor=average_color, markeredgecolor='white', markeredgewidth=1)
    
    # Configurar ejes y etiquetas
    ax.set_xlabel("Episodios de Entrenamiento", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid sutil
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Leyenda mejorada
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
              fontsize=10, ncol=1 if len(all_runs_returns) <= 3 else 2)
    
    # Formatear eje X para números grandes
    def format_episodes(x, p):
        if x >= 1_000_000:
            return f'{int(x/1_000_000)}M'
        elif x >= 1_000:
            return f'{int(x/1_000)}k'
        else:
            return f'{int(x)}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_episodes))
    
    # Añadir estadísticas en el gráfico
    final_performance = avg_returns[-1]
    initial_performance = avg_returns[0]
    improvement = final_performance - initial_performance
    
    stats_text = f'Rendimiento final: {final_performance:.3f}\nMejora total: {improvement:+.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {filename}")
    plt.show()


def plot_cliff_results(checkpoints, all_runs_returns, avg_returns, title, ylabel, filename):
    """
    Función especializada para Cliff Walking con valores muy negativos.
    """
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    individual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    average_color = '#e377c2'
    
    # === GRÁFICO 1: VISTA COMPLETA ===
    ax1.set_title(f"{title} - Vista Completa", fontsize=12, fontweight='bold')
    
    for i, run_returns in enumerate(all_runs_returns):
        ax1.plot(checkpoints, run_returns, 
                color=individual_colors[i % len(individual_colors)], 
                linestyle='-', alpha=0.6, linewidth=1.5,
                label=f'Corrida {i+1}')
    
    ax1.plot(checkpoints, avg_returns, 
            color=average_color, marker='o', markersize=3,
            linestyle='-', linewidth=3.0, alpha=0.9,
            label='Promedio', markerfacecolor=average_color, 
            markeredgecolor='white', markeredgewidth=1)
    
    ax1.set_xlabel("Episodios", fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # === GRÁFICO 2: ÚLTIMOS 50% EPISODIOS ===
    zoom_start = len(checkpoints) // 2
    zoom_checkpoints = checkpoints[zoom_start:]
    zoom_avg = avg_returns[zoom_start:]
    zoom_runs = [run[zoom_start:] for run in all_runs_returns]
    
    ax2.set_title(f"Últimos 50% de Episodios - Detalle", fontsize=12, fontweight='bold')
    
    for i, run_returns in enumerate(zoom_runs):
        ax2.plot(zoom_checkpoints, run_returns, 
                color=individual_colors[i % len(individual_colors)], 
                linestyle='-', alpha=0.6, linewidth=1.5)
    
    ax2.plot(zoom_checkpoints, zoom_avg, 
            color=average_color, marker='o', markersize=3,
            linestyle='-', linewidth=3.0, alpha=0.9,
            markerfacecolor=average_color, markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlabel("Episodios", fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Estadísticas
    improvement = avg_returns[-1] - avg_returns[0]
    ax2.text(0.02, 0.02, f'Mejora total: {improvement:+.1f}', 
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
             verticalalignment='bottom')
    
    # Formatear ejes X
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'
        ))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico Cliff guardado en: {filename}")
    plt.show()


def format_time(seconds):
    """Formatear tiempo en formato legible"""
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutos"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} horas"


def print_timing_summary(experiment_times):
    """Imprimir resumen completo de tiempos"""
    print("\n" + "="*60)
    print("RESUMEN DE TIEMPOS DE EJECUCIÓN")
    print("="*60)
    
    total_time = sum(data['total'] for data in experiment_times.values())
    
    for experiment, time_data in experiment_times.items():
        print(f"\n{experiment.upper()}:")
        print(f"   Tiempo total: {format_time(time_data['total'])}")
        print(f"   Tiempo por corrida: {format_time(time_data['per_run'])}")
        print(f"   Episodios totales: {time_data['total_episodes']:,}")
        print(f"   Episodios/segundo: {time_data['episodes_per_sec']:.0f}")
    
    print(f"\nTIEMPO TOTAL: {format_time(total_time)}")
    print("="*60)


if __name__ == "__main__":
    
    experiment_times = {}
    
    print("INICIANDO EXPERIMENTOS DE MONTE CARLO")
    print("="*50)
    
    # ===============================================
    # Experimento 1: Blackjack
    # ===============================================
    print("\nIniciando experimento de Blackjack...")
    
    BJ_EPISODES = 10_000_000
    BJ_RUNS = 5
    BJ_EVAL_EVERY = 500_000
    BJ_TEST_EPISODES = 100_000

    blackjack_start_time = time.time()
    
    blackjack_checkpoints = [1] + list(range(BJ_EVAL_EVERY, BJ_EPISODES + 1, BJ_EVAL_EVERY))
    all_blackjack_returns = []
    blackjack_run_times = []

    for i in range(BJ_RUNS):
        run_start_time = time.time()
        print(f"  Blackjack - Corrida {i + 1}/{BJ_RUNS}...")
        
        env = BlackjackEnv()
        agent = MonteCarlo(env, epsilon=0.01, gamma=1.0)
        run_returns = []

        # Evaluación inicial (episodio 1)
        trajectory = agent.generate_episode()
        agent.update_q_values(trajectory)
        performance = agent.evaluate(n_episodes=BJ_TEST_EPISODES)
        run_returns.append(performance)
        print(f"    Episodio 1: Retorno = {performance:.4f}")

        # Entrenamiento principal
        for episode in range(2, BJ_EPISODES + 1):
            trajectory = agent.generate_episode()
            agent.update_q_values(trajectory)

            if episode in blackjack_checkpoints:
                performance = agent.evaluate(n_episodes=BJ_TEST_EPISODES)
                run_returns.append(performance)
                if episode % (BJ_EVAL_EVERY * 4) == 0:
                    print(f"    Episodio {episode:,}: Retorno = {performance:.4f}")
        
        run_duration = time.time() - run_start_time
        blackjack_run_times.append(run_duration)
        all_blackjack_returns.append(run_returns)
        print(f"    Corrida {i+1} completada en: {format_time(run_duration)}")

    # Estadísticas de tiempo
    blackjack_total_time = time.time() - blackjack_start_time
    blackjack_avg_time = np.mean(blackjack_run_times)
    
    experiment_times['Blackjack'] = {
        'total': blackjack_total_time,
        'per_run': blackjack_avg_time,
        'total_episodes': BJ_EPISODES * BJ_RUNS,
        'episodes_per_sec': (BJ_EPISODES * BJ_RUNS) / blackjack_total_time,
    }
    
    avg_blackjack_returns = np.mean(all_blackjack_returns, axis=0)
    
    print(f"\nBLACKJACK COMPLETADO:")
    print(f"   Tiempo total: {format_time(blackjack_total_time)}")
    print(f"   Rendimiento inicial: {avg_blackjack_returns[0]:.4f}")
    print(f"   Rendimiento final: {avg_blackjack_returns[-1]:.4f}")
    print(f"   Mejora: {avg_blackjack_returns[-1] - avg_blackjack_returns[0]:+.4f}")
    
    # Graficar Blackjack
    plot_results(
        blackjack_checkpoints,
        all_blackjack_returns,
        avg_blackjack_returns,
        "Rendimiento de Monte Carlo en Blackjack",
        "Retorno Promedio (evaluación greedy con ε=0)",
        f"blackjack_results_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    )

    # ===============================================
    # Experimento 2: Cliff Walking (width=6)
    # ===============================================
    print("\nIniciando experimento de Cliff Walking (width=6)...")
    
    CLIFF_EPISODES = 200_000
    CLIFF_RUNS = 5
    CLIFF_EVAL_EVERY = 1_000
    
    cliff_start_time = time.time()
    
    cliff_checkpoints = [1] + list(range(CLIFF_EVAL_EVERY, CLIFF_EPISODES + 1, CLIFF_EVAL_EVERY))
    all_cliff_returns = []
    cliff_run_times = []

    for i in range(CLIFF_RUNS):
        run_start_time = time.time()
        print(f"  Cliff (w=6) - Corrida {i + 1}/{CLIFF_RUNS}...")
        
        env = CliffEnv(width=6)
        agent = MonteCarlo(env, epsilon=0.1, gamma=1.0)
        run_returns = []

        # Evaluación inicial
        trajectory = agent.generate_episode()
        agent.update_q_values(trajectory)
        performance = agent.evaluate(n_episodes=1)
        run_returns.append(performance)

        # Entrenamiento principal
        for episode in range(2, CLIFF_EPISODES + 1):
            trajectory = agent.generate_episode()
            agent.update_q_values(trajectory)

            if episode in cliff_checkpoints:
                performance = agent.evaluate(n_episodes=1)
                run_returns.append(performance)
        
        run_duration = time.time() - run_start_time
        cliff_run_times.append(run_duration)
        all_cliff_returns.append(run_returns)
        print(f"    Corrida {i+1} completada - Retorno final: {run_returns[-1]:.1f}")

    cliff_total_time = time.time() - cliff_start_time
    cliff_avg_time = np.mean(cliff_run_times)
    
    experiment_times['Cliff Walking (w=6)'] = {
        'total': cliff_total_time,
        'per_run': cliff_avg_time,
        'total_episodes': CLIFF_EPISODES * CLIFF_RUNS,
        'episodes_per_sec': (CLIFF_EPISODES * CLIFF_RUNS) / cliff_total_time,
    }

    avg_cliff_returns = np.mean(all_cliff_returns, axis=0)

    print(f"\nCLIFF (w=6) COMPLETADO:")
    print(f"   Retorno inicial: {avg_cliff_returns[0]:.2f}")
    print(f"   Retorno final: {avg_cliff_returns[-1]:.2f}")
    print(f"   Mejora: {avg_cliff_returns[-1] - avg_cliff_returns[0]:+.2f}")

    # Graficar Cliff
    plot_cliff_results(
        cliff_checkpoints,
        all_cliff_returns,
        avg_cliff_returns,
        "Rendimiento de Monte Carlo en Cliff Walking (width=6)",
        "Retorno (evaluación greedy con ε=0)",
        f"cliff_w6_results_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    )

    # ===============================================
    # Experimento 3: Cliff Walking (width=12)
    # ===============================================
    print("\nIniciando experimento de Cliff Walking (width=12)...")
    
    cliff12_start_time = time.time()
    cliff12_checkpoints = [1] + list(range(CLIFF_EVAL_EVERY, CLIFF_EPISODES + 1, CLIFF_EVAL_EVERY))
    all_cliff12_returns = []
    cliff12_run_times = []

    for i in range(CLIFF_RUNS):
        run_start_time = time.time()
        print(f"  Cliff (w=12) - Corrida {i + 1}/{CLIFF_RUNS}...")

        env12 = CliffEnv(width=12)
        agent12 = MonteCarlo(env12, epsilon=0.1, gamma=1.0)
        run_returns = []

        # Evaluación inicial
        trajectory = agent12.generate_episode()
        agent12.update_q_values(trajectory)
        performance = agent12.evaluate(n_episodes=1)
        run_returns.append(performance)

        # Entrenamiento principal
        for episode in range(2, CLIFF_EPISODES + 1):
            trajectory = agent12.generate_episode()
            agent12.update_q_values(trajectory)

            if episode in cliff12_checkpoints:
                performance = agent12.evaluate(n_episodes=1)
                run_returns.append(performance)

        run_duration = time.time() - run_start_time
        cliff12_run_times.append(run_duration)
        all_cliff12_returns.append(run_returns)
        print(f"    Corrida {i+1} completada - Retorno final: {run_returns[-1]:.1f}")

    cliff12_total_time = time.time() - cliff12_start_time
    cliff12_avg_time = np.mean(cliff12_run_times)
    
    experiment_times['Cliff Walking (w=12)'] = {
        'total': cliff12_total_time,
        'per_run': cliff12_avg_time,
        'total_episodes': CLIFF_EPISODES * CLIFF_RUNS,
        'episodes_per_sec': (CLIFF_EPISODES * CLIFF_RUNS) / cliff12_total_time,
    }

    avg_cliff12_returns = np.mean(all_cliff12_returns, axis=0)
    
    print(f"\nCLIFF (w=12) COMPLETADO:")
    print(f"   Retorno inicial: {avg_cliff12_returns[0]:.2f}")
    print(f"   Retorno final: {avg_cliff12_returns[-1]:.2f}")
    print(f"   Mejora: {avg_cliff12_returns[-1] - avg_cliff12_returns[0]:+.2f}")

    # Graficar Cliff w=12
    plot_cliff_results(
        cliff12_checkpoints,
        all_cliff12_returns,
        avg_cliff12_returns,
        "Rendimiento de Monte Carlo en Cliff Walking (width=12)",
        "Retorno (evaluación greedy con ε=0)",
        f"cliff_w12_results_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    )

    # ===============================================
    # Resumen final
    # ===============================================
    print_timing_summary(experiment_times)