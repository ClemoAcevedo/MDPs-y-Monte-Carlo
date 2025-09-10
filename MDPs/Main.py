import math
import random
import matplotlib.pyplot as plt

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem

from algorithms import (
    evaluate_uniform_policy,
    get_greedy_policy,
    evaluate_specific_policy,
    value_iteration,
    get_all_optimal_actions,
)  

def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        action = get_action_from_user(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_gambler_problem():
    p = 0.4
    problem = GamblerProblem(p)
    play(problem)


def play_grid_problem():
    size = 4
    problem = GridProblem(grid_size=size)
    play(problem)


def play_cookie_problem():
    size = 3
    problem = CookieProblem(grid_size=size)
    play(problem)

def check_and_report(key, problem, gamma, V_random, V_star, theta):
    greedy_policy = get_greedy_policy(problem, V_random, gamma)
    V_greedy, _ = evaluate_specific_policy(problem, greedy_policy, gamma, theta)
    s0 = problem.get_initial_state()
    optimal_actions = get_all_optimal_actions(problem, V_star, gamma)
    is_optimal = True
    for state, action in greedy_policy.items():
        if action not in optimal_actions.get(state, []):
            is_optimal = False
            break  

    print(f"{key} | Valor Greedy s0: {V_greedy[s0]:.3f} | Óptima: {'Sí' if is_optimal else 'No'}")


def plot_gambler_policy(policy, prob_head):
    states, actions = [], []
    for state in sorted(policy.keys()):
        for action in sorted(policy[state]):
            states.append(state)
            actions.append(action)

    plt.figure(figsize=(12, 7))
    plt.plot(states, actions, '.', markersize=4)
    plt.title(f'Políticas Óptimas para GamblerProblem (Probabilidad = {prob_head})')
    plt.xlabel('Estado (Capital)')
    plt.ylabel('Acción (Apuesta)')
    plt.grid(True, linestyle='--', alpha=0.6)
    file_name = f'gambler_policy_ph_{prob_head}.png'
    plt.savefig(file_name)
    print(f"Gráfico guardado en: {file_name}")
    plt.close()

def print_gambler_policy(policy, prob_head):
    print(f"\n--- Políticas Óptimas para Prob. Cara = {prob_head} ---")
    for state in sorted(policy.keys()):
        actions = policy[state]
        print(f"  Estado {state:3d}: Apuestas Óptimas {actions}")


if __name__ == '__main__':
    theta = 0.0000000001
    print("--- Parte (d): Evaluación de Política Aleatoria ---")
    results_d = {}       
    problems_info = {}  

    print("\n# GridProblem")
    gamma_grid = 1.0
    for size in range(3, 11):
        problem = GridProblem(grid_size=size)
        V_random, exec_time = evaluate_uniform_policy(problem, gamma_grid, theta)
        key = f"GridProblem_{size}x{size}"
        results_d[key] = V_random
        problems_info[key] = {'problem': problem, 'gamma': gamma_grid}
        print(f"Grid {size}x{size} | Valor Aleatorio: {V_random[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n# CookieProblem")
    gamma_cookie = 0.99
    for size in range(3, 11):
        problem = CookieProblem(grid_size=size)
        V_random, exec_time = evaluate_uniform_policy(problem, gamma_cookie, theta)
        key = f"CookieProblem_{size}x{size}"
        results_d[key] = V_random
        problems_info[key] = {'problem': problem, 'gamma': gamma_cookie}
        print(f"Grid {size}x{size} | Valor Aleatorio: {V_random[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n# GamblerProblem")
    gamma_gambler = 1.0
    for prob_head in [0.25, 0.4, 0.55]:
        problem = GamblerProblem(prob_head=prob_head)
        V_random, exec_time = evaluate_uniform_policy(problem, gamma_gambler, theta)
        key = f"GamblerProblem_{prob_head}"
        results_d[key] = V_random
        problems_info[key] = {'problem': problem, 'gamma': gamma_gambler}
        print(f"Prob. cara {prob_head} | Valor Aleatorio: {V_random[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n" + "="*50)
    print("--- Parte (f): Efecto de Gamma en el Tiempo de Convergencia ---")
    gamma_bajo = 0.5

    problem_f_grid = GridProblem(grid_size=8)
    _, t_alto_grid = evaluate_uniform_policy(problem_f_grid, gamma_grid, theta)
    _, t_bajo_grid = evaluate_uniform_policy(problem_f_grid, gamma_bajo, theta)
    print(f"GridProblem 8x8 | Tiempo con gamma alto ({gamma_grid}): {t_alto_grid:.3f}s | Tiempo con gamma bajo ({gamma_bajo}): {t_bajo_grid:.3f}s")

    problem_f_cookie = CookieProblem(grid_size=8)
    _, t_alto_cookie = evaluate_uniform_policy(problem_f_cookie, gamma_cookie, theta)
    _, t_bajo_cookie = evaluate_uniform_policy(problem_f_cookie, gamma_bajo, theta)
    print(f"CookieProblem 8x8 | Tiempo con gamma alto ({gamma_cookie}): {t_alto_cookie:.3f}s | Tiempo con gamma bajo ({gamma_bajo}): {t_bajo_cookie:.3f}s")

    problem_f_gambler = GamblerProblem(prob_head=0.4)
    _, t_alto_gambler = evaluate_uniform_policy(problem_f_gambler, gamma_gambler, theta)
    _, t_bajo_gambler = evaluate_uniform_policy(problem_f_gambler, gamma_bajo, theta)
    print(f"GamblerProblem | Tiempo con gamma alto ({gamma_gambler}): {t_alto_gambler:.3f}s | Tiempo con gamma bajo ({gamma_bajo}): {t_bajo_gambler:.3f}s")

    print("\n" + "="*50)
    print("--- Parte (h): Búsqueda de Valores Óptimos con Value Iteration ---")
    results_h_optimal_V = {}  

    print("\n# GridProblem")
    for size in range(3, 11):
        problem = GridProblem(grid_size=size)
        V_optimal, exec_time = value_iteration(problem, gamma_grid, theta)
        key = f"GridProblem_{size}x{size}"
        results_h_optimal_V[key] = V_optimal
        print(f"Grid {size}x{size} | Valor Óptimo: {V_optimal[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n# CookieProblem")
    for size in range(3, 11):
        problem = CookieProblem(grid_size=size)
        V_optimal, exec_time = value_iteration(problem, gamma_cookie, theta)
        key = f"CookieProblem_{size}x{size}"
        results_h_optimal_V[key] = V_optimal
        print(f"Grid {size}x{size} | Valor Óptimo: {V_optimal[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n# GamblerProblem")
    for prob_head in [0.25, 0.4, 0.55]:
        problem = GamblerProblem(prob_head=prob_head)
        V_optimal, exec_time = value_iteration(problem, gamma_gambler, theta)
        key = f"GamblerProblem_{prob_head}"
        results_h_optimal_V[key] = V_optimal
        print(f"Prob. cara {prob_head} | Valor Óptimo: {V_optimal[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n" + "="*50)
    print("--- Parte (g): Evaluación de Política Greedy ---")

    print("\n# GridProblem")
    for size in range(3, 11):
        key = f"GridProblem_{size}x{size}"
        problem, gamma, V_random = problems_info[key]['problem'], problems_info[key]['gamma'], results_d[key]
        V_star = results_h_optimal_V[key]
        check_and_report(key, problem, gamma, V_random, V_star, theta)

    print("\n# CookieProblem")
    for size in range(3, 11):
        key = f"CookieProblem_{size}x{size}"
        problem, gamma, V_random = problems_info[key]['problem'], problems_info[key]['gamma'], results_d[key]
        V_star = results_h_optimal_V[key]
        check_and_report(key, problem, gamma, V_random, V_star, theta)

    print("\n# GamblerProblem")
    for prob_head in [0.25, 0.4, 0.55]:
        key = f"GamblerProblem_{prob_head}"
        problem, gamma, V_random = problems_info[key]['problem'], problems_info[key]['gamma'], results_d[key]
        V_star = results_h_optimal_V[key]
        check_and_report(key, problem, gamma, V_random, V_star, theta)

    print("\n" + "="*50)
    print("--- Parte (i): Gráficos de Políticas Óptimas para GamblerProblem ---")
    for prob_head in [0.25, 0.4, 0.55]:
        print(f"Generando gráfico para Prob. Cara = {prob_head}...")
        key = f"GamblerProblem_{prob_head}"
        V_optimal = results_h_optimal_V[key]
        problem = GamblerProblem(prob_head=prob_head)
        optimal_policies = get_all_optimal_actions(problem, V_optimal, gamma_gambler)
        plot_gambler_policy(optimal_policies, prob_head)
        print_gambler_policy(optimal_policies, prob_head)
