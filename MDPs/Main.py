import random

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem

from Algorithms.policy_evaluator import iterative_policy_evaluation, get_greedy_policy, evaluate_specific_policy, value_iteration


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
    problem = GridProblem(size)
    play(problem)


def play_cookie_problem():
    size = 3
    problem = CookieProblem(size)
    play(problem)

def run_experiment(size, problem, gamma, theta):
    print(f"Running experiment for {problem.__class__.__name__} of size {size}x{size}")
    V, exec_time = iterative_policy_evaluation(problem, gamma, theta)
    print(f"Execution time: {exec_time:.6f} seconds")
    print("State values:")
    for state in sorted(V.keys()):
        print(f"State {state}: Value {V[state]:.4f}")
    print("\n")

def show_greedy_policy_grid(size):
    print(f"\n--- (Extra) Visualización Política Greedy para GridProblem {size}x{size} ---")
    problem = GridProblem(grid_size=size)
    gamma, theta = 1.0, 1e-9
    V_random, _ = iterative_policy_evaluation(problem, gamma, theta)
    greedy_policy = get_greedy_policy(problem, V_random, gamma)
    action_symbols = {"up": "↑", "down": "↓", "left": "←", "right": "→", None: "G"}
    location_id = 0
    for i in range(size):
        for j in range(size):
            print(f" {action_symbols.get(greedy_policy.get(location_id))} ", end="")
            location_id += 1
        print()


if __name__ == '__main__':
    
    theta = 1e-9
    
    # Almacenamiento de resultados de la parte (d) para usarlos en la parte (g)
    results_d = {}
    problems_info = {}

    # --- Parte (d): Evaluación de Política Aleatoria ---
    print("--- Parte (d): Evaluación de Política Aleatoria ---")
    
    print("\n# GridProblem")
    gamma_grid = 1.0
    for size in range(3, 11):
        problem = GridProblem(grid_size=size)
        V_random, exec_time = iterative_policy_evaluation(problem, gamma_grid, theta)
        key = f"GridProblem_{size}x{size}"
        results_d[key] = V_random
        problems_info[key] = {'problem': problem, 'gamma': gamma_grid}
        print(f"Grid {size}x{size} | Valor Aleatorio: {V_random[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")
        
    print("\n# CookieProblem")
    gamma_cookie = 0.99
    for size in range(3, 11):
        problem = CookieProblem(grid_size=size)
        V_random, exec_time = iterative_policy_evaluation(problem, gamma_cookie, theta)
        key = f"CookieProblem_{size}x{size}"
        results_d[key] = V_random
        problems_info[key] = {'problem': problem, 'gamma': gamma_cookie}
        print(f"Grid {size}x{size} | Valor Aleatorio: {V_random[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n# GamblerProblem")
    gamma_gambler = 1.0
    for prob_head in [0.25, 0.4, 0.55]:
        problem = GamblerProblem(prob_head=prob_head)
        V_random, exec_time = iterative_policy_evaluation(problem, gamma_gambler, theta)
        key = f"GamblerProblem_{prob_head}"
        results_d[key] = V_random
        problems_info[key] = {'problem': problem, 'gamma': gamma_gambler}
        print(f"Prob. cara {prob_head} | Valor Aleatorio: {V_random[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    # --- Parte (f): Efecto del Factor de Descuento (gamma) ---
    print("\n" + "="*50)
    print("--- Parte (f): Efecto de Gamma en el Tiempo de Convergencia ---")
    gamma_bajo = 0.5
    
    problem_f_grid = GridProblem(grid_size=8)
    _, t_alto_grid = iterative_policy_evaluation(problem_f_grid, gamma_grid, theta)
    _, t_bajo_grid = iterative_policy_evaluation(problem_f_grid, gamma_bajo, theta)
    print(f"GridProblem 8x8 | Tiempo con gamma alto ({gamma_grid}): {t_alto_grid:.3f}s | Tiempo con gamma bajo ({gamma_bajo}): {t_bajo_grid:.3f}s")

    problem_f_cookie = CookieProblem(grid_size=8)
    _, t_alto_cookie = iterative_policy_evaluation(problem_f_cookie, gamma_cookie, theta)
    _, t_bajo_cookie = iterative_policy_evaluation(problem_f_cookie, gamma_bajo, theta)
    print(f"CookieProblem 8x8 | Tiempo con gamma alto ({gamma_cookie}): {t_alto_cookie:.3f}s | Tiempo con gamma bajo ({gamma_bajo}): {t_bajo_cookie:.3f}s")
    
    problem_f_gambler = GamblerProblem(prob_head=0.4)
    _, t_alto_gambler = iterative_policy_evaluation(problem_f_gambler, gamma_gambler, theta)
    _, t_bajo_gambler = iterative_policy_evaluation(problem_f_gambler, gamma_bajo, theta)
    print(f"GamblerProblem | Tiempo con gamma alto ({gamma_gambler}): {t_alto_gambler:.3f}s | Tiempo con gamma bajo ({gamma_bajo}): {t_bajo_gambler:.3f}s")

    # --- Parte (g): Evaluación de Política Greedy ---
    print("\n" + "="*50)
    print("--- Parte (g): Evaluación de Política Greedy ---")
    
    print("\n# GridProblem")
    for size in range(3, 11):
        key = f"GridProblem_{size}x{size}"
        problem, gamma, V_random = problems_info[key]['problem'], problems_info[key]['gamma'], results_d[key]
        greedy_policy = get_greedy_policy(problem, V_random, gamma)
        V_greedy, _ = evaluate_specific_policy(problem, greedy_policy, gamma, theta)
        print(f"Grid {size}x{size} | Valor Greedy: {V_greedy[problem.get_initial_state()]:.3f}")

    print("\n# CookieProblem")
    for size in range(3, 11):
        key = f"CookieProblem_{size}x{size}"
        problem, gamma, V_random = problems_info[key]['problem'], problems_info[key]['gamma'], results_d[key]
        greedy_policy = get_greedy_policy(problem, V_random, gamma)
        V_greedy, _ = evaluate_specific_policy(problem, greedy_policy, gamma, theta)
        print(f"Grid {size}x{size} | Valor Greedy: {V_greedy[problem.get_initial_state()]:.3f}")

    print("\n# GamblerProblem")
    for prob_head in [0.25, 0.4, 0.55]:
        key = f"GamblerProblem_{prob_head}"
        problem, gamma, V_random = problems_info[key]['problem'], problems_info[key]['gamma'], results_d[key]
        greedy_policy = get_greedy_policy(problem, V_random, gamma)
        V_greedy, _ = evaluate_specific_policy(problem, greedy_policy, gamma, theta)
        print(f"Prob. cara {prob_head} | Valor Greedy: {V_greedy[problem.get_initial_state()]:.3f}")

    # --- Parte (h): Búsqueda de Valores Óptimos con Value Iteration ---
    print("\n" + "="*50)
    print("--- Parte (h): Búsqueda de Valores Óptimos con Value Iteration ---")
    
    print("\n# GridProblem")
    for size in range(3, 11):
        problem = GridProblem(grid_size=size)
        V_optimal, exec_time = value_iteration(problem, gamma_grid, theta)
        print(f"Grid {size}x{size} | Valor Óptimo: {V_optimal[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")
        
    print("\n# CookieProblem")
    for size in range(3, 11):
        problem = CookieProblem(grid_size=size)
        V_optimal, exec_time = value_iteration(problem, gamma_cookie, theta)
        print(f"Grid {size}x{size} | Valor Óptimo: {V_optimal[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")

    print("\n# GamblerProblem")
    for prob_head in [0.25, 0.4, 0.55]:
        problem = GamblerProblem(prob_head=prob_head)
        V_optimal, exec_time = value_iteration(problem, gamma_gambler, theta)
        print(f"Prob. cara {prob_head} | Valor Óptimo: {V_optimal[problem.get_initial_state()]:.3f} | Tiempo: {exec_time:.3f}s")