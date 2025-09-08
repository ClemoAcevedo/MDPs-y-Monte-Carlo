import time

def iterative_policy_evaluation(problem, gamma, theta):
    V = {s: 0.0 for s in problem.states}
    t0 = time.time()

    while True:
        delta = 0.0
        for s in V.keys():
            old_v = V[s]
            if problem.is_terminal(s):
                V[s] = 0.0
                continue

            new_v = 0.0
            available_actions = problem.get_available_actions(s)
            action_prob = 1.0 / len(available_actions)

            for action in available_actions:
                transitions = problem.get_transitions(s, action)
                expected_value_for_action = 0
                for (prob, s_next, reward) in transitions:
                    expected_value_for_action += prob * (reward + gamma * V[s_next])
                new_v += action_prob * expected_value_for_action
            
            V[s] = new_v
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break
            
    execution_time = time.time() - t0
    return V, execution_time

def get_greedy_policy(problem, V, gamma):
    policy = {}
    for s in V.keys():
        if problem.is_terminal(s):
            continue

        available_actions = problem.get_available_actions(s)
        best_action = None
        max_action_value = -float('inf')

        for action in available_actions:
            action_value = 0
            transitions = problem.get_transitions(s, action)
            for (prob, s_next, reward) in transitions:
                action_value += prob * (reward + gamma * V[s_next])
    
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = action
        
        policy[s] = best_action
        
    return policy

def evaluate_specific_policy(problem, policy, gamma, theta):
    V = {s: 0.0 for s in problem.states}
    t0 = time.time()

    while True:
        delta = 0.0
        for s in V.keys():
            old_v = V[s]
            if problem.is_terminal(s):
                V[s] = 0.0
                continue
            
            action = policy.get(s)
        
            new_v = 0
            if action:
                transitions = problem.get_transitions(s, action)
                for (prob, s_next, reward) in transitions:
                    new_v += prob * (reward + gamma * V[s_next])
            
            V[s] = new_v
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break
            
    execution_time = time.time() - t0
    return V, execution_time


def value_iteration(problem, gamma, theta):
    V = {s: 0.0 for s in problem.states}
    t0 = time.time()

    while True:
        delta = 0.0
        for s in V.keys():
            old_v = V[s]
            if problem.is_terminal(s):
                V[s] = 0.0
                continue

            available_actions = problem.get_available_actions(s)
            max_action_value = -float('inf')

            for action in available_actions:
                action_value = 0
                transitions = problem.get_transitions(s, action)
                for (prob, s_next, reward) in transitions:
                    action_value += prob * (reward + gamma * V[s_next])
                
                max_action_value = max(max_action_value, action_value)
            
            new_v = max_action_value if available_actions else 0.0
            V[s] = new_v
            
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break
            
    execution_time = time.time() - t0
    return V, execution_time


def get_all_optimal_actions(problem, V, gamma):
    policy = {}
    for s in V.keys():
        if problem.is_terminal(s):
            continue

        available_actions = problem.get_available_actions(s)
        if not available_actions:
            continue

        action_values = {}
        for action in available_actions:
            q_value = 0
            transitions = problem.get_transitions(s, action)
            for (prob, s_next, reward) in transitions:
                q_value += prob * (reward + gamma * V[s_next])
            action_values[action] = q_value
        
        max_q_value = -float('inf')
        for q in action_values.values():
            max_q_value = max(max_q_value, round(q, 5))
            
        optimal_actions = []
        for action, q in action_values.items():
            if round(q, 5) == max_q_value:
                optimal_actions.append(action)
        
        policy[s] = optimal_actions
        
    return policy