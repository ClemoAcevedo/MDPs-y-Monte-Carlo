import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class MonteCarlo:
    """
    Algoritmo de control Monte Carlo con política on-policy y actualizaciones every-visit.
    Utiliza una política epsilon-greedy para la exploración y promedio incremental para
    actualizar los valores de Q.
    """
    
    def __init__(self, env: Any, epsilon: float, gamma: float):
        """
        Inicializa el agente Monte Carlo.
        
        Args:
            env: El entorno de aprendizaje (debe tener métodos reset, step y action_space).
            epsilon: Tasa de exploración para la política epsilon-greedy.
            gamma: Factor de descuento para recompensas futuras.
        """
        self.env = env
        self.eps = epsilon
        self.gamma = gamma
        self.actions = list(self.env.action_space)
        
        # Q(s,a) -> valor de la acción 'a' en el estado 's'
        self.Q = defaultdict(float)
        # N(s,a) -> contador de visitas para el par estado-acción
        self.N = defaultdict(int)

    def get_best_action(self, state: Any) -> Any:
        """
        Encuentra la mejor acción para un estado dado según los valores Q actuales.
        OPTIMIZACIÓN: Una sola pasada por las acciones
        """
        best_actions = []
        max_q_value = float('-inf')
        
        # Una sola pasada: encontrar máximo Y recopilar mejores acciones
        for action in self.actions:
            q_value = self.Q[(state, action)]
            
            if q_value > max_q_value:
                # Nueva mejor acción encontrada
                max_q_value = q_value
                best_actions = [action]  # Reiniciar lista
            elif q_value == max_q_value:
                # Empate con la mejor acción actual
                best_actions.append(action)
        
        return random.choice(best_actions)

    def select_action(self, state: Any) -> Any:
        """
        Selecciona una acción utilizando la política epsilon-greedy.
        Con probabilidad epsilon, elige una acción al azar (exploración).
        Con probabilidad 1-epsilon, elige la mejor acción actual (explotación).
        """
        if random.random() < self.eps:
            return random.choice(self.actions)  # Exploración
        else:
            return self.get_best_action(state)  # Explotación

    def generate_episode(self) -> List[Tuple[Any, Any, float]]:
        """
        Genera un episodio completo siguiendo la política epsilon-greedy.
        Un episodio es una secuencia de (estado, acción, recompensa).
        """
        trajectory = []
        state = self.env.reset()
        done = False
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            
        return trajectory

    def update_q_values(self, trajectory: List[Tuple[Any, Any, float]]) -> None:
        """
        Actualiza los valores Q utilizando Monte Carlo every-visit con promedio incremental.
        Recorre el episodio hacia atrás para calcular los retornos (G) y actualizar Q.
        
        La fórmula de actualización es:
        Q(s,a) = Q(s,a) + (1/N(s,a)) * [G - Q(s,a)]
        
        Donde:
        - G es el retorno descontado desde el tiempo t
        - N(s,a) es el número de veces que hemos visitado el par (s,a)
        """
        G = 0.0  # Retorno acumulado
        
        # Iterar hacia atrás desde el final del episodio
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            
            # Calcular el retorno G_t = R_{t+1} + γ*G_{t+1}
            G = self.gamma * G + reward
            
            # Actualización Every-Visit: se actualiza cada vez que se visita el par (s,a)
            sa_pair = (state, action)
            self.N[sa_pair] += 1
            
            # Fórmula de promedio incremental
            # Nuevo_promedio = Viejo_promedio + (1/N) * (Nuevo_valor - Viejo_promedio)
            alpha = 1.0 / self.N[sa_pair]
            self.Q[sa_pair] += alpha * (G - self.Q[sa_pair])

    def evaluate(self, n_episodes: int) -> float:
        """
        Evalúa la política greedy actual corriendo 'n_episodes' sin exploración.
        
        Args:
            n_episodes: El número de episodios a promediar para la evaluación.
            
        Returns:
            El retorno promedio obtenido durante la evaluación.
        """
        total_return = 0.0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                # Actuar siempre de forma greedy usando la mejor acción conocida
                action = self.get_best_action(state)
                state, reward, done = self.env.step(action)
                episode_return += reward
                
            total_return += episode_return
            
        return total_return / n_episodes

    def get_policy(self, state: Any) -> Any:
        """
        Retorna la acción que tomaría la política greedy en un estado dado.
        Útil para análisis posterior de la política aprendida.
        """
        return self.get_best_action(state)
    
    def get_q_value(self, state: Any, action: Any) -> float:
        """
        Retorna el valor Q para un par estado-acción dado.
        """
        return self.Q[(state, action)]
    
    def get_visit_count(self, state: Any, action: Any) -> int:
        """
        Retorna cuántas veces se ha visitado un par estado-acción.
        """
        return self.N[(state, action)]