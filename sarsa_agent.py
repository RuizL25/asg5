import random
from collections import defaultdict


class SARSA:
    """
    Agente SARSA (on-policy TD control).

    Atributos
    ---------
    epsilon : float
        Probabilidad de exploración en la estrategia epsilon-greedy (default 0.9).
    gamma : float
        Factor de descuento para recompensas futuras (default 0.96).
    alpha : float
        Tasa de aprendizaje (default 0.81).
    Q : defaultdict
        Tabla de Q-valores: Q[(estado, accion)] -> float.
    env : objeto
        Referencia al ambiente con el que interactúa el agente.
    """

    def __init__(self, env, epsilon=0.9, gamma=0.96, alpha=0.81):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(float)  # Q[(s, a)] = 0.0 por defecto
        self.env = env

    def choose_action(self, state):
        """
        Selecciona una acción siguiendo la política epsilon-greedy.

        Con probabilidad epsilon elige aleatoriamente (exploración).
        Con probabilidad (1 - epsilon) elige la acción con mayor Q-valor (explotación).

        Parámetros
        ----------
        state : hashable
            Estado actual del agente.

        Retorna
        -------
        int
            Índice de la acción seleccionada.
        """
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        else:
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            max_q = max(q_values)
            # Romper empates aleatoriamente
            best_actions = [a for a, q in zip(self.env.actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def action_function(self, s1, a1, reward, s2, a2):
        """
        Actualiza Q(s1, a1) usando la regla de actualización SARSA.

        Q(s1, a1) <- (1 - alpha) * Q(s1, a1) + alpha * (reward + gamma * Q(s2, a2))

        Parámetros
        ----------
        s1 : hashable
            Estado actual.
        a1 : int
            Acción ejecutada desde s1.
        reward : float
            Recompensa recibida al transitar de s1 a s2.
        s2 : hashable
            Estado siguiente.
        a2 : int
            Acción seleccionada para s2 (on-policy: la que se ejecutará a continuación).
        """
        current_q = self.Q[(s1, a1)]
        next_q = self.Q[(s2, a2)]
        self.Q[(s1, a1)] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_q)
