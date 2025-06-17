# import numpy as np
# from scipy.optimize import linprog

# class Agent:
#     def __init__(self, environment, is_A=True, random_policy=False, **kwargs):
#         self.environment = environment
#         self.is_A = is_A
#         self.random_policy = random_policy

#         self.lr = kwargs.get('lr', 1.0)
#         self.exp_rate = kwargs.get('exp_rate', 0.2)
#         self.gamma = kwargs.get('gamma', 0.9)
#         self.decay = kwargs.get('decay', 0.9999954)

#         self.state_idx = {
#             f'({i}, {j})': i * self.environment.n_cols + j for i in range(self.environment.n_rows) for j in range(self.environment.n_cols)
#         }

#         self.action_idx = {'N': 0, 'S': 1, 'W': 2, 'E': 3, 'STAND': 4}
#         self.idx_action = {v: k for k, v in self.action_idx.items()}
#         self.idx_state = {k: v for v, k in self.state_idx.items()}

#         S = len(self.state_idx)
#         A = len(self.action_idx)

#         self.Q_vals = np.zeros((S, S, A, A), dtype=np.float32)
#         self.V_vals = np.ones((S, S), dtype=np.float32)
#         self.pi = np.ones((S, S, A), dtype=np.float32)
#         self.pi /= self.pi.sum(axis=2, keepdims=True)

#         self.belief_count = np.zeros((S, S, A), dtype=np.int16)
#         self.belief_prob = np.ones((S, S, A), dtype=np.float32)
#         self.belief_prob /= self.belief_prob.sum(axis=2, keepdims=True)

#     def choose_action(self):
#         state_a = self.environment.a_state
#         state_b = self.environment.b_state
#         s_a_idx = self.state_idx[str(state_a)]
#         s_b_idx = self.state_idx[str(state_b)]

#         if self.random_policy or np.random.rand() < self.exp_rate:
#             return np.random.choice(list(self.action_idx.keys()))
#         else:
#             probs = self.pi[s_a_idx, s_b_idx]
#             return np.random.choice(list(self.action_idx.keys()), p=probs)

#     def update_belief(self, old_state,action_opponent): 
#         s_a_idx_old = self.state_idx[str(old_state[0])]
#         s_b_idx_old = self.state_idx[str(old_state[1])]
#         b_idx = self.action_idx[action_opponent]  

#         self.belief_count[s_a_idx_old, s_b_idx_old, b_idx] += 1
#         counts = self.belief_count[s_a_idx_old, s_b_idx_old]
#         total = counts.sum()
#         if total > 0:
#             self.belief_prob[s_a_idx_old, s_b_idx_old] = counts / total

#     def update_policy(self, old_state):
#         s_a_idx = self.state_idx[str(old_state[0])]
#         s_b_idx = self.state_idx[str(old_state[1])]
#         A = len(self.action_idx)

#         payoff_matrix = np.zeros((A, A))
#         belief = self.belief_prob[s_a_idx, s_b_idx]

#         for my_action in range(A):
#             for opp_action in range(A):
#                 payoff_matrix[my_action, opp_action] = (
#                     belief[opp_action] * self.Q_vals[s_a_idx, s_b_idx, my_action, opp_action] if self.is_A else 
#                     belief[opp_action] * self.Q_vals[s_a_idx, s_b_idx, opp_action, my_action])
        

#         # LP: max min payoff
#         c = np.zeros(A + 1)
#         c[-1] = -1  # max v == min -v

#         A_eq = np.zeros((1, A + 1))
#         A_eq[0, :A] = 1
#         b_eq = [1]

#         A_ub = []
#         b_ub = []

#         for b in range(A):
#             if belief[b] > 0:
#                 constraint = np.zeros(A + 1)
#                 constraint[:A] = - payoff_matrix[:, b] 
#                 constraint[-1] = 1
#                 A_ub.append(constraint)
#                 b_ub.append(0)

#         bounds = [(0, 1) for _ in range(A)] + [(None, None)]

#         if A_ub:
#             result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
#                              A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')
#             if result.success:
#                 policy = np.clip(result.x[:A], 0, 1)
#                 policy /= policy.sum()
#                 self.pi[s_a_idx, s_b_idx] = policy
#             else:
#                 raise ValueError("LP failed to solve.")
#         else:
#             self.pi[s_a_idx, s_b_idx] = np.ones(A) / A

#     # def update_utilities(self, old_state, action):
#     #     action_a, action_b = action
#     #     s_a_idx_old = self.state_idx[str(old_state[0])]
#     #     s_b_idx_old = self.state_idx[str(old_state[1])]
#     #     s_a_idx_new = self.state_idx[str(self.environment.a_state)]
#     #     s_b_idx_new = self.state_idx[str(self.environment.b_state)]

#     #     a_idx = self.action_idx[action_a]
#     #     b_idx = self.action_idx[action_b]

#     #     reward = self.environment.reward if self.is_A else -self.environment.reward

#     #     # Q-value update
#     #     old_q = self.Q_vals[s_a_idx_old, s_b_idx_old, a_idx, b_idx]
#     #     next_v = self.V_vals[s_a_idx_new, s_b_idx_new]
#     #     self.Q_vals[s_a_idx_old, s_b_idx_old, a_idx, b_idx] = (
#     #         (1-self.lr) * old_q + (self.lr) * (reward + self.gamma * next_v)
#     #     )

#     #     # V-value update: expected value under current policy and belief
#     #     pi = self.pi[s_a_idx_old, s_b_idx_old]  # shape (A,)
#     #     belief = self.belief_prob[s_a_idx_old, s_b_idx_old]  # shape (A,)
#     #     Q = self.Q_vals[s_a_idx_old, s_b_idx_old]  # shape (A, A)

#     #     expected_values = np.zeros(len(self.action_idx))
#     #     for my_action in self.action_idx.values():
#     #         for opp_action in self.action_idx.values():
#     #             expected_values[my_action] += pi[my_action] * belief[opp_action] * (Q[my_action, opp_action] if self.is_A else Q[opp_action,my_action]) #non come nelle slides, ma qui in piu c'è il for su a e le pi[a] ->perchè sto usando behavioural strategy.

#     #     self.V_vals[s_a_idx_old, s_b_idx_old] = np.max(expected_values)

#     #     self.lr *= self.decay
    
#     def update_Q(self, old_state, action):
#         action_a, action_b = action
#         s_a_idx_old = self.state_idx[str(old_state[0])]
#         s_b_idx_old = self.state_idx[str(old_state[1])]
#         s_a_idx_new = self.state_idx[str(self.environment.a_state)]
#         s_b_idx_new = self.state_idx[str(self.environment.b_state)]

#         a_idx = self.action_idx[action_a]
#         b_idx = self.action_idx[action_b]

#         reward = self.environment.reward if self.is_A else -self.environment.reward

#         # Q-value update
#         old_q = self.Q_vals[s_a_idx_old, s_b_idx_old, a_idx, b_idx]
#         v_val = self.V_vals[s_a_idx_new, s_b_idx_new]
#         self.Q_vals[s_a_idx_old, s_b_idx_old, a_idx, b_idx] = (
#             (1-self.lr) * old_q + (self.lr) * (reward + self.gamma * v_val)
#         )
        
#     def update_V(self,old_state):#,action):
#         #action_a, action_b = action
#         s_a_idx_old = self.state_idx[str(old_state[0])]
#         s_b_idx_old = self.state_idx[str(old_state[1])]
#         # s_a_idx_new = self.state_idx[str(self.environment.a_state)]

#         pi = self.pi[s_a_idx_old, s_b_idx_old]  # shape (A,)
#         belief = self.belief_prob[s_a_idx_old, s_b_idx_old]  # shape (A,)
#         Q = self.Q_vals[s_a_idx_old, s_b_idx_old]  # shape (A, A)

#         expected_values = np.zeros(len(self.action_idx))
#         for my_action in self.action_idx.values():
#             for opp_action in self.action_idx.values():
#                 expected_values[my_action] += pi[my_action] * belief[opp_action] * (Q[my_action, opp_action] if self.is_A else Q[opp_action,my_action]) #non come nelle slides, ma qui in piu c'è il for su a e le pi[a] ->perchè sto usando behavioural strategy.

#         self.V_vals[s_a_idx_old, s_b_idx_old] = np.max(expected_values)
        
        
        
#     def update_agent(self, old_state, action):
#         if not self.random_policy:
#             self.update_belief(old_state,action[1] if self.is_A else action[0])
#             self.update_Q(old_state,action)
#             self.update_policy(old_state)
#             self.update_V(old_state)
#             self.lr *= self.decay
import numpy as np
from scipy.optimize import linprog

class Agent:
    def __init__(self, environment, is_A=True, random_policy=False, **kwargs):
        self.environment = environment
        self.is_A = is_A
        self.random_policy = random_policy

        self.lr = kwargs.get('lr', 1.0)
        self.exp_rate = kwargs.get('exp_rate', 0.2)
        self.gamma = kwargs.get('gamma', 0.9)
        self.decay = kwargs.get('decay', 0.9999954)

        # State indexing now includes possession
        self.state_idx = {
            f'({i}, {j})': i * self.environment.n_cols + j for i in range(self.environment.n_rows) for j in range(self.environment.n_cols)
        }
        
        # Possession indexing: 0 = A has possession, 1 = B has possession
        self.possession_idx = {True: 0, False: 1}  # True = A has possession
        self.idx_possession = {0: True, 1: False}

        self.action_idx = {'N': 0, 'S': 1, 'W': 2, 'E': 3, 'STAND': 4}
        self.idx_action = {v: k for k, v in self.action_idx.items()}
        self.idx_state = {k: v for v, k in self.state_idx.items()}

        S = len(self.state_idx)  # Number of position states
        P = len(self.possession_idx)  # Number of possession states (2)
        A = len(self.action_idx)  # Number of actions

        # Updated tensors to include possession dimension
        # Shape: (S_A, S_B, P, A, A) - positions A, positions B, possession, action A, action B
        self.Q_vals = np.zeros((S, S, P, A, A), dtype=np.float32)
        
        # Shape: (S_A, S_B, P) - positions A, positions B, possession
        self.V_vals = np.ones((S, S, P), dtype=np.float32)
        
        # Shape: (S_A, S_B, P, A) - positions A, positions B, possession, my actions
        self.pi = np.ones((S, S, P, A), dtype=np.float32)
        self.pi /= self.pi.sum(axis=3, keepdims=True)

        # Belief about opponent's actions
        # Shape: (S_A, S_B, P, A) - positions A, positions B, possession, opponent actions
        self.belief_count = np.zeros((S, S, P, A), dtype=np.int16)
        self.belief_prob = np.ones((S, S, P, A), dtype=np.float32)
        self.belief_prob /= self.belief_prob.sum(axis=3, keepdims=True)

    def _get_state_indices(self, state_a, state_b, possession_a):
        """Helper method to get state indices including possession"""
        s_a_idx = self.state_idx[str(state_a)]
        s_b_idx = self.state_idx[str(state_b)]
        p_idx = self.possession_idx[possession_a]
        return s_a_idx, s_b_idx, p_idx

    def choose_action(self):
        state_a = self.environment.a_state
        state_b = self.environment.b_state
        possession_a = self.environment.possession_a
        
        s_a_idx, s_b_idx, p_idx = self._get_state_indices(state_a, state_b, possession_a)

        if self.random_policy or np.random.rand() < self.exp_rate:
            return np.random.choice(list(self.action_idx.keys()))
        else:
            probs = self.pi[s_a_idx, s_b_idx, p_idx]
            return np.random.choice(list(self.action_idx.keys()), p=probs)

    def update_belief(self, old_state, old_possession, action_opponent): 
        s_a_idx_old, s_b_idx_old, p_idx_old = self._get_state_indices(
            old_state[0], old_state[1], old_possession
        )
        b_idx = self.action_idx[action_opponent]  

        self.belief_count[s_a_idx_old, s_b_idx_old, p_idx_old, b_idx] += 1
        counts = self.belief_count[s_a_idx_old, s_b_idx_old, p_idx_old]
        total = counts.sum()
        if total > 0:
            self.belief_prob[s_a_idx_old, s_b_idx_old, p_idx_old] = counts / total

    def update_policy(self, old_state, old_possession):
        s_a_idx, s_b_idx, p_idx = self._get_state_indices(
            old_state[0], old_state[1], old_possession
        )
        A = len(self.action_idx)

        payoff_matrix = np.zeros((A, A))
        belief = self.belief_prob[s_a_idx, s_b_idx, p_idx]

        for my_action in range(A):
            for opp_action in range(A):
                payoff_matrix[my_action, opp_action] = (
                    belief[opp_action] * self.Q_vals[s_a_idx, s_b_idx, p_idx, my_action, opp_action] if self.is_A else 
                    belief[opp_action] * self.Q_vals[s_a_idx, s_b_idx, p_idx, opp_action, my_action]
                )

        # LP: max min payoff
        c = np.zeros(A + 1)
        c[-1] = -1  # max v == min -v

        A_eq = np.zeros((1, A + 1))
        A_eq[0, :A] = 1
        b_eq = [1]

        A_ub = []
        b_ub = []

        for b in range(A):
            if belief[b] > 0:
                constraint = np.zeros(A + 1)
                constraint[:A] = - payoff_matrix[:, b] 
                constraint[-1] = 1
                A_ub.append(constraint)
                b_ub.append(0)

        bounds = [(0, 1) for _ in range(A)] + [(None, None)]

        if A_ub:
            result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                             A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')
            if result.success:
                policy = np.clip(result.x[:A], 0, 1)
                policy /= policy.sum()
                self.pi[s_a_idx, s_b_idx, p_idx] = policy
            else:
                raise ValueError("LP failed to solve.")
        else:
            self.pi[s_a_idx, s_b_idx, p_idx] = np.ones(A) / A
    
    def update_Q(self, old_state, old_possession, action):
        action_a, action_b = action
        
        # Old state indices
        s_a_idx_old, s_b_idx_old, p_idx_old = self._get_state_indices(
            old_state[0], old_state[1], old_possession
        )
        
        # New state indices
        s_a_idx_new, s_b_idx_new, p_idx_new = self._get_state_indices(
            self.environment.a_state, self.environment.b_state, self.environment.possession_a
        )

        a_idx = self.action_idx[action_a]
        b_idx = self.action_idx[action_b]

        reward = self.environment.reward if self.is_A else -self.environment.reward

        # Q-value update
        old_q = self.Q_vals[s_a_idx_old, s_b_idx_old, p_idx_old, a_idx, b_idx]
        v_val = self.V_vals[s_a_idx_new, s_b_idx_new, p_idx_new]
        self.Q_vals[s_a_idx_old, s_b_idx_old, p_idx_old, a_idx, b_idx] = (
            (1-self.lr) * old_q + (self.lr) * (reward + self.gamma * v_val)
        )
        
    def update_V(self, old_state, old_possession):
        s_a_idx_old, s_b_idx_old, p_idx_old = self._get_state_indices(
            old_state[0], old_state[1], old_possession
        )

        pi = self.pi[s_a_idx_old, s_b_idx_old, p_idx_old]  # shape (A,)
        belief = self.belief_prob[s_a_idx_old, s_b_idx_old, p_idx_old]  # shape (A,)
        Q = self.Q_vals[s_a_idx_old, s_b_idx_old, p_idx_old]  # shape (A, A)

        expected_values = np.zeros(len(self.action_idx))
        for my_action in self.action_idx.values():
            for opp_action in self.action_idx.values():
                expected_values[my_action] += (
                    pi[my_action] * belief[opp_action] * 
                    (Q[my_action, opp_action] if self.is_A else Q[opp_action, my_action])
                )

        self.V_vals[s_a_idx_old, s_b_idx_old, p_idx_old] = np.max(expected_values)
        
    def update_agent(self, old_state, old_possession, action):
        if not self.random_policy:
            self.update_belief(old_state, old_possession, action[1] if self.is_A else action[0])
            self.update_Q(old_state, old_possession, action)
            self.update_policy(old_state, old_possession)
            self.update_V(old_state, old_possession)
            self.lr *= self.decay