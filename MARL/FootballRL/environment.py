import numpy as np
from tabulate import tabulate
from itertools import product

# Constants
N_ROWS = 4
N_COLS = 5

INITIAL_A_STATE = (1,3)
INITIAL_B_STATE = (2,1)

POTENTIAL_A_GOAL = [(1,0), (2,0)]
POTENTIAL_B_GOAL = [(1,4), (2,4)]


positions = list(product(range(N_ROWS), range(N_COLS)))
AVAILABLE_STATES = [(a, b) for a in positions for b in positions]
AVAILABLE_ACTIONS = ["N", "S", "W", "E", "STAND"]  # N,S,W,E,STAND


class Environment:
	def __init__(
		self,
		n_rows = N_ROWS, n_cols = N_COLS, 
		initial_a = INITIAL_A_STATE, initial_b = INITIAL_B_STATE,
		potential_a_goal = POTENTIAL_A_GOAL, potential_b_goal = POTENTIAL_B_GOAL,
		available_states = AVAILABLE_STATES, available_actions = AVAILABLE_ACTIONS):
		
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.initial_a = initial_a
		self.initial_b = initial_b
		self.potential_a_goal = potential_a_goal
		self.potential_b_goal = potential_b_goal
		self.available_states = available_states
		self.available_actions = available_actions

		self.a_state = initial_a
		self.b_state = initial_b

		self.possession_a = np.random.choice([True, False], p=[0.5, 0.5]) 


		self.won_games_A = 0
		self.won_games_B = 0
		self.reward = 0
		self.terminated_games = 0
		self.goal = False
		
		self.match_length = 0
		self.match_length_list = []



	def __str__(self):
		mat = np.zeros((self.n_rows, self.n_cols), dtype=object)
		mat[self.a_state] = "A"
		mat[self.b_state] = "B"

		if self.possession_a:
			mat[self.a_state] = "(" + mat[self.a_state] + ")"
		else:
			mat[self.b_state] = "(" + mat[self.b_state] + ")"

		return tabulate(mat, tablefmt="grid",
					numalign="center", stralign="center")
	
	# choose_action, update_policy, update_belief, update_utilities
	def reset_players(self):
		self.a_state = self.initial_a
		self.b_state = self.initial_b
		self.possession_a = np.random.choice([True, False], p=[0.5, 0.5])
		self.match_length = 0

	def reset(self):
		"""
		Reset the match to its initial state.
		"""
		self.reset_players()
		self.won_games_A = 0
		self.won_games_B = 0
		self.reward = 0
		self.terminated_games = 0
		self.terminated = False
		self.match_length = 0
		self.match_length_list = []
	
	def _move(self, state, action):
		"""
		Move the player in the given state according to the action.
		Action can be "N", "S", "E", "W" or "STAND".
		"""
		deltas = {
			"N": (-1, 0),
			"S": (1, 0),
			"E": (0, 1),
			"W": (0, -1),
			"STAND": (0, 0),
		}

		dr, dc = deltas.get(action, (0, 0))
		row, col = state
		new_row = min(max(0, row + dr), self.n_rows - 1)
		new_col = min(max(0, col + dc), self.n_cols - 1)

		return (new_row, new_col)

	
	def _next(self, action):
		"""
		Requires action to be a tuple of (a_action, b_action)
		where a_action and b_action are either "N", "S", "E", "W" or "STAND"
		"""
		a_action, b_action = action
		new_a = self._move(self.a_state, a_action)
		new_b = self._move(self.b_state, b_action)

		first_player = np.random.choice(["A", "B"], p=[0.5, 0.5])
		# print(f"First player to move: {first_player}")
		if first_player == "A":
			if new_a == self.b_state:
				# If A moves to B's position, B gets the ball
				self.possession_a = False
			else:
				self.a_state = new_a
			
			if new_b == self.a_state:
				# If B moves to A's position, A gets the ball
				self.possession_a = True
			else:
				self.b_state = new_b
			
		else:
			if new_b == self.a_state:
				# If B moves to A's position, A gets the ball
				self.possession_a = True
			else:
				self.b_state = new_b

			if new_a == self.b_state:
				# If A moves to B's position, B gets the ball
				self.possession_a = False
			else:
				self.a_state = new_a

	def update_if_goal(self, action):
		"""
		Checks if the game should terminate based on the action taken.
		If A or B reaches their potential goal, the game terminates.
		"""
		a_action, b_action = action

		if self.possession_a and (self.a_state in self.potential_a_goal) and (a_action == "W"):
			#self.terminated = True
			self.won_games_A += 1 
			self.terminated_games += 1
			self.reward =1#+= 1
			self.match_length_list.append(self.match_length)
			self.match_length = 0

			#self.goal = True
		elif (not self.possession_a) and (self.b_state in self.potential_b_goal) and (b_action == "E"):
			#self.terminated = True
			self.won_games_B += 1
			self.terminated_games += 1
			self.reward = -1#-=1#= -1
			self.match_length_list.append(self.match_length)
			self.match_length = 0
			#self.goal = True
		else:
			#self.goal = False
			# self.terminated = False
			#if self.reward != 0:
			# 	self.a_state = self.initial_a
			# 	self.b_state = self.initial_b
			# 	self.possession_a = np.random.choice([True, False], p=[0.5, 0.5])
			self.reward = 0
		# 	self.reward = 0
	

	def next_turn(self, action):
		"""
		Processes the next turn based on the action taken.
		Updates the state and checks for termination.
		"""
		self.match_length += 1
		self.update_if_goal(action)
		
		if self.reward != 0:
			self.a_state = self.initial_a
			self.b_state = self.initial_b
			self.possession_a = np.random.choice([True, False], p=[0.5, 0.5])
		else:
			self._next(action)
		return self.a_state, self.b_state, self.possession_a



	
