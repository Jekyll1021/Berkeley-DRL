import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		# TODO: random select all actions before predict.

		# states = []
		# actions = []
		# next_states = []
		# states.append(np.repeat([state], self.num_simulated_paths, axis=0))
		# for i in range(self.horizon):
		# 	path_action = []
		# 	for k in range(self.num_simulated_paths):
		# 		path_action.append(self.env.action_space.sample())
		# 	actions.append(path_action)

		# for i in range(self.horizon):
		# 	batch_states = states[i]
		# 	batch_actions = actions[i]
		# 	batch_next_states = self.dyn_model.predict(batch_states, batch_actions)
		# 	states.append(batch_states)
		# 	next_states.append(batch_states)
		# traj_score = trajectory_cost_fn(self.cost_fn, np.array(states), np.array(actions), np.array(next_states))
		# ind_ac = np.argmin(traj_score)
		# return actions[0][ind_ac]

		paths = {}
		for k in range(self.num_simulated_paths):
			paths[k] = {}
			paths[k]["states"] = [state]
			paths[k]["actions"] = []
			paths[k]["next_states"] = []

		for i in range(self.horizon):
			states = np.concatenate([[paths[k]["states"][i]] for k in range(len(paths))])
			actions = []
			for k in range(self.num_simulated_paths):
				act = self.env.action_space.sample()
				paths[k]["actions"].append(act)
				actions.append(act)
			actions = np.array(actions)

			next_states = self.dyn_model.predict(states, actions)
			for k in range(self.num_simulated_paths):
				paths[k]["states"].append(next_states[k])
				paths[k]["next_states"].append(next_states[k])

		traj_score = []
		for k in range(self.num_simulated_paths):
			traj_score.append(trajectory_cost_fn(self.cost_fn, paths[k]["states"], paths[k]["actions"], paths[k]["next_states"]))
		traj_score = np.array(traj_score)
		ind_ac = np.argmin(traj_score)
		return paths[ind_ac]["actions"][0]


		""" Note: be careful to batch your simulations through the model for speed """

