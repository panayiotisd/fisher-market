import gym
import numpy as np
import random as rnd

from gym import spaces
from gym.utils import seeding

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from scipy.optimize import minimize, LinearConstraint


class fishery_market_environment(MultiAgentEnv):

	# ********** CONSTRUCTOR **********



	def __init__(self,
				 n_harvesters,
				 n_buyers,
				 n_resources,
				 skill_level,
				 growth_rate,
				 S_eq,
				 max_steps,
				 threshold=1e-4,
				 harvester_wastefulness_cost=0.0,
				 policymaker_harvesters_welfare_weight=1.0,
				 policymaker_buyers_welfare_weight=1.0,
				 policymaker_fairness_weight=1.0,
				 policymaker_wastefulness_weight=0.0,
				 policymaker_sustainability_weight=1.0,
				 fairness_metric='jain',
				 valuations_noise_method=None,
				 valuations_noise=0.01,
				 n_valuations_bins=100,
				 random_seed=42,
				 compute_market_eq=False,
				 debug=False,
				 ):
		"""
		Parameters
		----------
		n_harvesters : int
			Number of harvesters in the environment (sellers in the market)
		n_buyers : int
			Number of buyers in the marker
		n_resources : int
			Number of resource types available for harvesting in the environment
		skill_level : numpy array (n_harvesters, n_resources) of floats
			Skill level of each harvester for harvesting each resource
		growth_rate : numpy array (n_resources) of floats
			Intrinsic growth rate of each resource
		S_eq : numpy array (n_resources) of floats
			Equilibrium population stock of each resource
		max_steps : int
			Maximum number of steps in the environment
		threshold : float
			Minimum stock threshold; below this point the resource is considered depleted
		harvester_wastefulness_cost : float
			Cost of wasting harvested resources. If a harvester does no sell the entire harvest of a resource, he will incur this cost.
		policymaker_harvesters_welfare_weight : float
			Weight of the harvesters' social welfare objective in policymaker's reward function.
		policymaker_buyers_welfare_weight : float
			Weight of the buyers' social welfare objective in policymaker's reward function.
		policymaker_fairness_weight : float
			Weight of the fairness objective in policymaker's reward function.
		policymaker_wastefulness_weight : float
			Weight of the wastefulness objective in policymaker's reward function.
		policymaker_sustainability_weight : float
			Weight of the sustainability objective in policymaker's reward function.
		fairness_metric : 'jain', 'gini', 'atkinson'
			Use the Jain index, the Gini coefficient, or the Atkinson index to calculate fairness
		valuations_noise_method: None, 'uniform', 'bins'
			The policymaker receives as input the original valuations vector, or a noisy one with uniform noise, or one split into bins
		valuations_noise: float
			If valuations_noise_method == 'uniform', then this defines the amount of noise
		n_valuations_bins: int
			If valuations_noise_method == 'bins', then this defines the number of bins
		random_seed : int
			Random number generator seed
		compute_market_eq : boolean
			Computes the market equilibrium prices instead of using the policymaker
		debug : boolean
			Print debugging messages
		"""

		# Set parameter values from arguments
		self.n_policymakers = 1
		self.n_harvesters = n_harvesters
		self.n_buyers = n_buyers
		self.n_resources = n_resources
		self.skill_level = skill_level
		self.growth_rate = growth_rate
		self.S_eq = S_eq
		self.max_steps = max_steps
		self.threshold = threshold
		self.harvester_wastefulness_cost = harvester_wastefulness_cost
		self.policymaker_harvesters_welfare_weight = policymaker_harvesters_welfare_weight
		self.policymaker_buyers_welfare_weight = policymaker_buyers_welfare_weight
		self.policymaker_fairness_weight = policymaker_fairness_weight
		self.policymaker_wastefulness_weight = policymaker_wastefulness_weight
		self.policymaker_sustainability_weight = policymaker_sustainability_weight
		self.fairness_metric = fairness_metric
		self.valuations_noise_method = valuations_noise_method
		self.valuations_noise = valuations_noise
		self.n_valuations_bins = n_valuations_bins
		self.random_seed = random_seed
		self.compute_market_eq = compute_market_eq
		self.debug = debug


		########################################################
		# TODO: VERIFY THE SANITY OF THE VALUES!
		########################################################
		self.min_price = 0.0
		self.max_price = np.inf
		self.min_effort = 0.0
		self.max_effort = 1.0
		self.min_bugdet = 0.0
		self.max_bugdet = 1.0
		self.min_valuation = 0.0
		self.max_valuation = 1.0
		self.min_reward = -np.inf
		self.max_reward = np.inf
		self.min_stock = 0.0
		self.max_stock = np.inf


		# check validity of parameters
		if (not (isinstance(self.n_harvesters, int) and self.n_harvesters >= 2)):
			raise ValueError("There must be at least two harvesters: " + str(self.n_harvesters))
		if (not (isinstance(self.n_buyers, int) and self.n_buyers >= 2)):
			raise ValueError("There must be at least two buyers: " + str(self.n_buyers))
		if (not (isinstance(self.n_resources, int) and self.n_resources >= 2)):
			raise ValueError("There must be at least two resources: " + str(self.n_resources))
		if not (self.skill_level.shape == (self.n_harvesters, self.n_resources) and
				self.skill_level.dtype == (np.float64 or np.float32 or np.int) and
				np.count_nonzero(self.skill_level < 0) == 0 and
				np.count_nonzero(self.skill_level > 1) == 0):
			raise ValueError("Skill levels have to be a (" + str(self.n_harvesters) + "," + str(self.n_resources) + ") floats np array in [0, 1]: " + np.array2string(self.skill_level))
		if not (self.growth_rate.shape == (self.n_resources,) and
				self.growth_rate.dtype == (np.float64 or np.float32 or np.int) and
				np.count_nonzero(self.growth_rate <= 0) == 0):
			raise ValueError("Growth rates have to be a (" + str(self.n_resources) + ",) floats/int np array in (0, inf]: " + np.array2string(self.growth_rate))
		if not (self.S_eq.shape == (self.n_resources,) and
				self.S_eq.dtype == (np.float64 or np.float32 or np.int) and
				np.count_nonzero(self.S_eq <= 0) == 0):
			raise ValueError("Seq have to be a (" + str(self.n_resources) + ",) floats/int np array in (0, inf]: " + np.array2string(self.S_eq))
		if (not (isinstance(self.max_steps, int) and self.max_steps >= 1)):
			raise ValueError("max_steps must be >= 1: " + str(self.max_steps))
		if (not (0 < self.threshold < 1)):
			raise ValueError("threshold must be in (0, 1): " + str(self.threshold))
		if (not (0 <= self.harvester_wastefulness_cost)):
			raise ValueError("harvester_wastefulness_cost must be non-negative: " + str(self.harvester_wastefulness_cost))
		if (not (0 <= self.policymaker_harvesters_welfare_weight)):
			raise ValueError("policymaker_harvesters_welfare_weight must be non-negative: " + str(self.policymaker_harvesters_welfare_weight))
		if (not (0 <= self.policymaker_buyers_welfare_weight)):
			raise ValueError("policymaker_buyers_welfare_weight must be non-negative: " + str(self.policymaker_buyers_welfare_weight))
		if (not (0 <= self.policymaker_fairness_weight)):
			raise ValueError("policymaker_fairness_weight must be non-negative: " + str(self.policymaker_fairness_weight))
		if (not (0 <= self.policymaker_wastefulness_weight)):
			raise ValueError("policymaker_wastefulness_weight must be non-negative: " + str(self.policymaker_wastefulness_weight))
		if (not (0 <= self.policymaker_sustainability_weight)):
			raise ValueError("policymaker_sustainability_weight must be non-negative: " + str(self.policymaker_sustainability_weight))
		if (not (self.fairness_metric == 'jain' or self.fairness_metric == 'gini' or self.fairness_metric == 'atkinson')):
			raise ValueError("Select amongst the 'jain', 'gini', or 'atkinson' fairness metrics: " + str(self.fairness_metric))
		if self.valuations_noise_method:
			if (not (self.valuations_noise_method == 'uniform' or self.valuations_noise_method == 'bins')):
				raise ValueError("Select amongst the 'uniform' or 'bins' valuations noise method: " + str(self.valuations_noise_method))
		if (not (isinstance(self.valuations_noise, float) and valuations_noise < 1.0 and valuations_noise > 0)):
			raise ValueError("The valuations noise has to be a float in (0, 1): " + str(self.valuations_noise))
		if (not (isinstance(self.n_valuations_bins, int) and self.n_valuations_bins >= 2)):
			raise ValueError("The number of bins has to be an integer and there must be at least two bins: " + str(self.n_valuations_bins))
		if (not (isinstance(self.random_seed, int) or isinstance(self.random_seed, float))):
			raise ValueError("The seed must be a number: " + str(self.random_seed))
		if (not isinstance(self.compute_market_eq, bool)):
			raise ValueError("The compute_market_eq flag must be boolean: " + str(self.compute_market_eq))
		if (not isinstance(self.debug, bool)):
			raise ValueError("The debug flag must be boolean: " + str(self.debug))


		# Set up various parameters
		self.done = True    # Flag for the end of episode
		
		self.seed(random_seed)
		rnd.seed(random_seed)

		self.l_harvesters = ["harvester{}".format(i) for i in range(self.n_harvesters)] # Create list of harvesters
		self.l_policymakers = ["policymaker{}".format(i) for i in range(self.n_policymakers)] # Create list of policymakers


		# Create a dictionary with the history of the date from the previous timestep used to calculate rewards
		self.history = {}
		self.history['harvests'] = np.zeros([self.n_harvesters, self.n_resources], dtype=np.float64)
		self.history['prices'] = np.zeros(self.n_resources, dtype=np.float64)
		self.history['efforts'] = {}
		for harvester in self.l_harvesters:
			self.history['efforts'][harvester] = np.zeros(self.n_resources, dtype=np.float64)
		self.history['harvester_rewards'] = np.zeros(self.n_harvesters, dtype=np.float64)
		self.history['harvester_revenue'] = np.zeros(self.n_harvesters, dtype=np.float64)
		self.history['wasted_percentage'] = np.zeros([self.n_harvesters, self.n_resources], dtype=np.float64)
		self.history['buyers_utility'] = np.zeros(self.n_buyers, dtype=np.float64)


		# Define action and observation spaces
		self.observation_space_harvester = spaces.Box(low=np.array([self.min_price] * self.n_resources + [self.min_effort] * self.n_resources + [self.min_reward], dtype=np.float64), high=np.array([self.max_price] * self.n_resources + [self.max_effort] * self.n_resources + [self.max_reward], dtype=np.float64), shape=(self.n_resources + self.n_resources + 1,), dtype=np.float64)	# Each harvester observes the prices, efforts, and reward from the previous time-step
		self.action_space_harvester = spaces.Box(low=np.array([self.min_effort] * self.n_resources, dtype=np.float64), high=np.array([self.max_effort] * self.n_resources, dtype=np.float64), shape=(self.n_resources,), dtype=np.float64)	# Each harvester exerts some effort to harvest each resource

		self.observation_space_policymaker = spaces.Box(low=np.array([self.min_effort] * self.n_resources + [self.min_stock] * self.n_resources + [self.min_bugdet] * self.n_buyers + [self.min_valuation] * (self.n_buyers * self.n_resources), dtype=np.float64), high=np.array([self.max_effort * self.n_harvesters] * self.n_resources + [self.max_stock] * self.n_resources + [self.max_bugdet] * self.n_buyers + [self.max_valuation] * (self.n_buyers * self.n_resources), dtype=np.float64), shape=(self.n_resources + self.n_resources + self.n_buyers + self.n_buyers * self.n_resources,), dtype=np.float64)	# The policymaker observes the current harvest and the budgets and valuations of each buyer
		self.action_space_policymaker = spaces.Box(low=np.array([self.min_price] * self.n_resources, dtype=np.float64), high=np.array([self.max_price] * self.n_resources, dtype=np.float64), shape=(self.n_resources,), dtype=np.float64)	# The policymaker sets the price for each resource

		# Print parameters
		# if self.debug:
		print()
		print("-----fishery_market_environment------")
		print('n_harvesters = ' + str(self.n_harvesters))
		print('n_buyers = ' + str(self.n_buyers))
		print('n_resources = ' + str(self.n_resources))
		print('skill_level = ' + str(self.skill_level))
		print('growth_rate = ' + str(self.growth_rate))
		print('S_eq = ' + str(self.S_eq))
		print('max_steps = ' + str(self.max_steps))
		print('threshold = ' + str(self.threshold))
		print('harvester_wastefulness_cost = ' + str(self.harvester_wastefulness_cost))
		print('policymaker_harvesters_welfare_weight = ' + str(self.policymaker_harvesters_welfare_weight))
		print('policymaker_buyers_welfare_weight = ' + str(self.policymaker_buyers_welfare_weight))
		print('policymaker_fairness_weight = ' + str(self.policymaker_fairness_weight))
		print('policymaker_wastefulness_weight = ' + str(self.policymaker_wastefulness_weight))
		print('policymaker_sustainability_weight = ' + str(self.policymaker_sustainability_weight))
		print('fairness_metric = ' + str(self.fairness_metric))
		print('valuations_noise_method = ' + str(self.valuations_noise_method))
		print('valuations_noise = ' + str(self.valuations_noise))
		print('n_valuations_bins = ' + str(self.n_valuations_bins))
		print('random_seed = ' + str(self.random_seed))
		print('compute_market_eq = ' + str(self.compute_market_eq))
		print()
		print('observation_space_harvester = ' + str(self.observation_space_harvester))
		print('action_space_harvester = ' + str(self.action_space_harvester))
		print('observation_space_policymaker = ' + str(self.observation_space_policymaker))
		print('action_space_policymaker = ' + str(self.action_space_policymaker))
		print("-------------------------------------")
		print()





	# ********** FISHERY DYNAMICS **********



	def harvest_fn(self, efforts, skill_level, cur_stock, S_eq):
		assert efforts.shape == (self.n_harvesters, self.n_resources)

		efforts = np.multiply(efforts, skill_level)
		harvests = [self.h_fn(efforts[:, resource], cur_stock[resource], S_eq[resource]) for resource in range(self.n_resources)]

		return np.transpose(np.array(harvests))


	def h_fn(self, efforts, cur_stock, S_eq):
		E = np.sum(efforts)
		if E == 0:
			return np.array([0] * self.n_harvesters)
		q = cur_stock / (2 * S_eq) if cur_stock <= (2 * S_eq) else 1
		H = min(cur_stock, q * E)

		return H * efforts / E


	def spawner_recruit_fn(self, cumulative_harvest, cur_stock, growth_rate, S_eq):
		assert cumulative_harvest.shape == (self.n_resources,)

		new_stock = [self.f_fn(cumulative_harvest[resource], cur_stock[resource], growth_rate[resource], S_eq[resource]) for resource in range(self.n_resources)]

		return np.array(new_stock)


	@staticmethod
	def f_fn(harvested, cur_stock, growth_rate, S_eq):
		remaining = max(0, cur_stock - harvested)
		growth = np.exp(growth_rate * (1 - remaining / S_eq))
		return remaining * growth



	

	# ********** MARKET DYNAMICS **********



	def bugdet_fn(self):
		np.random.seed(rnd.randint(0, 9999)) # FIXME: Replace with below
		if self.valuations_noise_method:
			np.random.seed(self.cur_step)
		res = np.random.rand(self.n_buyers)
		res[res <= 0] = 1e-6	# Ensure there are no zeros
		return res


	def valuations_fn(self):
		np.random.seed(rnd.randint(0, 9999)) # FIXME: Replace with below
		if self.valuations_noise_method:
			np.random.seed(self.cur_step)
		res = np.random.rand(self.n_buyers, self.n_resources)
		res[res <= 0] = 1e-6	# Ensure there are no zeros
		return res

		
	def add_noise_fn(self, valuations):
		if (self.valuations_noise_method == 'uniform'):
			valuations = self.uniform_noise_fn(valuations)
			assert np.count_nonzero(self.valuations <= 0) == 0 and np.count_nonzero(self.valuations > 1) == 0, valuations
			return valuations
		elif (self.valuations_noise_method == 'bins'):
			valuations = self.split_into_bins_fn(valuations)
			assert np.count_nonzero(self.valuations <= 0) == 0 and np.count_nonzero(self.valuations > 1) == 0, valuations
			return valuations
		else:
			raise ValueError('Invalid valuations noise method: ' + self.valuations_noise_method)


	def uniform_noise_fn(self, valuations):
		noise = np.random.uniform(-self.valuations_noise, self.valuations_noise, [self.n_buyers, self.n_resources])
		valuations = valuations + noise
		valuations[valuations <= 0] = 1e-6
		valuations[valuations >= 1] = 1 - 1e-6
		return valuations


	def split_into_bins_fn(self, valuations):
		bins = np.linspace(0, 1, self.n_valuations_bins + 1, endpoint=True, dtype=np.float64)
		bins = np.around(bins, 3)

		valuations_bins = np.digitize(valuations, bins, right=True)

		for buyer in range(self.n_buyers):
			for resource in range(self.n_resources):
				valuations[buyer, resource] = (bins[valuations_bins[buyer, resource] - 1] + bins[valuations_bins[buyer, resource]]) / 2.0

		return valuations


	def optimal_allocation_given_prices_fn(self, prices, cumulative_harvest, budgets, valuations):
		# See https://realpython.com/python-scipy-cluster-optimize/
		assert prices.shape == (self.n_resources,), prices
		assert cumulative_harvest.shape == (self.n_resources,), cumulative_harvest
		assert budgets.shape == (self.n_buyers,), budgets
		assert valuations.shape == (self.n_buyers, self.n_resources), valuations

		if np.array_equal(cumulative_harvest, np.zeros(self.n_resources)):
			return np.zeros(self.n_resources), np.zeros(self.n_buyers)

		if np.allclose(cumulative_harvest, np.zeros(self.n_resources), rtol=0, atol=9e-3):
			return np.zeros(self.n_resources), np.zeros(self.n_buyers)

		cumulative_harvest[cumulative_harvest <=0 ] = 1e-6	# Ensure there are no zeros

		# Volume constraints:
		# The total amount of a resource that all buyers cumulative buy can not exceed the total harvest of that resource.
		# For example, for n_buyers = 2 and n_resources = 3 the constraint matrix will have 3 constraints:
		# constraint_matrix = [[1. 1. 0. 0. 0. 0.]
		# 					   [0. 0. 1. 1. 0. 0.]
		# 					   [0. 0. 0. 0. 1. 1.]]
		constraint_matrix = []
		for resource in range(self.n_resources):
			tmp = np.zeros([self.n_resources, self.n_buyers])
			tmp[resource] = np.ones(self.n_buyers)
			constraint_matrix.append(tmp.flatten())

		constraint_matrix = np.array(constraint_matrix)
		
		volume_constraints = [LinearConstraint(constraint_matrix[resource], lb=0, ub=cumulative_harvest[resource]) for resource in range(self.n_resources)]

		# Budget constraints
		# The total amount of resources any buyer buys can not exceed his budget.
		# For example, for n_buyers = 2, n_resources = 3, and prices = [3 4 5] the constraint matrix will have 2 constraints:
		# constraint_matrix = [[3. 0. 4. 0. 5. 0.]
 		#					   [0. 3. 0. 4. 0. 5.]]
		constraint_matrix = []
		for buyer in range(self.n_buyers):
			tmp = np.zeros([self.n_resources, self.n_buyers])
			tmp[:, buyer] = prices
			constraint_matrix.append(tmp.flatten())


		constraint_matrix = np.array(constraint_matrix)
		
		budget_constraints = [LinearConstraint(constraint_matrix[buyer], lb=0, ub=budgets[buyer]) for buyer in range(self.n_buyers)]

		# Feasibility constraints:
		# You can not buy more than the harvested amount.
		# You can not buy a negative amount of a resource.
		constraint_matrix = np.eye(self.n_resources * self.n_buyers, dtype=int)

		feasibility_constraints = [LinearConstraint(constraint_matrix[i], lb=0, ub=max(cumulative_harvest)) for i in range(self.n_resources * self.n_buyers)]


		# All constraints
		constraints = volume_constraints + budget_constraints + feasibility_constraints

		# Do not use bounds, only linear constraints. Gives better accuracy!
		# Feasibility bounds
		# You can not buy more than the harvested amount.
		# You can not buy a negative amount of a resource.
		# bounds = [(0, max(cumulative_harvest)) for _ in range(self.n_resources * self.n_buyers)]

		res = minimize(
			self.optimal_allocation_given_prices_objective_fn,
			x0=np.ones(self.n_resources * self.n_buyers),
			args=(valuations,),
			constraints=constraints,
			# bounds=bounds,
			method='SLSQP'
		)

		assert res.success, res

		allocation = np.copy(res.x)
		allocation = np.around(allocation, 4)
		allocation = allocation.reshape(self.n_resources, self.n_buyers)

		sold_resources = np.sum(allocation, 1)
		assert sold_resources.shape == (self.n_resources,)


		buyers_utility = res.x * np.transpose(valuations).flatten()
		buyers_utility = np.around(buyers_utility, 4)
		buyers_utility = buyers_utility.reshape(self.n_resources, self.n_buyers)
		buyers_utility = np.transpose(buyers_utility)
		buyers_utility = np.sum(buyers_utility, 1)

		assert buyers_utility.shape == (self.n_buyers,)

		return sold_resources, buyers_utility


	@staticmethod
	def optimal_allocation_given_prices_objective_fn(x, valuations):
		return -x.dot(np.transpose(valuations).flatten())



	def fisher_market_equilibrium_calculator_fn(self, cumulative_harvest, budgets, valuations, tolerance=1e-3):
		# See https://www.ics.uci.edu/~vazirani/market.pdf
		assert cumulative_harvest.shape == (self.n_resources,), cumulative_harvest
		assert budgets.shape == (self.n_buyers,), budgets
		assert valuations.shape == (self.n_buyers, self.n_resources), valuations

		if np.array_equal(cumulative_harvest, np.zeros(self.n_resources)):
			return np.zeros(self.n_resources), np.zeros(self.n_buyers), np.zeros(self.n_resources)

		if np.allclose(cumulative_harvest, np.zeros(self.n_resources), rtol=0, atol=1e-3):
			return cumulative_harvest, np.zeros(self.n_buyers), np.zeros(self.n_resources)

		cumulative_harvest[cumulative_harvest <=0 ] = 1e-6	# Ensure there are no zeros

		scaling_factor = 1.0
		if np.sum(cumulative_harvest) <=1:
			scaling_factor = 100.0	# The market solution is the same up to scaling, and scaling makes it easier for the solver
			cumulative_harvest = scaling_factor * cumulative_harvest

		# Volume constraints:
		# The total amount of a resource that all buyers cumulative buy can not exceed the total harvest of that resource.
		# For example, for n_buyers = 2 and n_resources = 3 the constraint matrix will have 3 constraints:
		# constraint_matrix = [[1. 1. 0. 0. 0. 0.]
		# 					   [0. 0. 1. 1. 0. 0.]
		# 					   [0. 0. 0. 0. 1. 1.]]
		constraint_matrix = []
		for resource in range(self.n_resources):
			tmp = np.zeros([self.n_resources, self.n_buyers])
			tmp[resource] = np.ones(self.n_buyers)
			constraint_matrix.append(tmp.flatten())

		constraint_matrix = np.array(constraint_matrix)

		volume_constraints = [LinearConstraint(constraint_matrix[resource], lb=1e-9, ub=cumulative_harvest[resource]) for resource in range(self.n_resources)]

		# print(volume_constraints[0].A)
		# print(volume_constraints[0].ub)

		# Feasibility constraints:
		# You can not buy more than the harvested amount.
		# You can not buy a negative amount of a resource.
		constraint_matrix = np.eye(self.n_resources * self.n_buyers, dtype=int)

		feasibility_constraints = [LinearConstraint(constraint_matrix[i], lb=1e-9, ub=max(cumulative_harvest)) for i in range(self.n_resources * self.n_buyers)]

		# All constraints
		constraints = volume_constraints + feasibility_constraints

		# Do not use bounds, only linear constraints. Gives better accuracy!
		# Feasibility bounds
		# You can not buy more than the harvested amount.
		# You can not buy a negative amount of a resource.
		# bounds = [(0, max(cumulative_harvest)) for _ in range(self.n_resources * self.n_buyers)]

		res = minimize(
			self.eisenberg_gale,
			# x0=np.ones(self.n_resources * self.n_buyers, dtype=np.float64),
			x0=np.random.dirichlet(np.ones(self.n_resources * self.n_buyers), size=1)[0],	# Start from a random solutions, sampled from a dirichlet distribution. The values sum up to one.
			args=(budgets, valuations * cumulative_harvest, self.n_resources, self.n_buyers,), # For the LP to be solved correctly, the valuations need to be adjusted to the total harvest. See https://www.ics.uci.edu/~vazirani/market.pdf
			constraints=constraints,
			# bounds=bounds,
			method='trust-constr',
			# tol=1e-20
			options={'gtol':tolerance,
			'xtol':tolerance,
			'barrier_tol':tolerance,  # Tolerances provide a trade-off between solution accuracy and speed. 1e-3 results in +-1e-2 close to the optimal, in 1.37+-0.66 sec of computation time
			'maxiter':2000},
		)

		assert res.success, res

		cumulative_harvest = cumulative_harvest / scaling_factor

		allocation = np.copy(res.x) / scaling_factor
		allocation = np.around(allocation, 6)
		allocation = allocation.reshape(self.n_resources, self.n_buyers)

		sold_resources = np.sum(allocation, 1)
		assert sold_resources.shape == (self.n_resources,)
		assert np.allclose(sold_resources, cumulative_harvest, rtol=0, atol=0.05), 'cumulative_harvest = ' + str(cumulative_harvest) + '\n sold_resources = ' + str(sold_resources) # FIXME: Enable

		sold_resources = np.copy(cumulative_harvest) # At market equilibrium, we sell the entire harvest



		buyers_utility = (res.x / scaling_factor) * np.transpose(valuations).flatten()
		buyers_utility = np.around(buyers_utility, 6)
		buyers_utility = buyers_utility.reshape(self.n_resources, self.n_buyers)
		buyers_utility = np.transpose(buyers_utility)
		buyers_utility = np.sum(buyers_utility, 1)

		assert buyers_utility.shape == (self.n_buyers,)



		# The equilibrium prices correspond to the Lagrange multipliers for the three volume constraints 
		prices = []
		for r in range(self.n_resources):
			prices.append(res.v[r][0])
		prices = np.array(prices)
		prices = scaling_factor * prices
		prices[abs(prices) < 1e-6] = 0	# Remove negative prices due to noise
		assert np.count_nonzero(prices < 0) == 0, prices

		allocation = np.transpose(allocation)
		used_budget = np.sum(allocation * prices, 1)
		assert np.allclose(used_budget, budgets, rtol=0, atol=0.05), 'budgets = ' + str(budgets) + '\n used_budget = ' + str(used_budget)  # At market equilibrium, the budget is exhausted # FIXME: Enable


		# leftover_budget = np.sum(cumulative_harvest * prices, 0) - np.sum(budgets)
		# print("allocation = \n" + str(allocation) + "\n")
		# print("prices = \n" + str(prices) + "\n")
		# print("leftover_budget = \n" + str(leftover_budget) + "\n")


		return sold_resources, buyers_utility, prices



	@staticmethod
	def eisenberg_gale(x, budgets, valuations, n_resources, n_buyers):
		# x_ij is the amount of good j allocated to buyer i 
		# budgets: e_i is the amount of money of buyer i
		# valuations: u_ij denotes the utility of i on obtaining a unit of good j
		# See https://www.ics.uci.edu/~vazirani/market.pdf
		assert x.shape == (n_resources * n_buyers,)
		x[x <= 0] = 1e-6 # To avoid negative values to the logarithm
		x = np.transpose(x.reshape(n_resources, n_buyers))  # x is now (n_buyers, n_resources)
		return -np.sum(budgets * np.log(np.sum(x * valuations, 1)), 0)





	# ********** REWARD FUNCTIONS **********



	def policymaker_reward_fn(self, harvester_rewards, harvester_revenue, wasted_percentage, cur_stock, S_eq, buyers_utility):
		assert harvester_rewards.shape == (self.n_harvesters,)
		assert harvester_revenue.shape == (self.n_harvesters,)
		assert wasted_percentage.shape == (self.n_harvesters, self.n_resources)
		assert cur_stock.shape == (self.n_resources,)
		assert S_eq.shape == (self.n_resources,)
		assert buyers_utility.shape == (self.n_buyers,)

		harvesters_social_welfare = self.harvesters_social_welfare_fn(harvester_revenue)	# Maximize harvesters' social welfare
		fairness = self.fairness_fn(harvester_rewards)	# Maximize harvesters' fairness
		wastefulness_cost = self.wastefulness_fn(wasted_percentage)	# Minimize harvesters' wasted harvest
		sustainability_cost, stock_difference = self.sustainability_fn(cur_stock, S_eq)	# Ensure all resources are harvested sustainably
		buyers_social_welfare = self.buyers_social_welfare_fn(buyers_utility)	# Maximize buyers' social welfare

		reward= self.policymaker_harvesters_welfare_weight * (1.0 / self.n_harvesters) * harvesters_social_welfare + \
				self.policymaker_fairness_weight * fairness + \
				self.policymaker_wastefulness_weight * wastefulness_cost + \
				self.policymaker_sustainability_weight * sustainability_cost + \
				self.policymaker_buyers_welfare_weight * (1.0 / self.n_buyers) * buyers_social_welfare

		if self.debug:
			print()
			print("-------policymaker_reward_fn---------")
			print('reward = \n' + str(reward))
			print('harvesters_social_welfare = \n' + str(harvesters_social_welfare))
			print('fairness = \n' + str(fairness))
			print('wastefulness_cost = \n' + str(wastefulness_cost))
			print('sustainability_cost = \n' + str(sustainability_cost))
			print('stock_difference = \n' + str(stock_difference))
			print('buyers_social_welfare = \n' + str(buyers_social_welfare))
			print("-------------------------------------")
			print()

		return reward, fairness, stock_difference


	@staticmethod
	def harvesters_social_welfare_fn(harvester_revenue):
		return np.sum(harvester_revenue, 0)


	@staticmethod
	def buyers_social_welfare_fn(buyers_utility):
		return np.sum(buyers_utility, 0)


	def fairness_fn(self, harvester_rewards):
		if (self.fairness_metric == 'jain'):
			fairness = self.jain_index_fn(harvester_rewards)
			assert 0 <= fairness <= 1
			return fairness
		elif (self.fairness_metric == 'gini'):
			fairness =  1 - self.gini_coefficient_fn(harvester_rewards) # We maximize fairness. According to the Gini coefficient, an allocation is fair iff the coefficient is 0.
			if fairness < 0:		# The Gini coefficient is not bounded
				fairness = 0
			assert 0 <= fairness <= 1
			return fairness
		elif (self.fairness_metric == 'atkinson'):
			fairness = 1 - self.atkinson_index_fn(harvester_rewards) # We maximize fairness. According to the Atkinson index, an allocation is fair iff the index is 0.
			assert 0 <= fairness <= 1
			return fairness
		else:
			raise ValueError('Invalid fairness metric: ' + self.fairness_metric)


	@staticmethod
	def jain_index_fn(rewards):
		if np.count_nonzero(rewards) == 0:
			return 1	# Fair allocation; everybody got reward 0
		rewards = rewards.astype(np.float64)

		return np.sum(rewards) ** 2 / ( np.sum(rewards ** 2) * rewards.shape[0] )


	@staticmethod
	def gini_coefficient_fn(rewards):
		if np.count_nonzero(rewards) == 0:
			return 0	# Fair allocation; everybody got reward 0
		rewards = rewards.astype(np.float64)

		G = np.sum(np.abs(rewards - np.array([np.roll(rewards,i) for i in range(rewards.shape[0])])))
		G /= sum(rewards) * 2 * rewards.shape[0]
		return G


	# This is the Atkinson index for epsilon = 1
	@staticmethod
	def atkinson_index_fn(rewards):
		# if np.count_nonzero(rewards) == 0:
		if np.allclose(rewards, np.zeros(rewards.shape[0]), rtol=0, atol=1e-6):
			return 0 # Fair allocation; everybody got reward 0
		rewards = rewards.astype(np.float64)
		product = np.prod(rewards)

		if not np.any(rewards == 0):
			assert product > 0, rewards # Ensure there are no precision errors

		return 1 - ( pow(product, 1.0 / rewards.shape[0]) / np.mean(rewards) )


	def wastefulness_fn(self, wasted_percentage):
		return -sum(sum(wasted_percentage * self.harvester_wastefulness_cost))


	@staticmethod
	def sustainability_fn(cur_stock, S_eq):
		stock_difference = cur_stock - S_eq
		# stock_difference[stock_difference > 0] = 0
		# return sum(stock_difference), stock_difference
		sustainability_cost = min(stock_difference)
		if sustainability_cost > 0:
			sustainability_cost = 0
		return sustainability_cost, stock_difference


	@staticmethod
	def social_mobility_fn():
		pass



	def harvesters_reward_fn(self, prices, harvests, budgets, valuations):
		assert prices.shape == (self.n_resources,)
		assert harvests.shape == (self.n_harvesters, self.n_resources)
		assert budgets.shape == (self.n_buyers,)
		assert valuations.shape == (self.n_buyers, self.n_resources)

		cumulative_harvest = np.sum(harvests, axis=0)
		assert cumulative_harvest.shape == (self.n_resources,)

		tolerance = 1e-3
		counter = 4
		while True:
			try:
				if not self.compute_market_eq:
					sold_resources, buyers_utility = self.optimal_allocation_given_prices_fn(prices, cumulative_harvest, budgets, valuations)
				else:
					sold_resources, buyers_utility, prices = self.fisher_market_equilibrium_calculator_fn(cumulative_harvest, budgets, valuations, tolerance)
			except ValueError as err:
				# Sometimes we might get a negative value in the log
				counter = counter - 1
				if counter == 0:
					print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ValueError! " + str(counter) + " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
					print("WARNING: Failed to compute the market equilibrium!")
					print('cumulative_harvest = np.' + repr(cumulative_harvest))
					print('budgets = np.' + repr(budgets))
					print('valuations = np.' + repr(valuations))
					if not self.compute_market_eq:
						print('prices = np.' + repr(prices))
					print()
					print(err)
					
					# if np.sum(cumulative_harvest) < 0.05:
					sold_resources, buyers_utility, prices = cumulative_harvest, np.zeros(self.n_buyers), np.zeros(self.n_resources)
					break
					# raise err
				tolerance = 1e-21
				continue
			except AssertionError as err:
				# Sometimes -- due to the randomness of the optimization method and the tolerances -- we might not have a good enough solution
				counter = counter - 1
				if counter == 0:
					print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> AssertionError! " + str(counter) + " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
					print("WARNING: Failed to compute the market equilibrium!")
					print('cumulative_harvest = np.' + repr(cumulative_harvest))
					print('budgets = np.' + repr(budgets))
					print('valuations = np.' + repr(valuations))
					if not self.compute_market_eq:
						print('prices = np.' + repr(prices))
					print()
					print(err)
					
					# if np.sum(cumulative_harvest) < 0.05:
					sold_resources, buyers_utility, prices = cumulative_harvest, np.zeros(self.n_buyers), np.zeros(self.n_resources)
					break
					# raise err
				tolerance = 1e-21
				continue
			else:
				break


		assert np.count_nonzero(sold_resources < 0) == 0, ('sold_resources = np.' + repr(sold_resources), 'buyers_utility = np.' + repr(buyers_utility), 'prices = np.' + repr(prices), 'cumulative_harvest = np.' + repr(cumulative_harvest), 'budgets = np.' + repr(budgets), 'valuations = np.' + repr(valuations))
		assert np.count_nonzero(prices < 0) == 0, ('sold_resources = np.' + repr(sold_resources), 'buyers_utility = np.' + repr(buyers_utility), 'prices = np.' + repr(prices), 'cumulative_harvest = np.' + repr(cumulative_harvest), 'budgets = np.' + repr(budgets), 'valuations = np.' + repr(valuations))

		# Calculate the revenue from sales
		harvest_relative = np.copy(harvests).astype(np.float64)
		for resource in range(self.n_resources):
			if cumulative_harvest[resource] == 0:
				continue
			harvest_relative[:, resource] = np.true_divide(harvest_relative[:, resource], cumulative_harvest[resource])

		# Revenue
		revenue = sold_resources * prices

		# We distribute the revenue proportionally
		harvester_rewards = np.zeros(self.n_harvesters)
		for harvester in range(self.n_harvesters):
			reward = harvest_relative[harvester] * revenue
			harvester_rewards[harvester] = np.sum(reward)

		harvester_revenue = np.copy(harvester_rewards)

		# Next, we calculate the cost of wasting resources
		harvest_relative = np.copy(harvests).astype(np.float64)
		for resource in range(self.n_resources):
			if cumulative_harvest[resource] == 0:
				continue
			harvest_relative[:, resource] = np.true_divide(harvest_relative[:, resource], cumulative_harvest[resource])

		wasted_percentage = np.zeros([self.n_harvesters, self.n_resources], dtype=np.float64)
		for harvester in range(self.n_harvesters):
			wasted = harvests[harvester] - (harvest_relative[harvester] * sold_resources)	# harvested - sold
			wasted = np.true_divide(wasted, harvests[harvester], where=(harvests[harvester]!=0))	# Percentage of wasted
			wasted_percentage[harvester] = np.copy(wasted)
			wasted *= self.harvester_wastefulness_cost

			harvester_rewards[harvester] -= np.sum(wasted)

		return harvester_rewards, harvester_revenue, wasted_percentage, buyers_utility, prices





	# ********** MAIN ENVIRONMENT FUNCTIONS **********



	# Environment reset function, called at the beginning of each episode
	def reset(self):

		self.cur_stock = np.copy(self.S_eq)	# The stock starts from the equilibrium population value

		self.cur_step = 0	# note that counter will increase for each harvest and policymaker step
		self.done = False


		# Reset history
		self.history = {}
		self.history['harvests'] = np.zeros([self.n_harvesters, self.n_resources], dtype=np.float64)
		self.history['prices'] = np.zeros(self.n_resources, dtype=np.float64)
		self.history['efforts'] = {}
		for harvester in self.l_harvesters:
			self.history['efforts'][harvester] = np.zeros(self.n_resources, dtype=np.float64)
		self.history['harvester_rewards'] = np.zeros(self.n_harvesters, dtype=np.float64)
		self.history['harvester_revenue'] = np.zeros(self.n_harvesters, dtype=np.float64)
		self.history['wasted_percentage'] = np.zeros([self.n_harvesters, self.n_resources], dtype=np.float64)
		self.history['buyers_utility'] = np.zeros(self.n_buyers, dtype=np.float64)

		
		init_state = self.observation_space_harvester.sample()	# Sample a random state
		init_state[self.n_resources:] = 0	# Set the previous effort and reward to zero

		# At the first timestep the harvesters will harvest. Hence we provide (reset) observations only for the harvesters and not the policymaker.
		harvester_states = {}
		for i, h in enumerate(self.l_harvesters):
			harvester_states[h] = init_state


		if self.debug:
			print("Environment reset completed!")

		return harvester_states





	# Step function: Perform one step in the environment
	def step(self, actions):
		"""
		Parameters
		----------
		actions : Dict
			Agent actions
		"""

		# Do not process step if episode finished
		if self.done:
			return None


		# Check if the current step is for harvesting or calculating prices
		acting_agents = list(actions)
		if( 'harvester' in acting_agents[0] ):
			states, rewards, dones, infos = self.harvester_step(actions)	# Harvest
		elif( 'policymaker' in acting_agents[0]):
			states, rewards, dones, infos = self.policymaker_step(actions)	# Calculate prices
		else:
			raise ValueError('Invalid agent name in the action dictionary: ' + str(acting_agents))


		# Return states, rewards, done flags and info dictionaries
		return states, rewards, dones, infos





	# Harvesting step
	def harvester_step(self, actions):
		"""
		Parameters
		----------
		actions : Dict
			Agent actions (harvesting effort for each resource)
		"""

		if self.debug:
			print("Harvesting Step " + str(self.cur_step))

		assert not(self.l_policymakers[0] in actions)

		efforts = np.zeros([self.n_harvesters, self.n_resources], dtype=np.float64)
		for i, h in enumerate(self.l_harvesters):
			if h in actions:
				efforts[i] = np.copy(actions[h])
			else:
				raise ValueError('Harvester ' + h + ' was not found in the dictionary of actions: ' + str(actions))


		# Sanity check
		if np.count_nonzero(efforts > 1) != 0 or np.count_nonzero(efforts < 0) != 0:
			raise ValueError('Harvesting efforts are not in [0, 1]: ' + str(efforts))


		# Calculate policymaker 's reward for the previous step.
		policymaker_reward, harvester_fairness, stock_difference = self.policymaker_reward_fn(self.history['harvester_rewards'], self.history['harvester_revenue'], self.history['wasted_percentage'], self.cur_stock, self.S_eq, self.history['buyers_utility'])

		# Calculate total harvest per resource
		harvests = self.harvest_fn(efforts, self.skill_level, self.cur_stock, self.S_eq)	# n_harvesters * n_resources
		cumulative_harvest = np.sum(harvests, axis=0)	# n_resources

		self.history['harvests'] = np.copy(harvests)	# Store harvests to calculate rewards in the policymaker step
		self.history['efforts'] = actions.copy()		# Store efforts to use as part of the state

		# Update stock levels
		self.cur_stock = self.spawner_recruit_fn(cumulative_harvest, self.cur_stock, self.growth_rate, self.S_eq)


		assert harvests.shape == (self.n_harvesters, self.n_resources), harvests
		assert cumulative_harvest.shape == (self.n_resources,), cumulative_harvest
		assert self.cur_stock.shape == (self.n_resources,), self.cur_stock
		assert stock_difference.shape == (self.n_resources,), stock_difference


		# Update buyers info
		self.budgets = self.bugdet_fn()
		self.valuations = self.valuations_fn()	


		# Set returns
		states = {}
		rewards = {}
		dones = {}
		infos = {}

		# Set policymaker observations
		policymaker = self.l_policymakers[0]
		states[policymaker] = np.concatenate([cumulative_harvest, self.cur_stock, self.budgets, self.valuations.flatten()])
		if self.valuations_noise_method:
			states[policymaker] = np.concatenate([cumulative_harvest, self.cur_stock, self.budgets, self.add_noise_fn(self.valuations).flatten()])
		rewards[policymaker] = policymaker_reward	# Policymaker 's reward for the previous step. Implemented like this because of RLLib's technical constraint which requires observation and reward dicts to have the same keys
		infos[policymaker] = {"harvester_fairness" : harvester_fairness, "stock_difference" : stock_difference, "harvests" : harvests, "efforts" : efforts}
		dones['__all__'] = False


		if self.debug:
			print()
			print("-----------harvester_step------------")
			print('efforts = \n' + str(efforts))
			print('harvests = \n' + str(harvests))
			print('cumulative_harvest = \n' + str(cumulative_harvest))
			print('cur_stock = \n' + str(self.cur_stock))
			print()
			print()
			print('harvester_fairness = \n' + str(harvester_fairness))
			print('stock_difference = \n' + str(stock_difference))
			print('policymaker_reward = \n' + str(policymaker_reward))
			print('policymaker_state = \n' + str(states[policymaker]))
			print("-------------------------------------")
			print()


		# Return states, rewards, done flags and info dictionaries
		return states, rewards, dones, infos




	# Implements the act step: The chosen transformation acts in the environment
	def policymaker_step(self, actions):
		"""
		Parameters
		----------
		actions : Dict
			Agent actions (Prices for each resource)
		"""

		if self.debug:
			print("Policymaking Step " + str(self.cur_step))

		assert len(actions) == 1	# There is only one policymaker

		# Increment timestep
		self.cur_step = self.cur_step + 1

		prices = np.copy(actions[self.l_policymakers[0]])

		# Sanity check
		assert prices.shape == (self.n_resources,)
		if np.count_nonzero(prices < 0) != 0:
			raise ValueError('We can not have negative prices: ' + str(prices))


		self.history['prices'] = np.copy(prices)	# Store prices to calculate rewards in the harvest step

		harvester_rewards, harvester_revenue, wasted_percentage, buyers_utility, eq_prices = self.harvesters_reward_fn(prices, self.history['harvests'], self.budgets, self.valuations)

		if self.compute_market_eq:
			self.history['prices'] = np.copy(eq_prices)
			prices = np.copy(eq_prices)

		assert harvester_rewards.shape == (self.n_harvesters,), harvester_rewards
		assert harvester_revenue.shape == (self.n_harvesters,), harvester_revenue
		assert wasted_percentage.shape == (self.n_harvesters, self.n_resources), wasted_percentage
		assert buyers_utility.shape == (self.n_buyers,), buyers_utility

		self.history['harvester_rewards'] = np.copy(harvester_rewards)
		self.history['harvester_revenue'] = np.copy(harvester_revenue)
		self.history['wasted_percentage'] = np.copy(wasted_percentage)
		self.history['buyers_utility'] = np.copy(buyers_utility)


		# Set returns:
		states = {}
		rewards = {}
		dones = {}
		infos = {}


		# set states, dones and infos for harvester agents only
		for i, harvester in enumerate(self.l_harvesters):
			states[harvester] = np.concatenate([prices, self.history['efforts'][harvester], [harvester_rewards[i]]])
			rewards[harvester] = harvester_rewards[i]
			infos[harvester] = { "harvester_rewards": harvester_rewards, "harvester_revenue": harvester_revenue, "wasted_percentage": wasted_percentage, "buyers_utility": buyers_utility, "prices": prices}
		

		dones['__all__'] = False
		# Set finished flag if maximum number of steps are reached or any of the resources is depleted
		if(self.cur_step >= self.max_steps or np.any(self.cur_stock <= self.threshold)):
			dones['__all__'] = True 	# flag to end the episode

			# Give the final observation and reward to the policymaker agent  (required by RLLib)
			policymaker = self.l_policymakers[0]
			states[policymaker] = self.observation_space_policymaker.sample()
			rewards[policymaker] = 1
			infos[policymaker] = {"harvester_fairness" : -np.inf, "stock_difference" : -np.inf}



		if self.debug:
			print()
			print("-----------harvester_step------------")
			if self.compute_market_eq:
				print('eq_prices = \n' + str(prices))
			else:
				print('prices = \n' + str(prices))
			print('harvester_rewards = \n' + str(harvester_rewards))
			print('harvester_revenue = \n' + str(harvester_revenue))
			print('wasted_percentage = \n' + str(wasted_percentage))
			print('buyers_utility = \n' + str(buyers_utility))
			print()
			print()
			print('harvesters_rewards = \n' + str(rewards))
			print('harvesters_states = \n' + str(states))
			print("-------------------------------------")
			print()


		# Return primitive states, rewards, done flags and info dictionaries
		return states, rewards, dones, infos





	# ********** HELPER FUNCTIONS **********



	# Create random seed
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def render(self):
		print("Current step: %d" % (self.cur_step))


	def close(self):
		return
