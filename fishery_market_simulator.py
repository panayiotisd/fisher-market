#!/usr/bin/env python
# coding: utf-8

# # Fishery Market Simulator
# 
# TODO: Add info
#
# Requirements:
# Python 3.8.5
# numpy v. 1.19.2
# tensorflow v. 2.4.1
# ray v. 0.8.7
# gym v. 0.18.0


###########################################
# @title Imports
###########################################


from fishery_market_environment import fishery_market_environment
from typing import Dict
import argparse
import datetime
import pickle
import pickletools
import gzip
import os
import math
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import interactive
# interactive(True)
import gym
from gym import spaces
import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.utils.types import AgentID, PolicyID
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import PPOTrainer
# from ray.rllib.agents.ddpg.ddpg import DDPGTrainer


###########################################
#@title Helper Function Definitions
###########################################


# Environment generator function
def generate_env_fn(env_context=None):
    return fishery_market_environment(n_harvesters = n_harvesters,
                                      n_buyers = n_buyers,
                                      n_resources = n_resources,
                                      skill_level = skill_level,
                                      growth_rate = growth_rate,
                                      S_eq = S_eq,
                                      max_steps = max_steps,
                                      threshold = threshold,
                                      harvester_wastefulness_cost = harvester_wastefulness_cost,
                                      policymaker_harvesters_welfare_weight = policymaker_harvesters_welfare_weight,
                                      policymaker_buyers_welfare_weight = policymaker_buyers_welfare_weight,
                                      policymaker_fairness_weight = policymaker_fairness_weight,
                                      policymaker_wastefulness_weight = policymaker_wastefulness_weight,
                                      policymaker_sustainability_weight = policymaker_sustainability_weight,
                                      policymaker_leftover_budget_weight = policymaker_leftover_budget_weight,
                                      policymaker_interventions_weight = policymaker_interventions_weight,
                                      fairness_metric = fairness_metric,
                                      valuations_noise_method = valuations_noise_method,
                                      valuations_noise = valuations_noise,
                                      n_valuations_bins = n_valuations_bins,
                                      effort_noise_method = effort_noise_method,
                                      effort_noise = effort_noise,
                                      random_seed = random_seed,
                                      compute_market_eq = compute_market_eq,
                                      compute_counterfactual_eq_prices = compute_counterfactual_eq_prices,
                                      debug = debug)


# policy mapping function
def policy_mapping_fn(agent_id):
    return "policy_" + agent_id


class MyCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        # Initialize global variables for the callback functions
        self.ep_number = 0
        self.ep_memory = []
        self.ag_memory = []
        self.log_dir = log_dir


    # Log saving function
    def save_log(self, ep_memory, ag_memory, worker_id):
        # print('---------------------------------------------save_log')

        log = {'n_harvesters' : n_harvesters,
            'n_buyers' : n_buyers,
            'n_resources' : n_resources,
            'skill_level' : skill_level,
            'growth_rate' : growth_rate,
            'S_eq' : S_eq,
            'Ms' : Ms,
            'max_steps' : max_steps,
            'threshold' : threshold,
            'harvester_wastefulness_cost' : harvester_wastefulness_cost,
            'policymaker_harvesters_welfare_weight' : policymaker_harvesters_welfare_weight,
            'policymaker_buyers_welfare_weight' : policymaker_buyers_welfare_weight,
            'policymaker_fairness_weight' : policymaker_fairness_weight,
            'policymaker_wastefulness_weight' : policymaker_wastefulness_weight,
            'policymaker_sustainability_weight' : policymaker_sustainability_weight,
            'policymaker_leftover_budget_weight' : policymaker_leftover_budget_weight,
            'policymaker_interventions_weight' : policymaker_interventions_weight,
            'fairness_metric' : fairness_metric,
            'valuations_noise_method' : valuations_noise_method,
            'valuations_noise' : valuations_noise,
            'n_valuations_bins' : n_valuations_bins,
            'effort_noise_method' : effort_noise_method,
            'effort_noise' : effort_noise,
            'random_seed' : random_seed,
            'compute_market_eq' : compute_market_eq,
            'compute_counterfactual_eq_prices' : compute_counterfactual_eq_prices,
            'num_workers' : num_workers,
            'n_episodes' : n_episodes,
            'train_algo' : train_algo,
            'lr' : lr,
            'gamma' : gamma,
            'hidden_layer_size' : hidden_layer_size,
            'episodes': ep_memory}

        filename = "logs_ep_memory_H_{0}_B_{1}_R_{2}_{3}_workerID_{4}_{5:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, worker_id, start_time)
        path = self.log_dir + '/' + filename
        try:
            print("Saving ep_memory logs to: {}".format(path))
            with gzip.open(path, "wb") as fp:
                # pickle.dump(log, fp)
                pickled = pickle.dumps(log)
                optimized_pickle = pickletools.optimize(pickled)
                fp.write(optimized_pickle)
        except:
            print("An exception occurred while saving ep_memory logs to {}!".format(path))


        log = {'n_harvesters' : n_harvesters,
            'n_buyers' : n_buyers,
            'n_resources' : n_resources,
            'skill_level' : skill_level,
            'growth_rate' : growth_rate,
            'S_eq' : S_eq,
            'Ms' : Ms,
            'max_steps' : max_steps,
            'threshold' : threshold,
            'harvester_wastefulness_cost' : harvester_wastefulness_cost,
            'policymaker_harvesters_welfare_weight' : policymaker_harvesters_welfare_weight,
            'policymaker_buyers_welfare_weight' : policymaker_buyers_welfare_weight,
            'policymaker_fairness_weight' : policymaker_fairness_weight,
            'policymaker_wastefulness_weight' : policymaker_wastefulness_weight,
            'policymaker_sustainability_weight' : policymaker_sustainability_weight,
            'policymaker_leftover_budget_weight' : policymaker_leftover_budget_weight,
            'policymaker_interventions_weight' : policymaker_interventions_weight,
            'fairness_metric' : fairness_metric,
            'valuations_noise_method' : valuations_noise_method,
            'valuations_noise' : valuations_noise,
            'n_valuations_bins' : n_valuations_bins,
            'effort_noise_method' : effort_noise_method,
            'effort_noise' : effort_noise,
            'random_seed' : random_seed,
            'compute_market_eq' : compute_market_eq,
            'compute_counterfactual_eq_prices' : compute_counterfactual_eq_prices,
            'num_workers' : num_workers,
            'n_episodes' : n_episodes,
            'train_algo' : train_algo,
            'lr' : lr,
            'gamma' : gamma,
            'hidden_layer_size' : hidden_layer_size,
            'episodes': ag_memory}

        filename = "logs_ag_memory_H_{0}_B_{1}_R_{2}_{3}_workerID_{4}_{5:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, worker_id, start_time)
        path = self.log_dir + '/' + filename
        try:
            print("Saving ag_memory logs to: {}".format(path))
            with gzip.open(path, "wb") as fp:
                # pickle.dump(log, fp)
                pickled = pickle.dumps(log)
                optimized_pickle = pickletools.optimize(pickled)
                fp.write(optimized_pickle)
        except:
            print("An exception occurred while saving ag_memory logs to {}!".format(path))



    # Callback function - episode start
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, **kwargs):

        # print('------------------------on_episode_start')
        # print('------------------------Worker ID = ' + str(worker.worker_index))
        # print('------------------------')

        # Initializations
        episode.user_data["observations"] = []
        episode.user_data["actions"] = []

        episode.user_data["harvester_fairness"] = []
        episode.user_data["stock_difference"] = []
        episode.user_data["harvests"] = []
        episode.user_data["efforts"] = []

        episode.user_data["harvester_rewards"] = []
        episode.user_data["harvester_revenue"] = []
        episode.user_data["wasted_percentage"] = []
        episode.user_data["buyers_utility"] = []
        episode.user_data["prices"] = []
        episode.user_data["leftover_budgets_percentage"] = []
        episode.user_data["price_difference_relative"] = []
        episode.user_data["counterfactual_eq_prices"] = []

        # Flag to track the harvest and policymaking steps to get the appropriate metrics from the info dictionaries
        self.harvesting_step = True # The environment starts with a harvest step
        self.first_step = True


    # Callback function - episode step
    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
      
        actions = {}
        observations = {}
        # Record agent actions and observations
        for agent_id in l_agents:
            actions[agent_id] = episode.last_action_for(agent_id)
            observations[agent_id] = episode.last_observation_for(agent_id)

        # Append to episode data
        episode.user_data["actions"].append(actions)
        episode.user_data["observations"].append(observations)
      

        # Update custom metrics
        # print('======================on_episode_step======================')
        # print('Worker ID = ' + str(worker.worker_index) + ' - harvesting_step = ' + str(self.harvesting_step))
        if self.harvesting_step and not self.first_step:
            info_dict = episode.last_info_for(l_harvesters[0]) # All harvesters have the same info
            assert (len(info_dict) != 0)

            # print('Supplementary environment information:')
            # print(info_dict)
            # print()

            episode.user_data["harvester_rewards"].append(info_dict["harvester_rewards"])
            episode.user_data["harvester_revenue"].append(info_dict["harvester_revenue"])
            episode.user_data["wasted_percentage"].append(info_dict["wasted_percentage"])
            episode.user_data["buyers_utility"].append(info_dict["buyers_utility"])
            episode.user_data["prices"].append(info_dict["prices"])
            episode.user_data["leftover_budgets_percentage"].append(info_dict["leftover_budgets_percentage"])
            episode.user_data["price_difference_relative"].append(info_dict["price_difference_relative"])
            episode.user_data["counterfactual_eq_prices"].append(info_dict["counterfactual_eq_prices"])

        elif not self.harvesting_step:
            assert self.first_step == False

            info_dict = episode.last_info_for(l_policymakers[0]) # We only have one policymaker
            assert (len(info_dict) != 0)

            # print('Supplementary environment information:')
            # print(info_dict)
            # print()

            if not (info_dict["harvester_fairness"] == -np.inf and info_dict["stock_difference"] == -np.inf): # Disregard the last two values
                episode.user_data["harvester_fairness"].append(info_dict["harvester_fairness"])
                episode.user_data["stock_difference"].append(info_dict["stock_difference"])
                episode.user_data["harvests"].append(info_dict["harvests"])
                episode.user_data["efforts"].append(info_dict["efforts"])
           

        self.harvesting_step = not self.harvesting_step
        self.first_step = False
      
      
      
    # Callback function - episode end
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        # print('---------------------------------------------on_episode_end')
        # print('------------------------Worker ID = ' + str(worker.worker_index))

        # Calculate fairness based on the cumulative rewards
        agent_rewards = episode.agent_rewards.copy()  # Summed rewards broken down by agent.
        if l_policymakers[0] in agent_rewards:
            del agent_rewards[l_policymakers[0]]
        agent_rewards = agent_rewards.values()
        agent_rewards = np.array(list(agent_rewards))
        harvester_fairness_at_end = self.fairness_fn(agent_rewards)

        # Get episode data lists saved during each step
        actions = episode.user_data["actions"]
        observations = episode.user_data["observations"]

        harvester_fairness = episode.user_data["harvester_fairness"]
        stock_difference = episode.user_data["stock_difference"]
        harvests = episode.user_data["harvests"]
        efforts = episode.user_data["efforts"]

        harvester_rewards = episode.user_data["harvester_rewards"]
        harvester_revenue = episode.user_data["harvester_revenue"]
        wasted_percentage = episode.user_data["wasted_percentage"]
        buyers_utility = episode.user_data["buyers_utility"]
        prices = episode.user_data["prices"]
        leftover_budgets_percentage = episode.user_data["leftover_budgets_percentage"]
        price_difference_relative = episode.user_data["price_difference_relative"]
        counterfactual_eq_prices = episode.user_data["counterfactual_eq_prices"]

        # TODO: Calculate episode metrics
        harvester_cumulative_reward = np.copy(agent_rewards)

        # print('--------------------------------------------- ep_number = ' + str(self.ep_number))
        # if self.ep_number < n_episodes:

        # Add data to agent and episode memory (we split into two to make the pickled files more manageable in size)
        self.ag_memory.append({'ep_number': self.ep_number, 'ep_len': episode.length,
                      'actions': actions, 'observations': observations,
                      'metrics':{'H_rew': harvester_cumulative_reward, 'H_fair' : harvester_fairness_at_end}})
        self.ep_memory.append({'ep_number': self.ep_number, 'ep_len': episode.length,
                      'agent_rewards': agent_rewards,
                      'harvester_fairness_at_end' : harvester_fairness_at_end,
                      'harvester_fairness': harvester_fairness,
                      'stock_difference': stock_difference,
                      'harvests': harvests,
                      'efforts': efforts,
                      'harvester_rewards': harvester_rewards,
                      'harvester_revenue': harvester_revenue,
                      'wasted_percentage': wasted_percentage,
                      'buyers_utility': buyers_utility,
                      'prices': prices,
                      'leftover_budgets_percentage': leftover_budgets_percentage,
                      'price_difference_relative': price_difference_relative,
                      'counterfactual_eq_prices': counterfactual_eq_prices})


        # Save periodically and at the end
        if (((self.ep_number + 1) * num_workers) % epdata_save_freq == 0) or (((self.ep_number + 1) * num_workers) >= n_episodes):
            print("on_episode_end: Saving log at ep_number = " + str(self.ep_number))
            self.save_log(self.ep_memory, self.ag_memory, worker.worker_index)

        self.ep_number += 1


    # Callback function - train result
    def on_train_result(self, trainer, result: dict, **kwargs):
        # print('---------------------------------------------on_train_result')
        # print(result.keys())
        # print()

        if ((self.ep_number + 1) * num_workers) < n_episodes:
            return

        # Find picklable objects
        f = open(self.log_dir + '/' + 'temp', 'wb')
        results_info = {}
        for key in result.keys():
            try:
                pickle.dump(result[key], f)
            except Exception as e:
                continue
            else:
                results_info[key] = result[key]
        f.close()
        os.remove(self.log_dir + '/' + 'temp') 

        # print(results_info)

        filename = "train_result_info_H_{0}_B_{1}_R_{2}_{3}_{4:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, start_time)
        path = self.log_dir + '/' + filename
        try:
            print("Saving training results info to: {}".format(path))
            with open(path, "wb") as fp:
                pickle.dump(results_info, fp)
        except Exception as e:
            print()
            print("An exception occurred while saving training results info to {}!".format(path))
            print(e)
            print()


    def fairness_fn(self, harvester_rewards):
        if (fairness_metric == 'jain'):
            fairness = self.jain_index_fn(harvester_rewards)
            assert 0 <= fairness <= 1
            return fairness
        elif (fairness_metric == 'gini'):
            fairness =  1 - self.gini_coefficient_fn(harvester_rewards) # We maximize fairness. According to the Gini coefficient, an allocation is fair iff the coefficient is 0.
            if fairness < 0:     # The Gini coefficient is not bounded
                fairness = 0
            assert 0 <= fairness <= 1
            return fairness
        elif (fairness_metric == 'atkinson'):
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
            assert product > 0 # Ensure there are no precision errors

        return 1 - ( pow(product, 1.0 / rewards.shape[0]) / np.mean(rewards) )




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

###########################################
#@title Definition of Parameters
###########################################


# *** Parse command arguments ***
parser = argparse.ArgumentParser()

# Environments arguments
parser.add_argument('--n_harvesters', default=8, type=int)
parser.add_argument('--n_buyers', default=8, type=int)
parser.add_argument('--n_resources', default=4, type=int)
parser.add_argument('--Ms', default=0.8, type=float)
parser.add_argument('--compute_market_eq', default=False, type=str2bool)
parser.add_argument('--compute_counterfactual_eq_prices', default=False, type=str2bool)

parser.add_argument('--harvester_wastefulness_cost', default=0.0, type=float)
parser.add_argument('--policymaker_harvesters_welfare_weight', default=1.0, type=float)
parser.add_argument('--policymaker_buyers_welfare_weight', default=1.0, type=float)
parser.add_argument('--policymaker_fairness_weight', default=1.0, type=float)
parser.add_argument('--policymaker_wastefulness_weight', default=0.0, type=float)
parser.add_argument('--policymaker_sustainability_weight', default=1.0, type=float)
parser.add_argument('--policymaker_leftover_budget_weight', default=0.0, type=float)
parser.add_argument('--policymaker_interventions_weight', default=0.0, type=float)
parser.add_argument('--fairness_metric', default='jain', type=str, choices={'jain', 'gini', 'atkinson'})

parser.add_argument('--valuations_noise_method', default=None, type=str, choices={'uniform', 'bins'})
parser.add_argument('--valuations_noise', default=0.01, type=float)
parser.add_argument('--n_valuations_bins', default=100, type=int)
parser.add_argument('--effort_noise_method', default=None, type=str, choices={'uniform'})
parser.add_argument('--effort_noise', default=0.01, type=float)

# Training arguments
parser.add_argument('--n_episodes', default=2400, type=int)
parser.add_argument('--max_steps', default=500, type=int)

# Workers (parallelism)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--num_cpus_per_worker', default=1, type=int)

# Miscellaneous
parser.add_argument('--random_seed', default=42, type=int)
parser.add_argument('--log_dir', default='./logs', type=str)
parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
parser.add_argument('--debug', default=False, type=str2bool)


args = parser.parse_args()

print("-----------------------------------------")
print("Parsed arguments:")
print("n_harvesters = " + str(args.n_harvesters))
print("n_buyers = " + str(args.n_buyers))
print("n_resources = " + str(args.n_resources))
print("Ms = " + str(args.Ms))
print("compute_market_eq = " + str(args.compute_market_eq))
print("compute_counterfactual_eq_prices = " + str(args.compute_counterfactual_eq_prices))

print("harvester_wastefulness_cost = " + str(args.harvester_wastefulness_cost))
print("policymaker_harvesters_welfare_weight = " + str(args.policymaker_harvesters_welfare_weight))
print("policymaker_buyers_welfare_weight = " + str(args.policymaker_buyers_welfare_weight))
print("policymaker_fairness_weight = " + str(args.policymaker_fairness_weight))
print("policymaker_wastefulness_weight = " + str(args.policymaker_wastefulness_weight))
print("policymaker_sustainability_weight = " + str(args.policymaker_sustainability_weight))
print("policymaker_leftover_budget_weight = " + str(args.policymaker_leftover_budget_weight))
print("policymaker_interventions_weight = " + str(args.policymaker_interventions_weight))
print("fairness_metric = " + str(args.fairness_metric))

print("valuations_noise_method = " + str(args.valuations_noise_method))
print("valuations_noise = " + str(args.valuations_noise))
print("n_valuations_bins = " + str(args.n_valuations_bins))
print("effort_noise_method = " + str(args.effort_noise_method))
print("effort_noise = " + str(args.effort_noise))

print("n_episodes = " + str(args.n_episodes))
print("max_steps = " + str(args.max_steps))

print("num_workers = " + str(args.num_workers))
print("num_cpus_per_worker = " + str(args.num_cpus_per_worker))

print("random_seed = " + str(args.random_seed))
print("log_dir = " + str(args.log_dir))
print("checkpoint_dir = " + str(args.checkpoint_dir))
print("debug = " + str(args.debug))
print("-----------------------------------------")


# ******** Environment Parameters ********
n_policymakers = 1
n_harvesters = args.n_harvesters # default: 8
n_buyers = args.n_buyers # default: 8
n_resources = args.n_resources # default: 4
# skill_level = np.array([[1, 0.5], [0.5, 1]])
skill_level = np.ones([n_harvesters, n_resources], dtype=np.float64) / 2.0
np.fill_diagonal(skill_level, 1, wrap=True)
growth_rate = np.ones(n_resources)
LSH = 0.79 # constant for the limit of sustainable harvesting (K)
Ms = args.Ms  #  default: 0.8
S_eq = np.array([Ms * LSH * n_harvesters] * n_resources)

threshold = 1e-4
harvester_wastefulness_cost = args.harvester_wastefulness_cost  # default: 0.0
policymaker_harvesters_welfare_weight = args.policymaker_harvesters_welfare_weight  # default: 1.0
policymaker_buyers_welfare_weight = args.policymaker_buyers_welfare_weight  # default: 1.0
policymaker_fairness_weight = args.policymaker_fairness_weight  # default: 1.0
policymaker_wastefulness_weight = args.policymaker_wastefulness_weight  # default: 0.0
policymaker_sustainability_weight = args.policymaker_sustainability_weight  # default: 1.0
policymaker_leftover_budget_weight = args.policymaker_leftover_budget_weight  # default: 0.0
policymaker_interventions_weight = args.policymaker_interventions_weight  # default: 0.0
fairness_metric = args.fairness_metric  # default: 'jain'

valuations_noise_method = args.valuations_noise_method  # default: None
valuations_noise = args.valuations_noise  # default: 0.01
n_valuations_bins = args.n_valuations_bins  # default: 100
effort_noise_method = args.effort_noise_method  # default: None
effort_noise = args.effort_noise  # default: 0.01

compute_market_eq = args.compute_market_eq  #  default: False
compute_counterfactual_eq_prices = args.compute_counterfactual_eq_prices  #  default: False
random_seed = args.random_seed # default: 42
debug = args.debug  # default: False


# ******** Training Parameters ********
n_episodes = args.n_episodes  # default: 2400
max_steps = args.max_steps  # default: 500

train_algo = "PPO"
lr = 1e-4
gamma = 0.99
lambda_trainer = 1.0

num_workers = args.num_workers # default: 1
num_cpus_per_worker = args.num_cpus_per_worker # default 1 # If run from notebook, this should be 0. This avoids running out of resources in the notebook environment when this cell is re-executed
num_gpus = 0
num_gpus_per_worker = 0

hidden_layer_size = 64
nw_model = {"fcnet_hiddens": [hidden_layer_size, hidden_layer_size],}  

log_dir = args.log_dir   # default: './logs', directory to save episode logs
checkpoint_dir = args.checkpoint_dir   # default: './checkpoints'
epdata_save_freq = math.ceil(args.n_episodes / 5.0)
checkpoint_interval = math.ceil(args.n_episodes / 5.0)


# ******** Conditions to cut training ********
early_stop_enable = False # set True to enable early stopping based on conditions defined below
cut_eps = 1000 # limit of no reward increase and episode length increase, in episodes
reward_increase_limit = 1.05  # ratio of reward increase limit
reward_decrease_limit = 0.95  # ratio of reward decrease limit 


# ******** Checkpoint restoration settings ********
# set by manually looking at the checkpoint folder name
# restore_checkpoint_folder =  './...'
# restore_checkpoint_num = 0  # checkpoint number to be restored, 0 if no checkpoint restoring



###########################################
#@title Simulation Setup
###########################################


# Initialize simulation
start_time = datetime.datetime.now()
print("--- Simulation started at {0:%Y%m%d_%H_%M_%S_%f} ---".format(start_time))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create checkpoint directory
folder_name = "checkpoint_H_{0}_B_{1}_R_{2}_{3}_{4:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, start_time)
folder_name = folder_name.replace(".", "p")

checkpoint_folder = checkpoint_dir + '/' + folder_name
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

# Initialize Ray
ray.shutdown()
ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

# Initialize environment
env_title = "env_H_{0}_B_{1}_R_{2}_{3}_{4:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, start_time)
print("Environment name: " + env_title)
register_env(env_title, generate_env_fn)  # Register environment
env = generate_env_fn() # Create environment

# Get action and observation spaces from environment implementation
obs_space_harvester = env.observation_space_harvester
act_space_harvester = env.action_space_harvester
obs_space_policymaker = env.observation_space_policymaker
act_space_policymaker = env.action_space_policymaker
    

# Initialize agents and policies
num_policies = n_harvesters + n_policymakers
l_harvesters = ["harvester{}".format(i) for i in range(n_harvesters)]
l_policymakers = ["policymaker{}".format(i) for i in range(n_policymakers)]
l_agents = l_harvesters + l_policymakers

# Map agent policies
policy_graphs_harvester = dict([("policy_harvester{}".format(i), (None, obs_space_harvester, act_space_harvester, {})) for i in range(n_harvesters)])
policy_graphs_policymaker = dict([("policy_policymaker{}".format(i), (None, obs_space_policymaker, act_space_policymaker, {})) for i in range(n_policymakers)])
# Concatenate harvester and policymaker policy graph dictionaries to get the total policy graph dictionary
policy_graphs = policy_graphs_harvester.copy()
policy_graphs.update(policy_graphs_policymaker)


# Set non-default settings in config dictionary
config_dict={
    "lr": lr,
    "gamma": gamma,
    "lambda": lambda_trainer,
    "model": nw_model,

    "seed": random_seed,

    "num_workers": num_workers,
    "num_cpus_per_worker": num_cpus_per_worker,
    "num_gpus": num_gpus,
    "num_gpus_per_worker": num_gpus_per_worker,

    "simple_optimizer": True,
    "clip_actions": True,

    "multiagent": {
        "policies": policy_graphs,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["policy_harvester{}".format(i) for i in range(n_harvesters)] + ["policy_policymaker{}".format(i) for i in range(n_policymakers)],
    },
    "callbacks": MyCallbacks,
    "log_level": "ERROR",
}


# Create trainer depending on input algorithm
if (train_algo == "PPO"):
    # Proximal Policy Optimization (PPO)
    print("Training algorithm: Proximal Policy Optimization (PPO)")
    trainer = PPOTrainer(
                env=env_title,
                config=config_dict)
elif (train_algo == "DDPG"):
    # Deep Deterministic Policy Gradient (DDPG) - NOT UP-TO-DATE
    print("Training algorithm: Deep Deterministic Policy Gradient (DDPG)")
    trainer = DDPGTrainer(
                env=env_title,
                config=config_dict)
else:
    raise ValueError("Unknown training algorithm: " + str(train_algo))

print("Successfully created " + str(train_algo) + " trainer.")




# TODO: Restore a checkpoint, only for the first simulation in the list
if False:
    trainer.restore(restore_checkpoint_folder + '/' + 'checkpoint_' + format(restore_checkpoint_num) + '/checkpoint-' + format(restore_checkpoint_num) )



eplen_max_started = False # flag for early stopping



###########################################
# @title Run Training
###########################################


last_checkpoint = 0
episodes_counter = 0
while episodes_counter < n_episodes:
    result = trainer.train()
    episodes_counter = result['episodes_total']
    # print('========================================== episodes_counter = ' + str(episodes_counter))

    print('\n-------------------------------------------------------------------------')
    print(pretty_print(result))
    print('-------------------------------------------------------------------------\n')


    if (episodes_counter - last_checkpoint) >= checkpoint_interval:
        print("Creating a checkpoint of the trainer at episodes_counter = " + str(episodes_counter))
        trainer.save(checkpoint_folder) # save checkpoint every (about) checkpoint_interval episodes
        last_checkpoint = episodes_counter


print("Creating the final checkpoint of the trainer.")
trainer.save(checkpoint_folder) # save checkpoint at the end of training

print("Experiment finished successfully! :)")


# Shutdown Ray
ray.shutdown()
