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
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy
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
                                      fairness_metric = fairness_metric,
                                      random_seed = random_seed,
                                      compute_market_eq = compute_market_eq,
                                      debug = debug)


# Log saving function
def save_log(log_dir, ep_memory, ag_memory):
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
          'fairness_metric' : fairness_metric,
          'random_seed' : random_seed,
          'compute_market_eq' : compute_market_eq,
          'num_workers' : num_workers,
          'n_episodes' : n_episodes,
          'train_algo' : train_algo,
          'lr' : lr,
          'gamma' : gamma,
          'hidden_layer_size' : hidden_layer_size,
          'episodes': ep_memory}

    filename = "logs_ep_memory_H_{0}_B_{1}_R_{2}_{3}_{4:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, start_time)
    path = log_dir + '/' + filename
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
          'fairness_metric' : fairness_metric,
          'random_seed' : random_seed,
          'compute_market_eq' : compute_market_eq,
          'num_workers' : num_workers,
          'n_episodes' : n_episodes,
          'train_algo' : train_algo,
          'lr' : lr,
          'gamma' : gamma,
          'hidden_layer_size' : hidden_layer_size,
          'episodes': ag_memory}

    filename = "logs_ag_memory_H_{0}_B_{1}_R_{2}_{3}_{4:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, start_time)
    path = log_dir + '/' + filename
    try:
        print("Saving ag_memory logs to: {}".format(path))
        with gzip.open(path, "wb") as fp:
            # pickle.dump(log, fp)
          pickled = pickle.dumps(log)
          optimized_pickle = pickletools.optimize(pickled)
          fp.write(optimized_pickle)
    except:
        print("An exception occurred while saving ag_memory logs to {}!".format(path))


# policy mapping function
def policy_mapping_fn(agent_id):
    return "policy_" + agent_id


# Callback function - episode start
def on_episode_start(info):
    # Initializations
    episode = info["episode"]
    episode.user_data["observations"] = []
    episode.user_data["actions"] = []

    episode.user_data["harvester_fairness"] = []
    episode.user_data["stock_difference"] = []

    episode.user_data["harvester_rewards"] = []
    episode.user_data["harvester_revenue"] = []
    episode.user_data["wasted_percentage"] = []
    episode.user_data["buyers_utility"] = []
    
    # Flag to track the harvest and policymaking steps to get the appropriate metrics from the info dictionaries
    global harvesting_step, first_step
    harvesting_step = True # The environment starts with a harvest step
    first_step = True


# Callback function - episode step
def on_episode_step(info):
    episode = info["episode"]
    
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
    global harvesting_step
    global first_step
    # print('======================on_episode_step======================')
    # print('harvesting_step = ' + str(harvesting_step))
    if harvesting_step and not first_step:
      info_dict = episode.last_info_for(l_harvesters[0]) # All harvesters have the same info
      assert (len(info_dict) != 0)

      # print('Supplementary environment information:')
      # print(info_dict)
      # print()

      episode.user_data["harvester_rewards"].append(info_dict["harvester_rewards"])
      episode.user_data["harvester_revenue"].append(info_dict["harvester_revenue"])
      episode.user_data["wasted_percentage"].append(info_dict["wasted_percentage"])
      episode.user_data["buyers_utility"].append(info_dict["buyers_utility"])
  
    elif not harvesting_step:
      assert first_step == False

      info_dict = episode.last_info_for(l_policymakers[0]) # We only have one policymaker
      assert (len(info_dict) != 0)

      # print('Supplementary environment information:')
      # print(info_dict)
      # print()

      if not (info_dict["harvester_fairness"] == -np.inf and info_dict["stock_difference"] == -np.inf): # Disregard the last two values
          episode.user_data["harvester_fairness"].append(info_dict["harvester_fairness"])
          episode.user_data["stock_difference"].append(info_dict["stock_difference"])
         

    harvesting_step = not harvesting_step
    first_step = False
    
    
    
# Callback function - episode end
def on_episode_end(info):
    # print('---------------------------------------------on_episode_end')

    global ep_number
    global ep_memory
    global ag_memory
    
    episode = info["episode"]


    # Calculate fairness based on the cumulative rewards
    agent_rewards = episode.agent_rewards.copy()  # Summed rewards broken down by agent.
    if l_policymakers[0] in agent_rewards:
        del agent_rewards[l_policymakers[0]]
    agent_rewards = agent_rewards.values()
    agent_rewards = np.array(list(agent_rewards))
    harvester_fairness_at_end = fairness_fn(agent_rewards)

    # Get episode data lists saved during each step
    actions = episode.user_data["actions"]
    observations = episode.user_data["observations"]

    harvester_fairness = episode.user_data["harvester_fairness"]
    stock_difference = episode.user_data["stock_difference"]

    harvester_rewards = episode.user_data["harvester_rewards"]
    harvester_revenue = episode.user_data["harvester_revenue"]
    wasted_percentage = episode.user_data["wasted_percentage"]
    buyers_utility = episode.user_data["buyers_utility"]

    # TODO: Calculate episode metrics
    harvester_cumulative_reward = np.copy(agent_rewards)

    # print('--------------------------------------------- ep_number = ' + str(ep_number))
    # if ep_number < n_episodes:

    # Add data to agent and episode memory (we split into two to make the pickled files more manageable in size)
    ag_memory.append({'ep_number': ep_number, 'ep_len': episode.length,
                    'actions': actions, 'observations': observations,
                    'metrics':{'H_rew': harvester_cumulative_reward, 'H_fair' : harvester_fairness_at_end}})
    ep_memory.append({'ep_number': ep_number, 'ep_len': episode.length,
                    'agent_rewards': agent_rewards,
                    'harvester_fairness_at_end' : harvester_fairness_at_end,
                    'harvester_fairness': harvester_fairness,
                    'stock_difference': stock_difference,
                    'harvester_rewards': harvester_rewards,
                    'harvester_revenue': harvester_revenue,
                    'wasted_percentage': wasted_percentage,
                    'buyers_utility': buyers_utility})
  
    
    # Save periodically and at the end
    if (((ep_number + 1) * num_workers) % epdata_save_freq == 0) or (((ep_number + 1) * num_workers) >= n_episodes):
        print("on_episode_end: Saving log at ep_number = " + str(ep_number))
        save_log(log_dir, ep_memory, ag_memory)

    ep_number += 1


# Callback function - train result
def on_train_result(info):
    # print('---------------------------------------------on_train_result')
    # print(info["result"].keys())
    # print()

    global ep_number
    if ((ep_number + 1) * num_workers) < n_episodes:
        return

    # Find picklable objects
    f = open(log_dir + '/' + 'temp', 'wb')
    results_info = {}
    for key in info["result"].keys():
        try:
            pickle.dump(info["result"][key], f)
        except Exception as e:
            continue
        else:
            results_info[key] = info["result"][key]
    f.close()
    os.remove(log_dir + '/' + 'temp') 

    # print(results_info)

    filename = "train_result_info_H_{0}_B_{1}_R_{2}_{3}_{4:%Y%m%d_%H_%M_%S_%f}".format(n_harvesters, n_buyers, n_resources, compute_market_eq, start_time)
    path = log_dir + '/' + filename
    try:
        print("Saving training results info to: {}".format(path))
        with open(path, "wb") as fp:
            pickle.dump(results_info, fp)
    except Exception as e:
        print()
        print("An exception occurred while saving training results info to {}!".format(path))
        print(e)
        print()


def fairness_fn(harvester_rewards):
  if (fairness_metric == 'jain'):
    return jain_index_fn(harvester_rewards)
  elif (fairness_metric == 'gini'):
    return 1 - gini_coefficient_fn(harvester_rewards) # We maximize fairness. In Gini coefficient an allocation is fair iff the coefficient is 0.
  else:
    raise ValueError('Invalid fairness metric: ' + self.fairness_metric)


def jain_index_fn(rewards):
  if np.count_nonzero(rewards) == 0:
    return 1	# Fair allocation; everybody got reward 0
  rewards = rewards.astype(np.float64)

  return np.sum(rewards) ** 2 / ( np.sum(rewards ** 2) * rewards.shape[0] )


def gini_coefficient_fn(rewards):
  if np.count_nonzero(rewards) == 0:
    return 0	# Fair allocation; everybody got reward 0
  rewards = rewards.astype(np.float64)

  G = np.sum(np.abs(rewards - np.array([np.roll(rewards,i) for i in range(rewards.shape[0])])))
  G /= sum(rewards) * 2 * rewards.shape[0]
  return G


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

# Training arguments
parser.add_argument('--n_episodes', default=3000, type=int)
parser.add_argument('--max_steps', default=500, type=int)

# Workers (parallelism)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--num_cpus_per_worker', default=1, type=int)


args = parser.parse_args()

print("-----------------------------------------")
print("Parsed arguments:")
print("n_harvesters = " + str(args.n_harvesters))
print("n_buyers = " + str(args.n_buyers))
print("n_resources = " + str(args.n_resources))
print("Ms = " + str(args.Ms))
print("compute_market_eq = " + str(args.compute_market_eq))

print("n_episodes = " + str(args.n_episodes))
print("max_steps = " + str(args.max_steps))

print("num_workers = " + str(args.num_workers))
print("num_cpus_per_worker = " + str(args.num_cpus_per_worker))
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
harvester_wastefulness_cost = 1.0
policymaker_harvesters_welfare_weight = 1.0
policymaker_buyers_welfare_weight = 1.0
policymaker_fairness_weight = 1.0
policymaker_wastefulness_weight = 0		# FIXME: 0?
policymaker_sustainability_weight = 1.0
fairness_metric = 'jain'

compute_market_eq = args.compute_market_eq  #  default: False
random_seed = 42
debug = False


# ******** Training Parameters ********
n_episodes = args.n_episodes  # default: 3000
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

log_dir='./logs' # directory to save episode logs
checkpoint_dir = './checkpoints'
epdata_save_freq = 500
checkpoint_interval = 500


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

# Initialize global variables for the callback functions
ep_number = 0
ep_memory = []
ag_memory = []


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
policy_graphs_harvester = dict([("policy_harvester{}".format(i), (None, obs_space_harvester, act_space_harvester, {})) 
    for i in range(n_harvesters)])
policy_graphs_policymaker = dict([("policy_policymaker{}".format(i), (None, obs_space_policymaker, act_space_policymaker, {})) 
    for i in range(n_policymakers)])
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

  "multiagent": {
    "policies": policy_graphs,
    "policy_mapping_fn": policy_mapping_fn,
    "policies_to_train": ["policy_harvester{}".format(i) for i in range(n_harvesters)] + ["policy_policymaker{}".format(i) for i in range(n_policymakers)],
  },
  "callbacks": {
    "on_episode_start": tune.function(on_episode_start),
    "on_episode_step": tune.function(on_episode_step),
    "on_episode_end": tune.function(on_episode_end),
    "on_train_result": tune.function(on_train_result),
  },
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



episode_rewards_mean = [] # for direct cumulative agent rewards



###########################################
# @title Run Training
###########################################


last_checkpoint = 0
episodes_counter = 0
while episodes_counter < n_episodes:
  result = trainer.train()
  episodes_counter = result['episodes_total']
  # print('========================================== episodes_counter = ' + str(episodes_counter))

  # episode_reward = result['episode_reward_mean']  # mean total reward of all agents
  
  # if (early_stop_enable):
  #     # stopping conditions: close to maximum episode length and no inrease in reward for a number of episodes
    
  #     if (eplen_max_started):  # already started counting
      
  #       # if significantly higher or lower reward obtained, restart limit
  #         if (episode_reward > eplen_max_base_reward*reward_increase_limit
  #           or episode_reward < eplen_max_base_reward*reward_decrease_limit):
  #             eplen_max_base_reward = episode_reward
  #             eplen_max_start_ep = episodes_counter
  #         else: # no higher reward obtaine, check stopping condition
  #             if (episodes_counter - eplen_max_start_ep > cut_eps):
  #                 break
      
  #     else: # first time, not yet started counting
  #         eplen_max_started = True
  #         eplen_max_base_reward = episode_reward
  #         eplen_max_start_ep = episodes_counter

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
