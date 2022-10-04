#!/bin/bash


##############################################################
# Random Seed
seed=42
##############################################################


# Uniform Noise
##############################################################
noise=0.05
##############################################################


# Market Equilibrium Calculation
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_true_effort_noise_uniform_'${noise} --compute_market_eq True --effort_noise_method 'uniform' --effort_noise ${noise} > out_true_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_noise_uniform_'${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (uniform)
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_valuations_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_valuations_noise_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_valuations_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (bins)
# TODO?


##############################################################
noise=0.1
##############################################################


# Market Equilibrium Calculation
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_true_effort_noise_uniform_'${noise} --compute_market_eq True --effort_noise_method 'uniform' --effort_noise ${noise} > out_true_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_noise_uniform_'${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (uniform)
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_valuations_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_valuations_noise_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_valuations_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (bins)
# TODO?







##############################################################
# Random Seed
seed=43
##############################################################


# Uniform Noise
##############################################################
noise=0.05
##############################################################


# Market Equilibrium Calculation
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_true_effort_noise_uniform_'${noise} --compute_market_eq True --effort_noise_method 'uniform' --effort_noise ${noise} > out_true_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_noise_uniform_'${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (uniform)
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_valuations_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_valuations_noise_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_valuations_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (bins)
# TODO?


##############################################################
noise=0.1
##############################################################


# Market Equilibrium Calculation
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_true_effort_noise_uniform_'${noise} --compute_market_eq True --effort_noise_method 'uniform' --effort_noise ${noise} > out_true_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_noise_uniform_'${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (uniform)
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_effort_valuations_noise_uniform_'${noise} --checkpoint_dir './checkpoints_effort_valuations_noise_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} --effort_noise_method 'uniform' --effort_noise ${noise} > out_effort_valuations_noise_uniform_${noise}.out 2>&1

# AI Policymaker with noisy valuations as well (bins)
# TODO?


