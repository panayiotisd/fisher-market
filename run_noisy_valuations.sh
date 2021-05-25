#!/bin/bash


##############################################################
# Random Seed
seed=42
##############################################################



##############################################################
# Market Equilibrium Calculation
##############################################################

# Base Case
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_uniform' --checkpoint_dir './checkpoints_true_uniform' --compute_market_eq True --valuations_noise_method 'uniform' > out_true_uniform.out 2>&1

##############################################################
# AI Policymaker
##############################################################

# Uniform Noise
noise=0.01
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_uniform_'${noise} --checkpoint_dir './checkpoints_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} > out_uniform_${noise}.out 2>&1


noise=0.025
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_uniform_'${noise} --checkpoint_dir './checkpoints_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} > out_uniform_${noise}.out 2>&1


noise=0.05
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_uniform_'${noise} --checkpoint_dir './checkpoints_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} > out_uniform_${noise}.out 2>&1


noise=0.075
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_uniform_'${noise} --checkpoint_dir './checkpoints_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} > out_uniform_${noise}.out 2>&1


noise=0.1
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_uniform_'${noise} --checkpoint_dir './checkpoints_uniform_'${noise} --valuations_noise_method 'uniform' --valuations_noise ${noise} > out_uniform_${noise}.out 2>&1



# Split into bins
bins=100
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_bins_'${bins} --checkpoint_dir './checkpoints_bins_'${bins} --valuations_noise_method 'bins' --n_valuations_bins ${bins} > out_bins_${bins}.out 2>&1

bins=75
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_bins_'${bins} --checkpoint_dir './checkpoints_bins_'${bins} --valuations_noise_method 'bins' --n_valuations_bins ${bins} > out_bins_${bins}.out 2>&1

bins=50
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_bins_'${bins} --checkpoint_dir './checkpoints_bins_'${bins} --valuations_noise_method 'bins' --n_valuations_bins ${bins} > out_bins_${bins}.out 2>&1

bins=25
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_bins_'${bins} --checkpoint_dir './checkpoints_bins_'${bins} --valuations_noise_method 'bins' --n_valuations_bins ${bins} > out_bins_${bins}.out 2>&1

bins=10
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_bins_'${bins} --checkpoint_dir './checkpoints_bins_'${bins} --valuations_noise_method 'bins' --n_valuations_bins ${bins} > out_bins_${bins}.out 2>&1