#!/bin/bash


##############################################################
# Random Seed
seed=42
##############################################################



##############################################################
# Market Equilibrium Calculation
##############################################################

# Base Case
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true' --checkpoint_dir './checkpoints_true' --compute_market_eq True  > out_true.out 2>&1

# Scarce Resources
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_Ms_0p45' --checkpoint_dir './checkpoints_true_Ms_0p45' --Ms 0.45 --compute_market_eq True  > out_true_Ms_0p45.out 2>&1



##############################################################
# AI Policymaker
##############################################################

# Base Case
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs' --checkpoint_dir './checkpoints'  > out.out 2>&1

# Scarce Resources
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_Ms_0p45' --checkpoint_dir './checkpoints_Ms_0p45' --Ms 0.45  > out_Ms_0p45.out 2>&1

# Scarce Resources with sustainability optimization
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_Ms_0p45_sustainability' --checkpoint_dir './checkpoints_Ms_0p45_sustainability' --Ms 0.45 --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=1.0 > out_Ms_0p45_sustainability.out 2>&1



# Harvesters' social welfare optimization
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_harvesters' --checkpoint_dir './checkpoints_harvesters' --policymaker_harvesters_welfare_weight=1.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=0.0 > out_harvesters.out 2>&1

# Buyers' social welfare optimization
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_buyers' --checkpoint_dir './checkpoints_buyers' --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=1.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=0.0 > out_buyers.out 2>&1

# Fairness optimization
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_fairness' --checkpoint_dir './checkpoints_fairness' --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=1.0 --policymaker_sustainability_weight=0.0 > out_fairness.out 2>&1

# Sustainability optimization
python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_sustainability' --checkpoint_dir './checkpoints_sustainability' --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=1.0 > out_sustainability.out 2>&1



##############################################################
# Alternative Fairness Metrics
##############################################################

python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_gini' --checkpoint_dir './checkpoints_true_gini' --compute_market_eq True --fairness_metric 'gini' > out_true_gini.out 2>&1

python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_true_atkinson' --checkpoint_dir './checkpoints_true_atkinson' --compute_market_eq True --fairness_metric 'atkinson' > out_true_atkinson.out 2>&1



python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_gini' --checkpoint_dir './checkpoints_gini' --fairness_metric 'gini' > out_gini.out 2>&1

python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_atkinson' --checkpoint_dir './checkpoints_atkinson' --fairness_metric 'atkinson' > out_atkinson.out 2>&1



python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_gini_fairness' --checkpoint_dir './checkpoints_gini_fairness' --fairness_metric 'gini' --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=1.0 --policymaker_sustainability_weight=0.0 > out_gini_fairness.out 2>&1

python fishery_market_simulator.py --num_workers 24 --random_seed ${seed} --log_dir './logs_atkinson_fairness' --checkpoint_dir './checkpoints_atkinson_fairness' --fairness_metric 'atkinson' --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=1.0 --policymaker_sustainability_weight=0.0 > out_atkinson_fairness.out 2>&1

