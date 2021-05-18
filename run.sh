# Base cases

python fishery_market_simulator.py --num_workers 24 --n_episodes 10000  > out_24_10000.out 2>&1

python fishery_market_simulator.py --num_workers 24 --n_episodes 1000 --compute_market_eq True  > out_24_1000_true.out 2>&1



# Run each objective independently

python fishery_market_simulator.py --num_workers=24 --n_episodes=2400 --policymaker_harvesters_welfare_weight=1.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=0.0 > out_24_5000_policymaker_harvesters_welfare_weight_1.out 2>&1

python fishery_market_simulator.py --num_workers=24 --n_episodes=2400 --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=1.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=0.0 > out_24_5000_policymaker_buyers_welfare_weight_1.out 2>&1

python fishery_market_simulator.py --num_workers=24 --n_episodes=2400 --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=1.0 --policymaker_sustainability_weight=0.0 > out_24_5000_policymaker_fairness_weight_1.out 2>&1

python fishery_market_simulator.py --num_workers=24 --n_episodes=2400 --policymaker_harvesters_welfare_weight=0.0 --policymaker_buyers_welfare_weight=0.0 --policymaker_fairness_weight=0.0 --policymaker_sustainability_weight=1.0 > out_24_5000_policymaker_sustainability_weight_1.out 2>&1


# Run in scarcer environments

python fishery_market_simulator.py --num_workers 24 --n_episodes 2400 --Ms 0.45 > out_24_5000_Ms_0p45.out 2>&1

python fishery_market_simulator.py --num_workers 24 --n_episodes 1000 --Ms 0.45 --compute_market_eq True > out_24_1000_Ms_0p45_true.out 2>&1