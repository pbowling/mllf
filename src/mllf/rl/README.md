# mllf.rl

Minimal RL scaffolding for training an A2C agent (Stable Baselines3).

Files:

- `env.py` — simple custom Gym environment `SimpleCustomEnv`.
- `wrappers.py` — environment factory `make_env`.
- `train.py` — script to train an A2C model (guarded by __main__).
- `eval.py` — load a saved model and evaluate mean reward.

Usage example (after installing dependencies):

python -m mllf.rl.train
