# How to run
```bash
python main.py --config=in/configs/fourooms.ini
```


# [OLD] How to run
## Baseline RL
```bash
python main.py --discount=0.99 --epsilon=0.05 --lr_critic=0.25 --nruns=100 --nsteps=10000 --nepisodes=500 --env_id="ConstFourrooms-v0" --id="sarsa"
```

## +Reward Shaping
### Subgoal-based RS with dynamic potentials
```bash
python main.py --discount=0.99 --epsilon=0.05 --lr_critic=0.25 --nruns=100 --nsteps=10000 --nepisodes=500 --env_id="ConstFourrooms-v0" --subgoal-path="in/subgoals/fourrooms_human_subgoals.csv" --id="subgoal"
```
You can switch subgoals by changing the path. 

### Subgoal-based RS with static potentials
```bash
python main.py --discount=0.99 --epsilon=0.05 --lr_critic=0.25 --nruns=100 --nsteps=10000 --nepisodes=500 --env_id="ConstFourrooms-v0" --eta=1.0 --rho=0.0 --subgoal-path="in/subgoals/fourroom_human_subgoals.csv" --id="srs-human"
```

### Subgoal naive RS
```bash
python main.py --discount=0.99 --epsilon=0.05 --lr_critic=0.25 --nruns=100 --nsteps=10000 --nepisodes=500 --env_id="ConstFourrooms-v0" --eta=1.0 --rho=0.0 --subgoal-path="in/subgoals/fourroom_human_subgoals.csv" --id="naive"
```

### SARSA-RS
```bash
python main.py --discount=0.99 --epsilon=0.05 --lr_critic=0.25 --nruns=100 --nsteps=10000 --nepisodes=500 --env_id="ConstFourrooms-v0" --eta=1.0 --rho=0.0 --subgoal-path="in/mappings/fourrooms_human_mapping.json" --id="sarsa-rs"
```

## Parameters
