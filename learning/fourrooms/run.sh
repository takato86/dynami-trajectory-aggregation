ETA=1.0
RHO=0.001
EPSILON=0.05
NRUNS=100
NEPISODES=1000
python main.py --discount=0.99 --epsilon=$EPSILON --lr_critic=0.25\
               --nruns=$NRUNS --nsteps=10000 --nepisodes=$NEPISODES \
               --env_id="ConstFourrooms-v0" --eta=$ETA --rho=$RHO \
               --id="sarsa" --subgoal-path="in/subgoals/fourrooms_human_subgoals.csv"
            #    online-human-subgoal