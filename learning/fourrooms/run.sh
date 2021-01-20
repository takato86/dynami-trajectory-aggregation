ETA=1.0
RHO=0.001
EPSILON=0.05
NRUNS=500  # 100 * 10
NEPISODES=1000
ALG="naive"
RLENV="DiagonalPartialFourrooms-v0"
echo "${ALG} on ${RLENV}"
python main.py --discount=0.99 --epsilon=$EPSILON --lr_critic=0.25\
               --nruns=$NRUNS --nsteps=10000 --nepisodes=$NEPISODES \
               --env_id=$RLENV \
               --subgoal-path="in/subgoals/diagonalpartialfourrooms_optimal_subgoals.csv" \
               --id=$ALG