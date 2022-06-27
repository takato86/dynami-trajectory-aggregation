EPSILON=0.05
NRUNS=1000  # 100 * 10
NEPISODES=1000 # 1000 for eval.
RHO=0
RLENV="ConstFourrooms-v0"
for ETA in 0.01
do
ALG="random-srs-eta=${ETA}"
echo "${ALG} on ${RLENV}"
python main.py --discount=0.99 --epsilon=$EPSILON --lr_critic=0.25 \
               --nruns=$NRUNS --nsteps=10000 --nepisodes=$NEPISODES \
               --env_id=$RLENV \
               --subgoal-path="in/subgoals/fourrooms_random_subgoals.csv" \
               --id=$ALG \
               --eta=$ETA \
               --rho=$RHO
done
