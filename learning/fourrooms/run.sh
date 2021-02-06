EPSILON=0.05
NRUNS=1000  # 100 * 10
NEPISODES=1000
ALG="sarsa-rs-optimal-para"
RLENV="DiagonalFourrooms-v0"
echo "${ALG} on ${RLENV}"
python main.py --discount=0.99 --epsilon=$EPSILON --lr_critic=0.25\
               --nruns=$NRUNS --nsteps=10000 --nepisodes=$NEPISODES \
               --env_id=$RLENV \
               --mapping-path="in/mapping/diagonal_fourrooms_optimal_mapping.json" \
               --id=$ALG