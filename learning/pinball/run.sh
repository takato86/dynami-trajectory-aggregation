for ETA in 100
do
echo "human-srs-eta=${ETA}"
python main.py --id="human-srs-eta=${ETA}" --nruns=100 --nepisodes=200 --subg-path="in/subgoals/human_subgoals.csv" --eta=$ETA --rho=0
echo "random-srs-eta=${ETA}"
python main.py --id="random-srs-eta=${ETA}" --nruns=100 --nepisodes=200 --subg-path="in/subgoals/random_subgoals.csv" --eta=$ETA --rho=0
done
# python main.py --id='sarsa-rs-jk3' --nruns=100 --nepisodes=200 --k=3
# python main.py --id="actor-critic" --nruns=100 --nepisodes=200
# python main.py --id="online-subgoal-human" --nruns=10 --nepisodes=200 --subg-path="in/subgoals/human_subgoals.csv"
# python main.py --id="naive-subgoal-human" --nruns=10 --nepisodes=200 --subg-path="in/subgoals/human_subgoals.csv"
# python main.py --id="online-subgoal-random" --nruns=10 --nepisodes=200 --subg-path="in/subgoals/random_subgoals.csv"