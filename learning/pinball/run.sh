# python main.py --id='srs' --nruns=100 --nepisodes=200 --subg-path="in/subgoals/human_subgoals.csv"
python main.py --id='sarsa-rs-jk3' --nruns=100 --nepisodes=200 --k=3
# python main.py --id="actor-critic" --nruns=100 --nepisodes=200
# python main.py --id="online-subgoal-human" --nruns=10 --nepisodes=200 --subg-path="in/subgoals/human_subgoals.csv"
# python main.py --id="naive-subgoal-human" --nruns=10 --nepisodes=200 --subg-path="in/subgoals/human_subgoals.csv"
# python main.py --id="online-subgoal-radom" --nruns=10 --nepisodes=200 --subg-path="in/subgoals/random_subgoals.csv"