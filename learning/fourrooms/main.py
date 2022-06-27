from datetime import datetime
import gym
from gym import wrappers
import numpy as np
import pandas as pd
import gym_fourrooms
import logging
import os
import json
import time
import configparser
from tqdm import tqdm, trange
from entity.sg_parser import parser
import matplotlib.pyplot as plt
from entity.sarsa_agent import SubgoalRSSarsaAgent, SarsaAgent, NaiveSubgoalSarsaAgent,\
                               SarsaRSSarsaAgent, SRSSarsaAgent
from concurrent.futures import ProcessPoolExecutor


'''
avg_duration: 1つのOptionが続けられる平均ステップ数
step        : 1エピソードに要したステップ数
'''

ALG_CHOICES = [
    "subgoal",
    "naive",
    "sarsa",
    "sarsa-rs"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
inifile = configparser.ConfigParser()
inifile.read("../../config.ini")


def export_steps(file_path, steps):
    with open(file_path, 'w') as f:
        f.write('\n'.join(list(map(str, steps))))


def export_runtimes(file_path, runtimes):
    runtimes_df = pd.DataFrame(runtimes, columns=["runtime"])
    runtimes_df.to_csv(file_path)


def export_arguments(file_path, arguments):
    with open(file_path, mode="w", encoding="utf-8") as f:
        json.dump(vars(arguments), f)


def prep_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def load_subgoals(file_path, task_id=None):
    logger.info(f"Loading {file_path}")
    subgoal_df = pd.read_csv(file_path)
    if task_id is not None:
        subgoal_df = subgoal_df[subgoal_df["task_id"] == task_id]
    subg_serieses_df = subgoal_df.groupby(["user_id", "task_id"]).agg(list)
    subg_serieses = []
    for subg_series in list(subg_serieses_df['state'].values):
        subg_serieses.append([subg_series])
    return subg_serieses


def learning_loop(env, agent, nruns, nepisodes, nsteps, 
                  id, env_id, learn_id, nprocess=None): 
    runtimes = []
    args = [
        [run, env, agent, nepisodes, nsteps, id, env_id, learn_id]
        for run in range(nruns)
    ]
    with ProcessPoolExecutor(max_workers=nprocess) as executor:
        ret = tqdm(executor.map(run_loop, args), total=nruns)
    runtimes = list(ret)
    export_runtimes(
        os.path.join(
            runtimes_dir,
            f"{env_id}-{learn_id}-{id}.csv"
        ),
        runtimes
    )


def run_loop(args):
    """The multiprocessed method."""
    run, env, agent, nepisodes, nsteps, id, env_id, learn_id = args
    start_time = time.time()
    agent.reset()
    steps = []
    runtimes = []
    for episode in range(nepisodes):
        next_observation = env.reset()
        logger.debug(f"start state: {next_observation}")
        cumreward = 0
        logger.debug(f"goal is at {env.goal}")
        for step in range(nsteps):
            observation = next_observation
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            # Critic update
            agent.update(observation, action, next_observation, reward, done)
            cumreward += reward
            if done:
                logger.debug("true goal: {}, actual goal: {}, reward: {}"
                        .format(env.goal, next_observation, reward))
                break
        steps.append(step)
        logger.debug('Run {} episode {} steps {} cumreward {}'
                .format(run, episode, step, agent.total_shaped_reward))
    runtimes.append(time.time() - start_time)
    export_steps(os.path.join(steps_dir,
                                f"{env_id}-{learn_id}-{run}-{id}.csv"),
                    steps)
    return runtimes


def main():
    logger.info(f"Start learning")
    logger.info("env: {}, alg: {}".format(args.env_id, args.id))
    env_to_wrap = gym.make(args.env_id)
    # env_to_wrap.env.init_states = [0]
    if args.video:
        movie_folder = prep_dir(os.path.join('res', 'movies', id))
        env = wrappers.Monitor(env_to_wrap, movie_folder, force=True,
                               video_callable=(lambda ep: ep%100 == 0 or (ep>30 and ep<35)))
    else:
        env = env_to_wrap

    # export_env(os.path.join(env_dir, f"{args.env_id}.csv"), env_to_wrap)
    nfeatures, nactions = env.observation_space.n, env.action_space.n
    subgoals = [[]]
    if "subgoal" in args.id or "naive" == args.id or "srs" in args.id:
        subgoals = load_subgoals(args.subgoal_path, task_id=1)
        logger.info(f"subgoals: {subgoals}")
    elif "sarsa-rs" in args.id:
        json_open = open(args.mapping_path)
        aggr_set = [l for _, l in json.load(json_open).items()]
        logger.info(f"mappings: {aggr_set}")

    for learn_id, subgoal in enumerate(subgoals):
        logger.info(f"Subgoal {learn_id+1}/{len(subgoals)}: {subgoal}")
        rng = np.random.RandomState(learn_id)
        if "naive" == args.id:
            logger.info("NaiveSubgoalSarsaAgent")
            agent = NaiveSubgoalSarsaAgent(args.discount, args.epsilon, args.lr_critic, nfeatures, nactions,
                                    args.temperature, rng, subgoal, args.eta, args.rho, subgoal_values=None)
        elif "subgoal" in args.id:
            logger.info("SubgoalRSSarsaAgent")
            agent = SubgoalRSSarsaAgent(args.discount, args.epsilon, args.lr_critic, nfeatures, nactions,
                                    args.temperature, rng, subgoal, args.eta, args.rho, subgoal_values=None)
        elif "sarsa" == args.id:
            logger.debug("SarsaAgent")
            agent = SarsaAgent(args.discount, args.epsilon, args.lr_critic, nfeatures, nactions, args.temperature, rng)
        elif "sarsa-rs" in args.id:
            logger.debug("SarsaRSAgent")
            agent = SarsaRSSarsaAgent(args.discount, args.epsilon, args.lr_critic, nfeatures, nactions,
                                      args.temperature, rng, aggr_set)
        elif "srs" in args.id:
            logger.debug("Subgoal Reward Shaping")
            agent = SRSSarsaAgent(args.discount, args.epsilon, args.lr_critic, env, args.temperature, rng, args.eta, args.rho, subgoal)
    
        learning_loop(env, agent, args.nruns, args.nepisodes, args.nsteps,
                      args.id, args.env_id, learn_id, args.nprocess)
    env.close()


if __name__ == '__main__':
    parser.add_argument('--ac', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--id', default='no_name', type=str)  # , choices=ALG_CHOICES
    parser.add_argument('--rho', default=0.05, type=float)
    parser.add_argument('--nprocess', default=1, type=int)
    args = parser.parse_args()
    dirname = "{}-{}".format(datetime.now(), args.id)
    env_dir = prep_dir(os.path.join("res", dirname, "env"))
    val_dir = prep_dir(os.path.join("res", dirname, "values"))
    episode_dir = prep_dir(os.path.join("res", dirname, "episode"))
    policy_dir = prep_dir(os.path.join("res", dirname, "policy"))
    analysis_dir = prep_dir(os.path.join("res", dirname, "analysis"))
    steps_dir = prep_dir(os.path.join("res", dirname, "steps"))
    runtimes_dir = prep_dir(os.path.join("res", dirname, "runtime"))
    export_arguments(os.path.join("res", dirname, "config.json"), args)
    main()
