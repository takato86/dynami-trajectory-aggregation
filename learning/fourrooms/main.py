import gym
from gym import wrappers
import numpy as np
import pandas as pd
import gym_fourrooms
import logging
import os
import time
import configparser
from tqdm import tqdm, trange
from entity.sg_parser import parser
import matplotlib.pyplot as plt
from entity.sarsa_agent import SubgoalRSSarsaAgent, SarsaAgent, NaiveSubgoalSarsaAgent,\
                               SarsaRSSarsaAgent

'''
avg_duration: 1つのOptionが続けられる平均ステップ数
step        : 1エピソードに要したステップ数
'''

ALG_CHOICES = [
    "subgoal",
    "naive",
    "actor-critic",
    "sarsa-rs"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
inifile = configparser.ConfigParser()
inifile.read("../../config.ini")


def export_env(file_path, env):
    to_state = np.full(env.occupancy.shape, -1)
    for k, val in env.tostate.items():
        to_state[k[0]][k[1]] = val
    pd.DataFrame(to_state).to_csv(file_path)


def export_steps(file_path, steps):
    with open(file_path, 'w') as f:
        f.write('\n'.join(list(map(str, steps))))


def export_runtimes(file_path, runtimes):
    runtimes_df = pd.DataFrame(runtimes, columns=["runtime"])
    runtimes_df.to_csv(file_path)


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
                  id, env_id, learn_id): 
    runtimes = []
    for run in trange(nruns):
        start_time = time.time()
        agent.reset()
        steps = []
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
    export_runtimes(
        os.path.join(
            runtimes_dir,
            f"{env_id}-{learn_id}-{id}.csv"
        ),
        runtimes
    )


def main():
    logger.info("env: {}, alg: {}".format(args.env_id, args.id))
    env_to_wrap = gym.make(args.env_id)
    # env_to_wrap.env.init_states = [0]
    if args.video:
        movie_folder = prep_dir(os.path.join('res', 'movies', id))
        env = wrappers.Monitor(env_to_wrap, movie_folder, force=True,
                               video_callable=(lambda ep: ep%100 == 0 or (ep>30 and ep<35)))
    else:
        env = env_to_wrap

    export_env(os.path.join(env_dir, f"{args.env_id}.csv"), env_to_wrap)
    nfeatures, nactions = env.observation_space.n, env.action_space.n
    subgoals = [[]]
    if "subgoal" in args.id:
        subgoals = load_subgoals(args.subgoal_path, task_id=1)
        logger.info(f"subgoals: {subgoals}")

    logger.info(f"Start learning in the case of eta={args.eta}, rho={args.rho}")

    for learn_id, subgoal in enumerate(subgoals):
        logger.info(f"Subgoal {learn_id+1}/{len(subgoals)}: {subgoal}")
        rng = np.random.RandomState(learn_id)
        if "naive" in args.id and "subgoal" in args.id:
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
        elif "sarsa-rs" == args.id:
            logger.debug("SarsaRSAgent")
            upper_left = [
                0, 1, 2, 3, 4, 
                10, 11, 12, 13, 14,
                20, 21, 22, 23, 24, 25,
                31, 32, 33, 34, 35,
                41, 42, 43, 44, 45, 51
            ]
            upper_right = [
                5, 6, 7 ,8, 9,
                15, 16, 17, 18, 19,
                26, 27, 28, 29, 30,
                36, 37, 38, 39, 40,
                46, 47, 48, 49, 50,
                52, 53, 54, 55, 56, 62
            ]
            lower_left = [
                57, 58, 59, 60, 61,
                63, 64, 65, 66, 67,
                73, 74, 75, 76, 77,
                83, 84, 85, 86, 87, 88,
                94, 95, 96, 97, 98
            ]
            lower_right = [
                68, 69, 70, 71, 72,
                78, 79, 80, 81, 82,
                89, 90, 91, 92, 93,
                99, 100, 101, 102, 103
            ]
            aggr_set = [upper_left, upper_right, lower_left, lower_right]
            agent = SarsaRSSarsaAgent(args.discount, args.epsilon, args.lr_critic, nfeatures, nactions,
                                      args.temperature, rng, aggr_set)

        learning_loop(env, agent, args.nruns, args.nepisodes, args.nsteps,
                      args.id, args.env_id, learn_id)
    env.close()


if __name__ == '__main__':
    parser.add_argument('--ac', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--id', default='no_name', choices=ALG_CHOICES, type=str)
    parser.add_argument('--rho', default=0.05, type=float)
    args = parser.parse_args()
    env_dir = prep_dir(os.path.join("res", "env"))
    val_dir = prep_dir(os.path.join("res", "values"))
    episode_dir = prep_dir(os.path.join("res", "episode"))
    policy_dir = prep_dir(os.path.join("res", "policy"))
    analysis_dir = prep_dir(os.path.join("res", "analysis"))
    steps_dir = prep_dir(os.path.join("res", "steps"))
    runtimes_dir = prep_dir(os.path.join("res", "runtime"))
    main()