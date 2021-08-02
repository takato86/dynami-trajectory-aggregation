import argparse
import time
import gym
from datetime import datetime
import os
import logging
from gym import wrappers, logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from entity.ac_agent import SubgoalACAgent, \
                            ActorCriticAgent, \
                            NaiveSubgoalACAgent, \
                            SarsaRSACAgent,\
                            SRSACAgent
from entity.mapping import Mapper
import gym_pinball
from tqdm import tqdm, trange
from visualizer import Visualizer
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_TYPES = [
    'total_reward',
    'td_error',
    'steps', 'runtime'
]
ALG_CHOICES = [
    "subgoal",
    "naive",
    "actor-critic",
    "sarsa-rs",
    "srs"
]


def export_csv(file_path, file_name, array):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    array = pd.DataFrame(array)
    saved_path = os.path.join(file_path, file_name)
    array.to_csv(saved_path)


def moved_average(data, window_size):
    b=np.ones(window_size)/window_size
    return np.convolve(data, b, mode='same')


def load_subgoals(file_path):
    logger.info(f"Loading {file_path}")
    subgoals_df = pd.read_csv(file_path)
    subgoals = subgoals_df.groupby(["user_id", "task_id"]).agg(list)
    xs = subgoals["x"].values.tolist()
    ys = subgoals["y"].values.tolist()
    rads = subgoals["rad"].values.tolist()
    subg_serieses = []
    for x, y, rad in zip(xs, ys, rads):
        subg_series = []
        for x_i, y_i, rad_i in zip(x, y, rad):
            subg_series.append({
                "pos_x": x_i,
                "pos_y": y_i,
                "rad": rad_i
                })
        subg_serieses.append([subg_series])
    return subg_serieses


def load_subgoals_new(file_path):
    logger.info(f"Loading {file_path}")
    subgoals_df = pd.read_csv(file_path)
    subgoals = subgoals_df.groupby(["user_id", "task_id"]).agg(list)
    xs = subgoals["x"].values.tolist()
    ys = subgoals["y"].values.tolist()
    rads = subgoals["rad"].values.tolist()
    subg_serieses = []
    for x, y, rad in zip(xs, ys, rads):
        subg_series = []
        for x_i, y_i, rad_i in zip(x, y, rad):
            subg_series.append(
                np.array([
                    x_i, y_i, np.nan, np.nan
                ])
            )
        subg_serieses.append(subg_series)
    return subg_serieses


def get_file_names(exe_id, l_id, run, eta=None, rho=None, k=None):
    fnames = {}
    for _type in OUTPUT_TYPES:
        fname = f"{exe_id}_{_type}_{l_id}_{run}"
        if eta is not None:
            fname += "_eta={}".format(eta)
        if rho is not None:
            fname += "_rho={}".format(rho)
        if k is not None:
            fname += "_k={}".format(k)
        fname += ".csv"
        fnames[_type] = fname
    return fnames


def learning_loop(args):
    run, env_id, episode_count, model, visual, exe_id, rho, eta, subgoals, l_id, k = args
    logger.debug(f"start run {run}")
    subg_confs = list(itertools.chain.from_iterable(subgoals))
    env = gym.make(env_id, subg_confs=subg_confs)
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    if "subgoal" == exe_id:
        logger.debug("Subgoal AC Agent")
        agent = SubgoalACAgent(run, env.action_space, env.observation_space, rho=rho, eta=eta, subgoals=subgoals)
        fnames = get_file_names(exe_id, l_id, run)
    elif "actor-critic" == exe_id:
        logger.debug("Actor-Critic Agent")
        agent = ActorCriticAgent(run, env.action_space, env.observation_space)
        fnames = get_file_names(exe_id, l_id, run)
    elif "naive" == exe_id:
        logger.debug("Naive Subgoal AC Agent")
        agent = NaiveSubgoalACAgent(run, env.action_space, env.observation_space, rho=rho, eta=eta, subgoals=subgoals)
        fnames = get_file_names(exe_id, l_id, run, eta, rho)
    elif "sarsa-rs" in exe_id:
        logger.debug("Sarsa RS AC Agent")
        params = {
            "vid": "table",
            "aggr_id": "ndisc",
            "params": {
                "env": env,
                "n": k
            }
        }
        agent = SarsaRSACAgent(
            run, env.action_space, env.observation_space, env, params
        )
        fnames = get_file_names(exe_id, l_id, run, k=k)
    elif "srs" in exe_id:
        logger.debug("Subgoal-based Reward Shaping AC Agent")
        params = {
            "vid": "table",
            "aggr_id": "dta",
            "eta": eta,
            "rho": rho,
            "params":{
                "env_id": env.spec.id,
                "_range": 0.04,
                "n_obs": env.observation_space.shape[0],
                "subgoals": subgoals
            }
        }
        agent = SRSACAgent(
            run, env, params
        )
        fnames = get_file_names(exe_id, l_id, run, eta, rho)
    else:
        raise NotImplemented

    vis = Visualizer(["ACC_X", "ACC_Y", "DEC_X", "DEC_Y", "NONE"])
    if model:
        agent.load_model(model)
    reward = 0
    done = False
    total_reward_list = []
    steps_list = []
    max_q_list = []
    max_q_episode_list = []
    runtimes = []
    max_q = 0.0
    start_time = time.time()
    for i in range(episode_count):
        total_reward = 0
        total_shaped_reward = 0
        n_steps = 0
        ob = env.reset()
        action = agent.act(ob)
        pre_action = action
        is_render = False
        while True:
            if (i+1) % 20 == 0 and visual:
                env.render()
                is_render = True
            pre_obs = ob
            ob, reward, done, _ = env.step(action)
            # TODO
            reward = 0 if reward < 0 else reward
            n_steps += 1
            # rand_basis = np.random.uniform()
            pre_action = action
            action = agent.act(ob)
            shaped_reward = agent.update(pre_obs, pre_action, reward, ob, action, done)
            total_reward += reward
            total_shaped_reward += shaped_reward
            tmp_max_q = agent.get_max_q(ob)
            max_q_list.append(tmp_max_q)
            max_q = tmp_max_q if tmp_max_q > max_q else max_q
            if done:
                logger.debug("episode: {}, steps: {}, total_reward: {}, total_shaped_reward: {}, max_q: {}, max_td_error: {}"
                        .format(i, n_steps, total_reward, int(total_shaped_reward), int(max_q), int(agent.get_max_td_error())))
                total_reward_list.append(total_reward)
                steps_list.append(n_steps)
                break
            
            if is_render:
                vis.set_action_dist(agent.vis_action_dist, action)
                vis.pause(.0001)
        max_q_episode_list.append(max_q)
        agent.save_model(d_kinds["mo"], i)
    # export process
    runtimes.append(time.time() - start_time)
    export_csv(
        d_kinds['tr'],
        fnames['total_reward'],
        total_reward_list
    )
    td_error_list = agent.td_error_list
    export_csv(
        d_kinds["td"],
        fnames['td_error'],
        td_error_list
    )
    total_reward_list = np.array(total_reward_list)
    steps_list = np.array(steps_list)
    max_q_list = np.array(max_q_list)
    logger.debug("Average return: {}".format(np.average(total_reward_list)))
    steps_file_path = os.path.join(d_kinds["st"], fnames['steps'])
    pd.DataFrame(steps_list).to_csv(steps_file_path)
    runtime_file_path = os.path.join(d_kinds["ru"], fnames['runtime'])
    runtimes_df = pd.DataFrame(runtimes, columns=["runtime"]).to_csv(runtime_file_path)
    env.close()


def main():
    logger.info("ENV: {}".format(args.env_id))
    learning_time = time.time()
    if "srs" in args.id:
        subg_serieses = load_subgoals_new(args.subg_path)
    elif len(args.subg_path) == 0:
        logger.debug("Nothing subgoal path.")
        subg_serieses = [[[{"pos_x":0.512, "pos_y": 0.682, "rad":0.04}, {"pos_x":0.683, "pos_y":0.296, "rad":0.04}]]] # , {"pos_x":0.9 , "pos_y":0.2 ,"rad": 0.04}
    else:
        subg_serieses = load_subgoals(args.subg_path)
    
    for l_id, subg_series in enumerate(subg_serieses):
        logger.debug(f"learning: {l_id+1}/{len(subg_serieses)}")
        logger.debug(f"subgoals: {subg_series}")
        arguments = [
            [
                run, args.env_id, args.nepisodes, args.model,
                args.vis, args.id, args.rho, args.eta, subg_series, l_id, args.k
            ]
            for run in range(args.nruns)
        ]
        with ProcessPoolExecutor() as executor:
            tqdm(executor.map(learning_loop, arguments), total=args.nruns)
        
        # Single Process
        # for run in range(args.nruns):
        #     logger.debug(f"Run: {run+1}/{args.nruns}")
        #     learning_loop(arguments[run])
        #     # Close the env and write monitor result info to disk
    duration = time.time() - learning_time
    logger.info("Learning time: {}m {}s".format(int(duration//60), int(duration%60)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Actor-Critic Learning.')
    parser.add_argument('env_id', nargs='?', default='Pinball-Subgoal-v0', help='Select the environment to run.')
    parser.add_argument('--vis', action='store_true', help='Attach when you want to look visual results.')
    parser.add_argument('--model', help='Input model dir path')
    parser.add_argument('--nepisodes', default=250, type=int)
    parser.add_argument('--nruns', default=25, type=int)
    parser.add_argument('--id') # , choices=ALG_CHOICES
    parser.add_argument('--subg-path', default='', type=str)
    parser.add_argument('--eta', default=10000, type=float)
    parser.add_argument('--rho', default=0.01, type=float)
    parser.add_argument('--k', type=int, help='How many to discretize the observation related to space.')
    
    args = parser.parse_args()
    saved_dir = os.path.join("out", args.id)
    d_kinds = {
        "tr": os.path.join(saved_dir, "total_reward"),
        "st": os.path.join(saved_dir, "steps"),
        "td": os.path.join(saved_dir, "td_error"),
        "mo": os.path.join(saved_dir, "model"),
        "ru": os.path.join(saved_dir, "runtime")
    }
    for fpath in d_kinds.values():
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    main()
