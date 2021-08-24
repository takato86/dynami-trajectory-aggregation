import gym
from gym import wrappers
import numpy as np
import pandas as pd
import gym_fourrooms
import logging
import os
import json
import time
import argparse
from tqdm import tqdm
from utils.config import config
from entity.sarsa_agent import SubgoalRSSarsaAgent, SarsaAgent,\
                               NaiveSubgoalSarsaAgent,\
                               SarsaRSSarsaAgent, SRSSarsaAgent
from concurrent.futures import ProcessPoolExecutor


def export_steps(file_path, steps):
    with open(file_path, 'w') as f:
        f.write('\n'.join(list(map(str, steps))))


def export_runtimes(file_path, runtimes):
    runtimes_df = pd.DataFrame(runtimes, columns=["runtime"])
    runtimes_df.to_csv(file_path)


def export_details(file_path, details):
    detail_df = pd.DataFrame.from_dict(details, orient="index")
    detail_df.to_csv(file_path)


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
                  id, env_id, learn_id, nprocess):
    runtimes = []
    args = [
        [run, env, agent, nepisodes, nsteps, id, env_id, learn_id]
        for run in range(nruns)
    ]
    with ProcessPoolExecutor(max_workers=nprocess) as executor:
        ret = tqdm(executor.map(run_loop, args), total=nruns)
    runtimes = list(ret)

    # single process
    # runtimes = []
    # for arg in args:
    #     runtimes.append(
    #         run_loop(arg)
    #     )

    export_runtimes(
        os.path.join(
            runtimes_dir,
            f"{env_id}-{learn_id}-{id}.csv"
        ),
        runtimes
    )


def run_loop(args):
    run, env, agent, nepisodes, nsteps, id, env_id, learn_id = args
    start_time = time.time()
    agent.reset()
    steps = []
    runtimes = []
    details = {}
    for episode in range(nepisodes):
        next_observation = env.reset()
        logger.debug(f"start: {next_observation}, goal: {env.goal}")
        cumreward = 0
        for step in range(nsteps):
            observation = next_observation
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            # Critic update
            agent.update(observation, action, next_observation, reward, done)
            info = agent.info(observation)
            if info:
                ind = episode * nepisodes + step
                info["episode"] = episode
                info["step"] = step
                details[ind] = info
            cumreward += reward
            if done:
                break
        steps.append(step)
        logger.debug('Run {} episode {} steps {}, cumreward {:.4f}, min_Q: {:.4f}, max_Q: {:.4f}'.format(
                run, episode, step, cumreward,
                agent.get_min_value(), agent.get_max_value()
            )
        )
    runtimes.append(time.time() - start_time)
    export_steps(
        os.path.join(
            steps_dir,
            f"{env_id}-{learn_id}-{run}-{id}.csv"
        ),
        steps
    )
    export_details(
        os.path.join(
            detail_dir,
            f"{env_id}-{learn_id}-{run}-{id}.csv"
        ),
        details
    )
    return runtimes


def main():
    logger.info(
        "env: {}, alg: {}".format(
            config["ENV"]["env_id"], config["SHAPING"]["alg"]
        )
    )
    env_to_wrap = gym.make(config["ENV"]["env_id"])
    # env_to_wrap.env.init_states = [0]
    if config.getboolean("ENV", "video"):
        movie_folder = prep_dir(
            os.path.join(
                'out', 'movies', config["SHAPING"]["alg"]
            )
        )
        env = wrappers.Monitor(
            env_to_wrap, movie_folder, force=True,
            video_callable=(lambda ep: ep % 100 == 0 or (ep > 30 and ep < 35)))
    else:
        env = env_to_wrap

    # export_env(
    #   os.path.join(env_dir, f"{config["ENV"]["env_id"]}.csv"), env_to_wrap
    # )
    nfeatures, nactions = env.observation_space.n, env.action_space.n
    subgoals = [[]]
    if "subgoal" in config["SHAPING"]["alg"] or\
       "naive" == config["SHAPING"]["alg"] or\
       "srs" in config["SHAPING"]["alg"]:
        subgoals = load_subgoals(config["SHAPING"]["subgoal_path"], task_id=1)
        logger.info(f"subgoals: {subgoals}")
    elif "sarsa-rs" in config["SHAPING"]["alg"]:
        json_open = open(config["SHAPING"]["mapping_path"])
        aggr_set = [l for _, l in json.load(json_open).items()]
        logger.info(f"mappings: {aggr_set}")

    logger.info(
        "Start learning in the case of eta={}, rho={}".format(
            config["SHAPING"]["eta"], config["SHAPING"]["rho"]
        )
    )

    for learn_id, subgoal in enumerate(subgoals):
        logger.info(f"Subgoal {learn_id+1}/{len(subgoals)}: {subgoal}")
        rng = np.random.RandomState(learn_id)
        if "naive" == config["SHAPING"]["alg"]:
            logger.info("NaiveSubgoalSarsaAgent")
            agent = NaiveSubgoalSarsaAgent(
                float(config["AGENT"]["discount"]),
                float(config["AGENT"]["epsilon"]),
                float(config["AGENT"]["lr"]), nfeatures, nactions,
                float(config["AGENT"]["temperature"]), rng, subgoal,
                float(config["SHAPING"]["eta"]), float(config["SHAPING"]["rho"]),
                subgoal_values=None)
        elif "subgoal" in config["SHAPING"]["alg"]:
            logger.info("SubgoalRSSarsaAgent")
            agent = SubgoalRSSarsaAgent(
                float(config["AGENT"]["discount"]),
                float(config["AGENT"]["epsilon"]),
                float(config["AGENT"]["lr"]), nfeatures, nactions,
                float(config["AGENT"]["temperature"]), rng, subgoal,
                float(config["SHAPING"]["eta"]), float(config["SHAPING"]["rho"]),
                subgoal_values=None)
        elif "sarsa-rs" in config["SHAPING"]["alg"]:
            logger.debug("SarsaRSAgent")
            agent = SarsaRSSarsaAgent(
                float(config["AGENT"]["discount"]),
                float(config["AGENT"]["epsilon"]),
                float(config["AGENT"]["lr"]), nfeatures, nactions,
                float(config["AGENT"]["temperature"]), rng, aggr_set)
        elif "sarsa" in config["SHAPING"]["alg"]:
            logger.debug("SarsaAgent")
            agent = SarsaAgent(
                float(config["AGENT"]["discount"]),
                float(config["AGENT"]["epsilon"]),
                float(config["AGENT"]["lr"]), nfeatures, nactions,
                float(config["AGENT"]["temperature"]), rng)
        elif "srs" in config["SHAPING"]["alg"]:
            logger.debug("Subgoal Reward Shaping")
            agent = SRSSarsaAgent(
                float(config["AGENT"]["discount"]),
                float(config["AGENT"]["epsilon"]),
                float(config["AGENT"]["lr"]), env,
                float(config["AGENT"]["temperature"]),
                rng, float(config["SHAPING"]["eta"]),
                float(config["SHAPING"]["rho"]), subgoal
            )
        learning_loop(
            env, agent, int(config["AGENT"]["nruns"]),
            int(config["AGENT"]["nepisodes"]),
            int(config["AGENT"]["nsteps"]), config["SHAPING"]["alg"],
            config["ENV"]["env_id"], learn_id,
            int(config["ENV"]["nprocess"])
        )
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config.read(args.config)
    out_dir = prep_dir(os.path.join("out", config["SHAPING"]["alg"]))
    steps_dir = prep_dir(os.path.join(out_dir, "steps"))
    runtimes_dir = prep_dir(os.path.join(out_dir, "runtime"))
    detail_dir = prep_dir(os.path.join(out_dir, "detail"))
    main()
