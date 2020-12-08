import pandas as pd
import numpy as np
import glob
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_file_list(file_pattern):
    return glob.glob(file_pattern)


def export(out_file_path, content):
    l = max(map(lambda x: len(x[1]), content.items()))
    filled_content = {}
    try:
        for key, val in content.items():
            filled_content[key] = list(val) + [None]*(l-len(val))
    except ValueError:
        import pdb; pdb.set_trace()
    export_df = pd.DataFrame(filled_content)
    export_df.to_csv(out_file_path)
    logger.info(f"Export {out_file_path}")


def load_total_reward(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = []
    for file_path in file_list:
        logger.info(f"Processing on {file_path} ....")
        total_reward_df = pd.read_csv(file_path, index_col=0)
        total_reward = total_reward_df.values.reshape(len(total_reward_df.index))
        total_rewards.append(total_reward.tolist())
    return total_rewards


def average_values(file_pattern):
    file_list = get_file_list(file_pattern)
    total_rewards = load_total_reward(file_pattern)
    total_rewards = np.array(total_rewards)
    mean_total_rewards = np.mean(total_rewards, axis=0)
    var_total_rewards = np.var(total_rewards, axis=0)
    standard_error = (var_total_rewards / len(file_list))**0.5
    return mean_total_rewards, standard_error


def get_asymptotic_performance(file_pattern, n_window=10, episode=200):
    asymptotic_performance = []
    file_list = get_file_list(file_pattern)
    for file_path in file_list:
        value_df = pd.read_csv(file_path, index_col=0)
        values = value_df.values.tolist()[episode - n_window: episode]
        asymptotic_performance.append(np.mean(values))
    return asymptotic_performance


def get_time_to_threshold(file_pattern, threshold, n_window=10, n_episodes=200):
    # 移動平均を取ったあとにTime2Thresholdを取得
    file_list = get_file_list(file_pattern)
    time_to_thresholds = []
    for file_path in file_list:
        value_df = pd.read_csv(file_path, index_col=0)
        value_df = value_df[:n_episodes]
        maveraged_df = value_df.rolling(window=n_window, min_periods=5).mean()
        v_list = maveraged_df.values.tolist()
        time_to_threshold = len(v_list)
        for step, row in enumerate(v_list[5:]):
            if row[0] <= threshold:
                time_to_threshold = step + 1
                break
        time_to_thresholds.append(time_to_threshold)
    return time_to_thresholds


def output_standard_error(file_pattern):
    total_rewards = load_total_reward(file_pattern)
    total_rewards = np.array(total_rewards)
    std_total_rewards = np.std(total_rewards, axis=0)
    ste_total_rewards = std_total_rewards / np.sqrt(total_rewards.shape[0])
    return ste_total_rewards


def output_time_to_threshold(file_pattern, threshold):
    total_rewards = load_total_reward(file_pattern)
    time_to_threshold = []
    moved_averaged = calc_moved_average(total_rewards)
    for total_reward in moved_averaged:
        flag = False
        for episode, reward in enumerate(total_reward):
            if threshold < reward:
                time_to_threshold.append(episode)
                flag = True
                break
        if episode == len(total_reward) - 1 and not flag:
            time_to_threshold.append(episode)
    return time_to_threshold


def output_asymptotic_performance(file_pattern, window=10):
    total_rewards = load_total_reward(file_pattern)
    asymptotic_performances = []
    for total_reward in total_rewards:
        t_reward = np.array(total_reward[len(total_reward)-window:])
        asymptotic_performances.append(np.mean(total_reward))
    return asymptotic_performances


def calc_moved_average(total_rewards, window=10):
    moved_average_total_rewards = []
    mean_total_rewards = np.mean(total_rewards, axis=0)
    for total_reward in total_rewards:
        moved_average_total_reward = []
        for episode, reward in enumerate(total_reward):
            if episode < window:
                moved_average = mean_total_rewards[episode]
            else:
                moved_average\
                    = sum(total_reward[episode - window: episode+1]) / window
            moved_average_total_reward.append(moved_average)
        moved_average_total_rewards.append(moved_average_total_reward)
    return moved_average_total_rewards


def main():
    argvs = sys.argv[1:]
    averaged_value = {}
    se_value = {}
    t2thres_3000 = {}
    t2thres_500 = {}
    t2thres_2000 = {}
    t2thres_1000 = {}
    asym_perf = {}
    for argv in argvs:
        logger.info("Loading...\n {}".format(argv))
        averaged_value[argv], se_value[argv] = average_values(argv)
        t2thres_3000[argv] = get_time_to_threshold(argv, 3000)
        t2thres_500[argv] = get_time_to_threshold(argv, 500)
        t2thres_2000[argv] = get_time_to_threshold(argv, 2000)
        t2thres_1000[argv] = get_time_to_threshold(argv, 1000)
        
        asym_perf[argv] = get_asymptotic_performance(argv, 10 ,200)
    out_dir = 'out'
    file_name = 'mean_ste_total_reward.csv'
    file_path = os.path.join(out_dir, file_name)
    logger.info("Exporting...")
    export(file_path, averaged_value)
    export(os.path.join(out_dir, "standard_error.csv"), se_value)
    export(os.path.join(out_dir, "time_to_threshold_3000.csv"), t2thres_3000)
    export(os.path.join(out_dir, "time_to_threshold_500.csv"), t2thres_500)
    export(os.path.join(out_dir, "time_to_threshold_2000.csv"), t2thres_2000)
    export(os.path.join(out_dir, "time_to_threshold_1000.csv"), t2thres_1000)
    export(os.path.join(out_dir, "asymptotic_performance.csv"), asym_perf)

if __name__ == "__main__":
    main()