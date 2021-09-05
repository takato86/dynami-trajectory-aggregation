import numpy as np
import logging
import csv
import copy
from .tabular import Tabular

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PotentialBasedShapingReward:
    def __init__(self, discount, nfeatures):
        self.discount = discount
        self.features = Tabular(nfeatures)

    def fit(self, state, reward):
        return NotImplementedError

    def value(self, state, done):
        # self.discount * cur_potential - pre_potential
        return NotImplementedError


class SubgoalReward:
    def __init__(self, discount, eta, subgoal_serieses, nfeatures):
        self.discount = discount
        self.eta = eta
        self.subgoal_serieses = subgoal_serieses
        self.curr_subgoal_series = {}
        self.curr_index = 0
        self.curr_val = 0
        self.features = Tabular(nfeatures)

    def done(self, state):
        if len(self.curr_subgoal_series) == 0:
            for subgoal_series in self.subgoal_serieses:
                if state in subgoal_series:
                    return True
        elif self.curr_index < len(self.curr_subgoal_series):
            if state == self.curr_subgoal_series[self.curr_index]:
                return True
        return False
    
    def fit(self, state, reward, done):
        pass


class NaiveSubgoalRewardShaping(SubgoalReward):
    """サブゴールに到達したときにηの報酬を生成する関数、二度目の訪問には報酬は生成しない
    
    Raises:
        Exception: [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self, discount, eta, subgoal_serieses, nfeatures):
        logger.debug("SubgoalOnceRewardShaping!")
        super().__init__(discount, eta, subgoal_serieses, nfeatures)

    def value(self, state, done):
        last_val = self.curr_val
        self.curr_val = 0
        if self.curr_subgoal_series is None:
            for subgoal_series in self.subgoal_serieses:
                if state == subgoal_series[0]:
                    logger.debug(f"Hit subgoal {state}")
                    self.curr_subgoal_series = copy.copy(subgoal_series)
                    del self.curr_subgoal_series[0]
                    self.curr_val = self.eta
                    logger.debug(f"subgoal series {self.curr_subgoal_series}")
                    logger.debug(f"Current Potential: {self.curr_val}, Previous Potential: {last_val}")
        else:
            if len(self.curr_subgoal_series) > 0:
                if state == self.curr_subgoal_series[0]:
                    logger.debug(f"Hit subgoal {state}")
                    self.curr_val = self.eta
                    del self.curr_subgoal_series[0]
                    logger.debug(f"subgoal series {self.curr_subgoal_series}")
                    logger.debug(f"Current Potential: {self.curr_val}, Previous Potential: {last_val}")

        additional_reward = self.discount * self.curr_val - last_val

        if done:
            self.curr_subgoal_series = None
            self.curr_val = 0
            self.curr_index = 0
        
        return additional_reward


class DTARewardShaping(SubgoalReward):
    def __init__(self, discount, eta, subgoal_serieses, nfeatures, discount_v, lr_v, subgoal_values=None):
        logger.debug("DTARewardShaping!")
        super().__init__(discount, eta, subgoal_serieses, nfeatures)
        self.discount_v = discount_v
        self.lr_v = lr_v
        self.subgoal_values = [[0]]
        if subgoal_values is None:
            logger.debug("Subgoal Values are set by 0")
            self.subgoal_values += [[0 for s in subgoal_series] for subgoal_series in subgoal_serieses]
        else:
            logger.debug("Subgoal Values are set by {}".format(subgoal_values))
            self.subgoal_values += subgoal_values
        self.curr_index = (0, 0)
        self.pre_index = (0, 0)
        self.time = 0
        self.subgoal_serieses = [[]] + self.subgoal_serieses

    def value(self, state, done):
        potential = self.subgoal_values[self.curr_index[0]][self.curr_index[1]]
        pre_potential = self.subgoal_values[self.pre_index[0]][self.pre_index[1]]
        shaping_reward = self.discount * potential - pre_potential
        if done:
            self.curr_index = (0, 0)
            self.time = 0
        if shaping_reward != 0:
            logger.debug(f"additional reward: {shaping_reward}")
            logger.debug(f"potential value: {self.curr_val}")
        return shaping_reward

    def fit(self, state, reward, done):
        self.pre_index = self.curr_index
        self.curr_index = self.aggregate(state, done)
        h_reward = self.get_h_reward(state, reward, done, self.time)
        if self.pre_index != self.curr_index or h_reward != 0:
            logger.debug(f"pre: {self.pre_index}, cur: {self.curr_index}, h_reward: {h_reward}")
            pre_y, pre_x = self.pre_index
            cur_y, cur_x = self.curr_index
            # if h_reward != 0:
            if reward > 0:
                self.subgoal_values[pre_y][pre_x] \
                    = (1 - self.lr_v) * self.subgoal_values[pre_y][pre_x]\
                    + self.lr_v * h_reward
            else:
                self.subgoal_values[pre_y][pre_x] \
                    = (1 - self.lr_v) * self.subgoal_values[pre_y][pre_x]\
                    + self.lr_v * (h_reward + self.discount_v ** (self.time + 1) * self.subgoal_values[cur_y][cur_x])
            self.time = 0
        else:
            self.time += 1
    
    def aggregate(self, state, done):
        if self.curr_index == (0, 0):
            # サブゴールを1つも発見していない状況
            for y, subgoal_series in enumerate(self.subgoal_serieses[1:]):
                if state == subgoal_series[0]:
                    logger.debug(f"Hit subgoal {state}")
                    return (y + 1, 0)
        else:
        # elif self.curr_k < len(self.curr_subgoal_series):
            # サブゴールがまだある
            if state in self.subgoal_serieses[self.curr_index[0]]:
                x = self.subgoal_serieses[self.curr_index[0]].index(state)
                # サブゴールが進行していればcurr_kを更新
                if x == (self.curr_index[1] + 1):
                    logger.debug(f"Hit subgoal at {state}, index: {self.curr_index}.")
                    return (self.curr_index[0], x)
                    # 到達したらタイムステップはリセット
        return self.curr_index

    def get_h_reward(self, state, reward, done, time):
        y, x = self.curr_index
        if y == 0:
            return 0
        if len(self.subgoal_serieses[y]) <= x + 1:
            # すべてのサブゴールを達成してからゴール
            return self.discount_v ** time * reward
        else:
            return 0


# Online learning of shaping rewards in reinforcement learning
# Grzes M, Kudenko D
# 10.1016/j.neunet.2010.01.001
class SarsaRewardShaping(PotentialBasedShapingReward):
    def __init__(self, discount, nfeatures, discount_v, lr_v, aggr_set):
        super().__init__(discount, nfeatures)
        self.discount_v = discount_v
        self.lr_v = lr_v
        self.subgoal_values = [0 for _ in range(len(aggr_set))]
        # self.subgoal_values += [[0 for s in subgoal_series] for subgoal_series in subgoal_serieses]
        self.to_z = self.trans(aggr_set)
        self.curr_z = 0
        self.pre_z = 0
        self.time = 0
    
    def trans(self, aggr_set):
        aggregation_dict = {}
        for i, aggr_states in enumerate(aggr_set):
            for state in aggr_states:
                aggregation_dict[state] = i
        return aggregation_dict
        
    def value(self, state, done):
        potential = self.subgoal_values[self.curr_z]
        pre_potential = self.subgoal_values[self.pre_z]
        shaping_reward = self.discount * potential - pre_potential
        if done:
            self.curr_z = 0
            self.time = 0
        if shaping_reward != 0:
            logger.debug(f"additional reward: {shaping_reward}")
            logger.debug(f"potential value: {potential}")
        return shaping_reward

    def fit(self, state, reward, done):
        self.time += 1
        self.pre_z = self.curr_z
        self.curr_z = self.to_z[state]
        logger.debug(f"previous: {self.pre_z}, current: {self.curr_z}")
        if self.pre_z != self.curr_z or reward != 0:
            self.subgoal_values[self.pre_z] \
                = (1 - self.lr_v) * self.subgoal_values[self.pre_z]\
                  + self.lr_v * (reward + self.discount_v ** self.time * self.subgoal_values[self.curr_z])
            self.time = 0
