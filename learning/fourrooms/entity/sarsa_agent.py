import numpy as np
import logging
from entity.policy import SoftmaxPolicy
from entity.tabular import Tabular
from entity.reward_shaping import NaiveSubgoalRewardShaping,\
                            DTARewardShaping,\
                            SarsaRewardShaping
import shaner
from utils.config import config
from entity.achiever import TworoomsAchiever

logger = logging.getLogger(__name__)


class SarsaAgent:
    def __init__(self, discount, epsilon, lr, nfeatures, nactions, temperature, rng, q_value={}):
        logger.debug("SarsaAgent is going to perform!")
        self.discount = discount
        self.epsilon = epsilon
        self.lr = lr
        self.nfeatures = nfeatures
        self.nactions = nactions
        self.temperature = temperature
        self.q_value = q_value
        self.policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
        self.critic = Sarsa(discount, lr, self.policy.weights)
        self.features = Tabular(nfeatures)
        self.total_shaped_reward = 0
        for state, value in q_value.items():
            phi = self.features(state)
            self.critic.initialize(phi, value)

    def act(self, state):
        return self.policy.sample(self.features(state))

    def update(self, state, action, next_state, reward, done, info):
        phi = self.features(state)
        next_phi = self.features(next_state)
        next_action = self.act(next_state)
        _ = self.critic.update(phi, action, next_phi, reward, done, next_action)

    def reset(self):
        rng = np.random.RandomState(np.random.randint(0, 100))
        self.policy = SoftmaxPolicy(rng, self.nfeatures, self.nactions, self.temperature)
        self.critic = Sarsa(self.discount, self.lr, self.policy.weights)
        self.features = Tabular(self.nfeatures)
        self.total_shaped_reward = 0
        for state, value in self.q_value.items():
            phi = self.features(state)
            self.critic.initialize(phi, value)

    def get_max_value(self):
        return np.amax(self.critic.weights)

    def get_min_value(self):
        return np.amin(self.critic.weights)

    def info(self, state):
        phi = self.features(state)
        return {
            "state": state,
            "v": max(self.critic.value(phi))
        }


class SubgoalRSSarsaAgent(SarsaAgent):
    def __init__(self, discount, epsilon, lr, nfeatures, nactions, temperature, rng, subgoals, eta, rho=0, subgoal_values=None):
        super().__init__(discount, epsilon, lr, nfeatures, nactions, temperature, rng)
        logger.debug("SubgoalRSSarsaAgent is going to perform!")
        self.subgoals = subgoals
        self.eta = eta
        self.rho = rho
        self.subgoal_values = subgoal_values
        self.reward_shaping = DTARewardShaping(
            discount, eta, subgoals, nfeatures, discount, lr
        )

    def update(self, state, action, next_state, reward, done, info):
        phi = self.features(state)
        next_phi = self.features(next_state)
        self.reward_shaping.fit(next_state, reward, done)
        reward += self.reward_shaping.value(next_state, done)
        next_action = self.act(next_state)
        self.total_shaped_reward += reward
        _ = self.critic.update(
            phi, action, next_phi, reward, done, next_action
        )

    def reset(self):
        rng = np.random.RandomState(np.random.randint(0, 100))
        self.__init__(self.discount, self.epsilon, self.lr, self.nfeatures, self.nactions,
                      self.temperature, rng, self.subgoals, self.eta, self.rho, self.subgoal_values)

    def info(self, state):
        phi = self.features(state)
        return {
            "state": state,
            "v_z": self.reward_shaping.subgoal_values[self.reward_shaping.curr_index[0]][self.reward_shaping.curr_index[1]],
            "v": max(self.critic.value(phi))
        }


class NaiveSubgoalSarsaAgent(SubgoalRSSarsaAgent):
    def __init__(self, discount, epsilon, lr, nfeatures, nactions, temperature, rng, subgoals, eta, rho=0, subgoal_values=None):
        super().__init__(discount, epsilon, lr, nfeatures, nactions, temperature, rng, subgoals, eta, rho=0,
                         subgoal_values=subgoal_values)
        self.reward_shaping = NaiveSubgoalRewardShaping(discount, eta, subgoals, nfeatures)


class Sarsa:
    def __init__(self, discount, lr, weights):
        self.lr = lr
        self.discount = discount
        self.weights = weights

    def initialize(self, phi, q_value, action=None):
        if action is None:
            self.weights[phi, :] = q_value
        else:
            self.weights[phi, action] = q_value

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def advantage(self, phi, action=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if action is None:
            return advantages
        return advantages[action]

    def update(self, phi, action, next_phi, reward, done, next_action):
        # One-step update target
        update_target = reward
        if not done:
            next_values = self.value(next_phi)
            update_target += self.discount * next_values[next_action]
        # Dense gradient update step
        tderror = update_target - self.value(phi, action)
        self.weights[phi, action] += self.lr * tderror
        return update_target


class SarsaRSSarsaAgent(SarsaAgent):
    def __init__(self, discount, epsilon, lr, nfeatures, nactions, temperature, rng, aggr_set):
        logger.debug("SarsaRSSarsaAgent is going to perform!")
        super().__init__(
            discount, epsilon, lr, nfeatures, nactions, temperature, rng
        )
        self.aggr_set = aggr_set
        self.reward_shaping = SarsaRewardShaping(
            discount, nfeatures, discount, lr, aggr_set
        )

    def update(self, state, action, next_state, reward, done, info):
        phi = self.features(state)
        next_phi = self.features(next_state)
        self.reward_shaping.fit(next_state, reward, done)
        reward += self.reward_shaping.value(next_state, done)
        next_action = self.act(next_state)
        self.total_shaped_reward += reward
        _ = self.critic.update(
            phi, action, next_phi, reward, done, next_action
        )

    def reset(self):
        rng = np.random.RandomState(np.random.randint(0, 100))
        self.__init__(self.discount, self.epsilon, self.lr, self.nfeatures,
                      self.nactions, self.temperature, rng, self.aggr_set)


class ShapingSarsaAgent(SarsaAgent):
    def __init__(self, discount, epsilon, lr, env, temperature, rng, subgoals):
        nfeatures = env.observation_space.n
        nactions = env.action_space.n
        super().__init__(
            discount, epsilon, lr, nfeatures, nactions, temperature, rng
        )
        self.reward_shaping = self._generate_shaping(env, subgoals)

    def _generate_shaping(self, env, subgoals):
        raise NotImplementedError

    def update(self, state, action, next_state, reward, done, info):
        F = self.reward_shaping.perform(
            state, next_state, reward, done, info
        )
        if np.random.rand() < 0.001:
            logger.debug("shaping reward: {}".format(F))
        super().update(state, action, next_state, reward + F, done, info)

    def reset(self):
        super().reset()
        self.reward_shaping.reset()

    def info(self, state):
        phi = self.features(state)
        return {
            "state": state,
            "v_z": self.reward_shaping.potential(
                        self.reward_shaping.aggregater.current_state
                   ),
            "v": max(self.critic.value(phi))
        }


class SRSSarsaAgent(ShapingSarsaAgent):
    def __init__(self, discount, epsilon, lr, env, temperature, rng, subgoals):
        super().__init__(
            discount, epsilon, lr, env, temperature, rng, subgoals
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        params = {
            'vid': config["SHAPING"]["vid"],
            'aggr_id': config["SHAPING"]["aggr_id"],
            'eta': float(config["SHAPING"]["eta"]),
            'rho': float(config["SHAPING"]["rho"]),
            'params': {
                'achiever': TworoomsAchiever(
                                float(config["SHAPING"]["_range"]),
                                nfeatures, subgoals
                            )
            }
        }
        return shaner.SubgoalRS(
            float(config["AGENT"]["lr"]), float(config["AGENT"]["discount"]), env, params
        )


class DTAAgent(ShapingSarsaAgent):
    def __init__(self, discount, epsilon, lr, env, temperature, rng, subgoals):
        super().__init__(
            discount, epsilon, lr, env, temperature, rng, subgoals
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        params = {
            'vid': config["SHAPING"]["vid"],
            'aggr_id': config["SHAPING"]["aggr_id"],
            'params': {
                'achiever': TworoomsAchiever(
                                float(config["SHAPING"]["_range"]),
                                nfeatures, subgoals
                            )
            },
        }
        return shaner.SarsaRS(
            float(config["AGENT"]["lr"]), float(config["AGENT"]["discount"]),
            env, params, is_success
        )


def is_success(done, info):
    return done
