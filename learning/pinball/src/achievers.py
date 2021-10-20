from shaner.aggregater import AbstractAchiever
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PinballAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals):
        super().__init__(_range, n_obs)
        self.subgoals = subgoals  # 2d-ndarray shape(#obs, #subgoals)

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False

        subgoal = np.array(self.subgoals[current_state])
        idxs = np.argwhere(subgoal == subgoal)  # np.nanでない要素を取り出し
        b_in = l2_norm_dist(
            subgoal[idxs].reshape(-1),
            obs[idxs].reshape(-1)
        ) <= self._range
        res = np.all(b_in)

        if res:
            logger.debug("Achieve the subgoal{}".format(current_state))
        return res


def l2_norm_dist(x_arr, y_arr):
    return np.linalg.norm(x_arr - y_arr, ord=2)
