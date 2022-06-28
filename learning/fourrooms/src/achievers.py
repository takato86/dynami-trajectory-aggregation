from shaper.achiever import AbstractAchiever


class RoomsAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals, **params):
        self.__subgoals = subgoals[0]

    @property
    def subgoals(self):
        return self.__subgoals

    def eval(self, obs, current_state):
        if len(self.__subgoals) <= current_state:
            return False
        subgoal = self.__subgoals[current_state]
        return obs == subgoal
