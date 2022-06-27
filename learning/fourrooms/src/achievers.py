from shaner.aggregater.entity.achiever import AbstractAchiever


class RoomsAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals, **params):
        super().__init__(_range, n_obs)
        self.subgoals = subgoals[0]

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = self.subgoals[current_state]
        return obs == subgoal
