import shaner
from src.agents.shaped import ShapedAgent
from src.achievers import RoomsAchiever


class DTAAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(
            raw_agent, env, subgoals, config
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        return shaner.SarsaRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["AGENT"]["lr"]),
            env, self.config["SHAPING"]["aggr_id"],
            RoomsAchiever(
                float(self.config["SHAPING"]["_range"]),
                nfeatures, subgoals
            ),
            self.config["SHAPING"]["vid"],
            is_success
        )

    def info(self, state):
        super_info = super().info(state)
        info = {
            "z": self.reward_shaping.get_current_state()
        }
        joined_info = {**super_info, **info}
        return joined_info


def is_success(done, info):
    return done
