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
            float(self.config["AGENT"]["lr"]),
            float(self.config["AGENT"]["discount"]),
            env, self.config["SHAPING"]["aggr_id"],
            RoomsAchiever(
                float(self.config["SHAPING"]["_range"]),
                nfeatures, subgoals
            ),
            self.config["SHAPING"]["vid"],
            is_success
        )


def is_success(done, info):
    return done
