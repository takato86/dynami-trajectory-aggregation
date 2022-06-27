import logging
import shaner
from src.agents.shaped import ShapedAgent
from src.achievers import RoomsAchiever

logger = logging.getLogger(__name__)


class NaiveRSAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(
            raw_agent, env, subgoals, config
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        return shaner.NaiveSRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["AGENT"]["lr"]),
            float(self.config["SHAPING"]["eta"]),
            float(self.config["SHAPING"]["rho"]),
            self.config["SHAPING"]["aggr_id"],
            RoomsAchiever(
                float(self.config["SHAPING"]["_range"]),
                nfeatures, subgoals
            )
        )
