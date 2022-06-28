import logging
import shaper
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation
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
        achiever = RoomsAchiever(
            float(self.config["SHAPING"]["_range"]),
            nfeatures, subgoals
        )
        aggregator = DynamicTrajectoryAggregation(achiever)
        return shaper.NaiveSRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["SHAPING"]["eta"]),
            aggregator
        )
