import logging
import shaner
from src.agents.shaped import ShapedAgent
from src.achievers import RoomsAchiever

logger = logging.getLogger(__name__)


class SPRSAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(
            raw_agent, env, subgoals, config
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        value_func = self.raw_agent.get_value
        return shaner.SubgoalPulseRS(
            float(self.config["AGENT"]["discount"]),
            value_func,
            RoomsAchiever(
                float(self.config["SHAPING"]["_range"]),
                nfeatures, subgoals
            )
        )

    def info(self, state):
        # TODO potentialの取得をp_potentialの値に
        raw_agent_info = self.raw_agent.info(state)
        info = {
            "potential": self.reward_shaping.potential(state)
        }
        joined_info = {**raw_agent_info, **info}
        return joined_info
