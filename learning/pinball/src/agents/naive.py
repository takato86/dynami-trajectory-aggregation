import shaner
from src.agents.shaped import ShapedAgent
from src.achievers import PinballAchiever


class NaiveRSAgent(ShapedAgent):
    def _create_reward_shaping(self, env):
        return shaner.NaiveSRS(
            self.config["agent"]["gamma"],
            self.config["agent"]["lr_theta"],
            self.config["shaping"]["eta"],
            self.config["shaping"]["rho"],
            self.config["shaping"]["aggr_id"],
            PinballAchiever(
                self.config["shaping"]["range"],
                env.observation_space.shape[0],
                self.subgoals
            )
        )
