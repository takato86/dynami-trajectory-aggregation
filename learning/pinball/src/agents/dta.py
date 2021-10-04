import shaner
from src.achievers import PinballAchiever
from src.agents.shaped import ShapedAgent


class DTAAgent(ShapedAgent):
    def _create_reward_shaping(self, env):
        return shaner.SarsaRS(
            self.config["agent"]["gamma"],
            self.config["agent"]["lr_theta"],
            env,
            self.config["shaping"]["aggr_id"],
            PinballAchiever(
                self.config["shaping"]["range"],
                env.observation_space.shape[0],
                self.subgoals
            ),
            self.config["shaping"]["vid"],
            is_success
        )


def is_success(done, info):
    return done
